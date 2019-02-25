# -*- coding: utf-8 -*-

# This package is a translation of a part of the BOSARIS toolkit.
# The authors thank Niko Brummer and Agnitio for allowing them to
# translate this code and provide the community with efficient structures
# and tools.
#
# The BOSARIS Toolkit is a collection of functions and classes in Matlab
# that can be used to calibrate, fuse and plot scores from speaker recognition
# (or other fields in which scores are used to test the hypothesis that two
# samples are from the same source) trials involving a model and a test segment.
# The toolkit was written at the BOSARIS2010 workshop which took place at the
# University of Technology in Brno, Czech Republic from 5 July to 6 August 2010.
# See the User Guide (available on the toolkit website)1 for a discussion of the
# theory behind the toolkit and descriptions of some of the algorithms used.
#
# The BOSARIS toolkit in MATLAB can be downloaded from `the website
# <https://sites.google.com/site/bosaristoolkit/>`_.

"""
This is the 'key' module
"""
import numpy
import sys
import h5py
import logging
from sidekit.bosaris.ndx import Ndx
from sidekit.sidekit_wrappers import check_path_existance

__author__ = "Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brummer", "Edward de Villiers"]


def diff(list1, list2):
    c = [item for item in list1 if item not in list2]
    c.sort()
    return c


def ismember(list1, list2):
    c = [item in list2 for item in list1]
    return c


class Key:
    """A class for representing a Key i.e. it classifies trials as                                                          
    target or non-target trials.

    :attr modelset: list of the models into a ndarray of strings
    :attr segset: list of the test segments into a ndarray of strings
    :attr tar: 2D ndarray of booleans which rows correspond to the models 
            and columns to the test segments. True if target trial.
    :attr non: 2D ndarray of booleans which rows correspond to the models 
            and columns to the test segments. True is non-target trial.
    """

    def __init__(self, key_file_name='',
                 models=numpy.array([]),
                 testsegs=numpy.array([]),
                 trials=numpy.array([])):
        """Initialize a Key object.
        :param key_file_name: name of the file to load. Default is ''.
        :param models: a list of models
        :param testsegs: a list of test segments
        
        In case the key_file_name is empty, initialize an empty Key object.
        """
        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.tar = numpy.array([], dtype="bool")
        self.non = numpy.array([], dtype="bool")

        if key_file_name == '':
            modelset = numpy.unique(models)
            segset = numpy.unique(testsegs)
    
            tar = numpy.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
            non = numpy.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")

            for idx_m, model in enumerate(modelset):
                idx_current_model = numpy.argwhere(models == model).flatten()
                current_model_keys = dict(zip(testsegs[idx_current_model], 
                                              trials[idx_current_model]))
                for idx_s, seg in enumerate(segset):
                    if seg in current_model_keys:
                        tar[idx_m, idx_s] = (current_model_keys[seg] == 'target')
                        non[idx_m, idx_s] = (current_model_keys[seg] == 'nontarget')
    
            self.modelset = modelset
            self.segset = segset
            self.tar = tar
            self.non = non
            assert self.validate(), "Wrong Key format"            

        else:
            tmp = self.read(key_file_name)
            self.modelset = tmp.modelset
            self.segset = tmp.segset
            self.tar = tmp.tar
            self.non = tmp.non

    @check_path_existance
    def write(self, output_file_name):
        """ Save Key in HDF5 format

        :param output_file_name: name of the file to write to
        """
        assert self.validate(), "Error: wrong Key format"

        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("modelset", data=self.modelset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("segset", data=self.segset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            trialmask = numpy.array(self.tar, dtype='int8') - numpy.array(self.non, dtype='int8')
            f.create_dataset("trial_mask", data=trialmask,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)

    @check_path_existance
    def write_txt(self, output_file_name):
        """Save a Key object to a text file.

        :param output_file_name: name of the output text file
        """
        fid = open(output_file_name, 'w')
        for m in range(self.modelset.shape[0]):
            segs = self.segset[self.tar[m, ]]
            for s in range(segs.shape[0]):
                fid.write('{} {} {}\n'.format(self.modelset[m], segs[s], 'target'))
            segs = self.segset[self.non[m, ]]
            for s in range(segs.shape[0]):
                fid.write('{} {} {}\n'.format(self.modelset[m], segs[s], 'nontarget'))
        fid.close()

    def filter(self, modlist, seglist, keep):
        """Removes some of the information in a key.  Useful for creating a
        gender specific key from a pooled gender key.  Depending on the
        value of \'keep\', the two input lists indicate the strings to
        retain or the strings to discard.

        :param modlist: a cell array of strings which will be compared with
            the modelset of 'inkey'.
        :param seglist: a cell array of strings which will be compared with
            the segset of 'inkey'.
        :param keep: a boolean indicating whether modlist and seglist are the
            models to keep or discard.

        :return: a filtered version of 'inkey'.
        """
        if keep:
            keepmods = modlist
            keepsegs = seglist
        else:
            keepmods = diff(self.modelset, modlist)
            keepsegs = diff(self.segset, seglist)

        keepmodidx = numpy.array(ismember(self.modelset, keepmods))
        keepsegidx = numpy.array(ismember(self.segset, keepsegs))

        outkey = Key()
        outkey.modelset = self.modelset[keepmodidx]
        outkey.segset = self.segset[keepsegidx]
        tmp = self.tar[numpy.array(keepmodidx), :]
        outkey.tar = tmp[:, numpy.array(keepsegidx)]
        tmp = self.non[numpy.array(keepmodidx), :]
        outkey.non = tmp[:, numpy.array(keepsegidx)]

        assert(outkey.validate())

        if self.modelset.shape[0] > outkey.modelset.shape[0]:
            logging.info('Number of models reduced from %d to %d', self.modelset.shape[0], outkey.modelset.shape[0])
        if self.segset.shape[0] > outkey.segset.shape[0]:
            logging.info('Number of test segments reduced from %d to %d', self.segset.shape[0], outkey.segset.shape[0])
        return outkey

    def to_ndx(self):
        """Create a Ndx object based on the Key object

        :return: a Ndx object based on the Key
        """
        ndx = Ndx()
        ndx.modelset = self.modelset
        ndx.segset = self.segset
        ndx.trialmask = self.tar | self.non
        return ndx

    def validate(self):
        """Checks that an object of type Key obeys certain rules that
        must always be true.

        :return: a boolean value indicating whether the object is valid.
        """
        ok = isinstance(self.modelset, numpy.ndarray)
        ok &= isinstance(self.segset, numpy.ndarray)
        ok &= isinstance(self.tar, numpy.ndarray)
        ok &= isinstance(self.non, numpy.ndarray)
        ok &= self.modelset.ndim == 1
        ok &= self.segset.ndim == 1
        ok &= self.tar.ndim == 2
        ok &= self.non.ndim == 2
        ok &= self.tar.shape == self.non.shape
        ok &= self.tar.shape[0] == self.modelset.shape[0]
        ok &= self.tar.shape[1] == self.segset.shape[0]
        return ok

    @staticmethod
    def read(input_file_fame):
        """Reads a Key object from an hdf5 file.
  
        :param input_file_fame: name of the file to read from
        """
        with h5py.File(input_file_fame, "r") as f:

            key = Key()
            key.modelset = f.get("modelset").value
            key.segset = f.get("segset").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                key.modelset = key.modelset.astype('U100', copy=False)
                key.segset = key.segset.astype('U100', copy=False)

            trialmask = f.get("trial_mask").value
            key.tar = (trialmask == 1)
            key.non = (trialmask == -1)

            assert key.validate(), "Error: wrong Key format"
            return key

    @staticmethod
    def read_txt(input_file_name):
        """Creates a Key object from information stored in a text file.

            :param input_file_name: name of the file to read from
        """
        key = Key()

        models, testsegs, trial = numpy.loadtxt(input_file_name,
                                                delimiter=' ',
                                                dtype={'names': ('mod', 'seg', 'key'),
                                                       'formats': ('S1000', 'S1000', 'S10')},
                                                unpack=True)

        models = models.astype('|O', copy=False).astype('S', copy=False)
        testsegs = testsegs.astype('|O', copy=False).astype('S', copy=False)
        trial = trial.astype('|O', copy=False).astype('S', copy=False)

        if sys.version_info[0] == 3:
            models = models.astype('U', copy=False)
            testsegs = testsegs.astype('U', copy=False)
            trial = trial.astype('U', copy=False)

        modelset = numpy.unique(models)
        segset = numpy.unique(testsegs)

        tar = numpy.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
        non = numpy.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")

        for idx_m, model in enumerate(modelset):
            idx_current_model = numpy.argwhere(models == model).flatten()
            current_model_keys = dict(zip(testsegs[idx_current_model], trial[idx_current_model]))
            for idx_s, seg in enumerate(segset):
                if seg in current_model_keys:
                    tar[idx_m, idx_s] = (current_model_keys[seg] == 'target')
                    non[idx_m, idx_s] = (current_model_keys[seg] == 'nontarget')

        key.modelset = modelset
        key.segset = segset
        key.tar = tar
        key.non = non
        assert key.validate(), "Wrong Key format"
        return key

    def merge(self, key_list):
        """Merges Key objects. This function takes as input a list of
        Key objects to merge in the curent one.

        :param key_list: the list of Keys to merge
        """
        # the output key must have all models and segment in the input
        # keys (only once) and the same target and non-target trials.
        # It is an error if a trial is a target in one key and a
        # non-target in another, but a target or non-target marker will
        # override a 'non-trial' marker.
        assert isinstance(key_list, list), "Input is not a list"
        for key in key_list:
            assert isinstance(key_list, list), \
                    '{} {} {}'.format("Element ", key, " is not a list")

        for key2 in key_list:
            key_new = Key()
            key1 = self

            # create new ndx with empty masks
            key_new.modelset = numpy.union1d(key1.modelset, key2.modelset)
            key_new.segset = numpy.union1d(key1.segset, key2.segset)

            # expand ndx1 mask
            tar_1 = numpy.zeros((key_new.modelset.shape[0],
                                key_new.segset.shape[0]),
                                dtype="bool")
            non_1 = numpy.zeros((key_new.modelset.shape[0],
                                key_new.segset.shape[0]), dtype="bool")
            model_index_a = numpy.argwhere(numpy.in1d(key_new.modelset, key1.modelset))
            model_index_b = numpy.argwhere(numpy.in1d(key1.modelset, key_new.modelset))
            seg_index_a = numpy.argwhere(numpy.in1d(key_new.segset, key1.segset))
            seg_index_b = numpy.argwhere(numpy.in1d(key1.segset, key_new.segset))
            tar_1[model_index_a[:, None], seg_index_a] = key1.tar[model_index_b[:, None], seg_index_b]
            non_1[model_index_a[:, None], seg_index_a] = key1.non[model_index_b[:, None], seg_index_b]

            # expand ndx2 mask
            tar_2 = numpy.zeros((key_new.modelset.shape[0],
                                key_new.segset.shape[0]), dtype="bool")
            non_2 = numpy.zeros((key_new.modelset.shape[0],
                                key_new.segset.shape[0]), dtype="bool")
            model_index_a = numpy.argwhere(numpy.in1d(key_new.modelset, key2.modelset))
            model_index_b = numpy.argwhere(numpy.in1d(key2.modelset, key_new.modelset))
            seg_index_a = numpy.argwhere(numpy.in1d(key_new.segset, key2.segset))
            seg_index_b = numpy.argwhere(numpy.in1d(key2.segset, key_new.segset))
            tar_2[model_index_a[:, None], seg_index_a] = key2.tar[model_index_b[:, None], seg_index_b]
            non_2[model_index_a[:, None], seg_index_a] = key2.non[model_index_b[:, None], seg_index_b]

            # merge masks
            tar = tar_1 | tar_2
            non = non_1 | non_2

            # check for clashes
            assert numpy.sum(tar & non) == 0, "Conflict in the new Key"

            # build new key
            key_new.tar = tar
            key_new.non = non
            self.modelset = key_new.modelset
            self.segset = key_new.segset
            self.tar = key_new.tar
            self.non = key_new.non
            self.validate()
