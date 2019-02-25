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
This is the 'ndx' module
"""
import h5py
import logging
import numpy
import sys
from sidekit.sidekit_wrappers import check_path_existance, deprecated

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


class Ndx:
    """A class that encodes trial index information.  It has a list of
    model names and a list of test segment names and a matrix
    indicating which combinations of model and test segment are
    trials of interest.
    
    :attr modelset: list of unique models in a ndarray
    :attr segset:  list of unique test segments in a ndarray
    :attr trialmask: 2D ndarray of boolean. Rows correspond to the models 
            and columns to the test segments. True if the trial is of interest.
    """

    def __init__(self, ndx_file_name='',
                 models=numpy.array([]),
                 testsegs=numpy.array([])):
        """Initialize a Ndx object by loading information from a file
        in HDF5 or text format.

        :param ndx_file_name: name of the file to load
        """
        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.trialmask = numpy.array([], dtype="bool")

        if ndx_file_name == '':
            modelset = numpy.unique(models)
            segset = numpy.unique(testsegs)
    
            trialmask = numpy.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
            for m in range(modelset.shape[0]):
                segs = testsegs[numpy.array(ismember(models, modelset[m]))]
                trialmask[m, ] = ismember(segset, segs)
    
            self.modelset = modelset
            self.segset = segset
            self.trialmask = trialmask
            assert self.validate(), "Wrong Ndx format"

        else:
            ndx = Ndx.read(ndx_file_name)
            self.modelset = ndx.modelset
            self.segset = ndx.segset
            self.trialmask = ndx.trialmask

    @check_path_existance
    def write(self, output_file_name):
        """ Save Ndx object in HDF5 format

        :param output_file_name: name of the file to write to
        """
        assert self.validate(), "Error: wrong Ndx format"

        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("modelset", data=self.modelset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("segset", data=self.segset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("trial_mask", data=self.trialmask.astype('int8'),
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)

    @check_path_existance
    def save_txt(self, output_file_name):

        """Save a Ndx object in a text file

        :param output_file_name: name of the file to write to
        """
        fid = open(output_file_name, 'w')
        for m in range(self.modelset.shape[0]):
            segs = self.segset[self.trialmask[m, ]]
            for s in segs:
                fid.write('{} {}\n'.format(self.modelset[m], s))
        fid.close()

    def filter(self, modlist, seglist, keep):
        """Removes some of the information in an Ndx. Useful for creating a
        gender specific Ndx from a pooled gender Ndx.  Depending on the
        value of \'keep\', the two input lists indicate the strings to
        retain or the strings to discard.

        :param modlist: a cell array of strings which will be compared with
                the modelset of 'inndx'.
        :param seglist: a cell array of strings which will be compared with
                the segset of 'inndx'.
        :param keep: a boolean indicating whether modlist and seglist are the
                models to keep or discard.

        :return: a filtered version of the current Ndx object.
        """
        if keep:
            keepmods = modlist
            keepsegs = seglist
        else:
            keepmods = diff(self.modelset, modlist)
            keepsegs = diff(self.segset, seglist)

        keepmodidx = numpy.array(ismember(self.modelset, keepmods))
        keepsegidx = numpy.array(ismember(self.segset, keepsegs))

        outndx = Ndx()
        outndx.modelset = self.modelset[keepmodidx]
        outndx.segset = self.segset[keepsegidx]
        tmp = self.trialmask[numpy.array(keepmodidx), :]
        outndx.trialmask = tmp[:, numpy.array(keepsegidx)]

        assert outndx.validate, "Wrong Ndx format"

        if self.modelset.shape[0] > outndx.modelset.shape[0]:
            logging.info('Number of models reduced from %d to %d', self.modelset.shape[0], outndx.modelset.shape[0])
        if self.segset.shape[0] > outndx.segset.shape[0]:
            logging.info('Number of test segments reduced from %d to %d', self.segset.shape[0], outndx.segset.shape[0])
        return outndx

    def validate(self):
        """Checks that an object of type Ndx obeys certain rules that
        must always be true.

        :return: a boolean value indicating whether the object is valid
        """
        ok = isinstance(self.modelset, numpy.ndarray)
        ok &= isinstance(self.segset, numpy.ndarray)
        ok &= isinstance(self.trialmask, numpy.ndarray)

        ok &= (self.modelset.ndim == 1)
        ok &= (self.segset.ndim == 1)
        ok &= (self.trialmask.ndim == 2)

        ok &= (self.trialmask.shape == (self.modelset.shape[0], self.segset.shape[0]))
        return ok

    @staticmethod
    def read(input_file_name):
        """Creates an Ndx object from the information in an hdf5 file.

        :param input_file_name: name of the file to read from
        """
        with h5py.File(input_file_name, "r") as f:
            ndx = Ndx()
            ndx.modelset = f.get("modelset").value
            ndx.segset = f.get("segset").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                ndx.modelset = ndx.modelset.astype('U100', copy=False)
                ndx.segset = ndx.segset.astype('U100', copy=False)

            ndx.trialmask = f.get("trial_mask").value.astype('bool')

            assert ndx.validate(), "Error: wrong Ndx format"
            return ndx

    @classmethod
    @check_path_existance
    def read_txt(cls, input_filename):
        """Creates an Ndx object from information stored in a text file.

        :param input_filename: name of the file to read from
        """
        ndx = Ndx()

        with open(input_filename, 'r') as fid:
            lines = [l.rstrip().split() for l in fid]

        models = numpy.empty(len(lines), '|O')
        testsegs = numpy.empty(len(lines), '|O')
        for ii in range(len(lines)):
            models[ii] = lines[ii][0]
            testsegs[ii] = lines[ii][1]

        modelset = numpy.unique(models)
        segset = numpy.unique(testsegs)

        trialmask = numpy.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
        for m in range(modelset.shape[0]):
            segs = testsegs[numpy.array(ismember(models, modelset[m]))]
            trialmask[m, ] = ismember(segset, segs)

        ndx.modelset = modelset
        ndx.segset = segset
        ndx.trialmask = trialmask

        assert ndx.validate(), "Wrong Ndx format"
        return ndx

    def merge(self, ndx_list):
        """Merges a list of Ndx objects into the current one.
        The resulting ndx must have all models and segment in the input
        ndxs (only once).  A trial in any ndx becomes a trial in the
        output ndx

        :param ndx_list: list of Ndx objects to merge
        """
        assert isinstance(ndx_list, list), "Input is not a list"
        for ndx in ndx_list:
            assert isinstance(ndx_list, list), \
                '{} {} {}'.format("Element ", ndx, " is not an Ndx")

        self.validate()
        for ndx2 in ndx_list:
            ndx_new = Ndx()
            ndx1 = self

            # create new ndx with empty masks
            ndx_new.modelset = numpy.union1d(ndx1.modelset, ndx2.modelset)
            ndx_new.segset = numpy.union1d(ndx1.segset, ndx2.segset)

            # expand ndx1 mask
            trials_1 = numpy.zeros((ndx_new.modelset.shape[0], ndx_new.segset.shape[0]), dtype="bool")
            model_index_a = numpy.argwhere(numpy.in1d(ndx_new.modelset, ndx1.modelset))
            model_index_b = numpy.argwhere(numpy.in1d(ndx1.modelset, ndx_new.modelset))
            seg_index_a = numpy.argwhere(numpy.in1d(ndx_new.segset, ndx1.segset))
            seg_index_b = numpy.argwhere(numpy.in1d(ndx1.segset, ndx_new.segset))
            trials_1[model_index_a[:, None], seg_index_a] = ndx1.trialmask[model_index_b[:, None], seg_index_b]

            # expand ndx2 mask
            trials_2 = numpy.zeros((ndx_new.modelset.shape[0], ndx_new.segset.shape[0]), dtype="bool")
            model_index_a = numpy.argwhere(numpy.in1d(ndx_new.modelset, ndx2.modelset))
            model_index_b = numpy.argwhere(numpy.in1d(ndx2.modelset, ndx_new.modelset))
            seg_index_a = numpy.argwhere(numpy.in1d(ndx_new.segset, ndx2.segset))
            seg_index_b = numpy.argwhere(numpy.in1d(ndx2.segset, ndx_new.segset))
            trials_2[model_index_a[:, None], seg_index_a] = ndx2.trialmask[model_index_b[:, None], seg_index_b]

            # merge masks
            trials = trials_1 | trials_2

            # build new ndx
            ndx_new.trialmask = trials
            self.modelset = ndx_new.modelset
            self.segset = ndx_new.segset
            self.trialmask = ndx_new.trialmask
