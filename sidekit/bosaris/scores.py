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
This is the 'scores' module

"""
import h5py
import logging
import numpy
import os
from sidekit.bosaris.ndx import Ndx
from sidekit.bosaris.key import Key
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


class Scores:
    """A class for storing scores for trials.  The modelset and segset
    fields are lists of model and test segment names respectively.
    The element i,j of scoremat and scoremask corresponds to the
    trial involving model i and test segment j.

    :attr modelset: list of unique models in a ndarray 
    :attr segset: list of unique test segments in a ndarray
    :attr scoremask: 2D ndarray of boolean which indicates the trials of interest 
            i.e. the entry i,j in scoremat should be ignored if scoremask[i,j] is False
    :attr scoremat: 2D ndarray of scores
    """
    def __init__(self, scores_file_name=''):
        """ Initialize a Scores object by loading information from a file HDF5 format.

        :param scores_file_name: name of the file to load
        """
        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.scoremask = numpy.array([], dtype="bool")
        self.scoremat = numpy.array([])

        if scores_file_name == '':
            pass
        else:
            tmp = Scores.read(scores_file_name)
            self.modelset = tmp.modelset
            self.segset = tmp.segset
            self.scoremask = tmp.scoremask
            self.scoremat = tmp.scoremat

    def __repr__(self):
        ch = 'modelset:\n'
        ch += self.modelset+'\n'
        ch += 'segset:\n'
        ch += self.segset+'\n'
        ch += 'scoremask:\n'
        ch += self.scoremask.__repr__()+'\n'
        ch += 'scoremat:\n'
        ch += self.scoremat.__repr__()+'\n'

    @check_path_existance
    def write(self, output_file_name):
        """ Save Scores in HDF5 format

        :param output_file_name: name of the file to write to
        """
        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("modelset", data=self.modelset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("segset", data=self.segset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("score_mask", data=self.scoremask.astype('int8'),
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("scores", data=self.scoremat,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)

    @check_path_existance
    def write_txt(self, output_file_name):
        """Save a Scores object in a text file

        :param output_file_name: name of the file to write to
        """
        if not os.path.exists(os.path.dirname(output_file_name)):
            os.makedirs(os.path.dirname(output_file_name))
        
        with open(output_file_name, 'w') as fid:
            for m in range(self.modelset.shape[0]):
                segs = self.segset[self.scoremask[m, ]]
                scores = self.scoremat[m, self.scoremask[m, ]]
                for s in range(segs.shape[0]):
                    fid.write('{} {} {}\n'.format(self.modelset[m], segs[s], scores[s]))

    @check_path_existance
    def write_matlab(self, output_file_name):
        """Save a Scores object in Bosaris compatible HDF5 format
        
        :param output_file_name: name of the file to write to  
        """
        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("/ID/row_ids", data=self.modelset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("/ID/column_ids", data=self.segset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("score_mask", data=self.scoremask.astype('int8'),
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("scores", data=self.scoremat,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)

    def get_tar_non(self, key):
        """Divides scores into target and non-target scores using
        information in a key.

        :param key: a Key object.

        :return: a vector of target scores.
            :return: a vector of non-target scores.
        """
        new_score = self.align_with_ndx(key)
        tarndx = key.tar & new_score.scoremask
        nonndx = key.non & new_score.scoremask
        tar = new_score.scoremat[tarndx]
        non = new_score.scoremat[nonndx]
        return tar, non

    def align_with_ndx(self, ndx):
        """The ordering in the output Scores object corresponds to ndx, so
        aligning several Scores objects with the same ndx will result in
        them being comparable with each other.

        :param ndx: a Key or Ndx object

        :return: resized version of the current Scores object to size of \'ndx\'
                and reordered according to the ordering of modelset and segset in \'ndx\'.
        """
        aligned_scr = Scores()
        aligned_scr.modelset = ndx.modelset
        aligned_scr.segset = ndx.segset

        hasmodel = numpy.array(ismember(ndx.modelset, self.modelset))
        rindx = numpy.array([numpy.argwhere(self.modelset == v)[0][0]
                            for v in ndx.modelset[hasmodel]]).astype(int)
        hasseg = numpy.array(ismember(ndx.segset, self.segset))
        cindx = numpy.array([numpy.argwhere(self.segset == v)[0][0]
                            for v in ndx.segset[hasseg]]).astype(int)

        aligned_scr.scoremat = numpy.zeros((ndx.modelset.shape[0], ndx.segset.shape[0]))
        aligned_scr.scoremat[numpy.where(hasmodel)[0][:, None],
                             numpy.where(hasseg)[0]] = self.scoremat[rindx[:, None], cindx]

        aligned_scr.scoremask = numpy.zeros((ndx.modelset.shape[0], ndx.segset.shape[0]), dtype='bool')
        aligned_scr.scoremask[numpy.where(hasmodel)[0][:, None],
                              numpy.where(hasseg)[0]] = self.scoremask[rindx[:, None], cindx]

        assert numpy.sum(aligned_scr.scoremask) <= (numpy.sum(hasmodel) * numpy.sum(hasseg)), 'Error in new scoremask'

        if isinstance(ndx, Ndx):
            aligned_scr.scoremask = aligned_scr.scoremask & ndx.trialmask
        else:
            aligned_scr.scoremask = aligned_scr.scoremask & (ndx.tar | ndx.non)

        if numpy.sum(hasmodel) < ndx.modelset.shape[0]:
            logging.info('models reduced from %d to %d', ndx.modelset.shape[0], numpy.sum(hasmodel))
        if numpy.sum(hasseg) < ndx.segset.shape[0]:
            logging.info('testsegs reduced from %d to %d', ndx.segset.shape[0], numpy.sum(hasseg))

        if isinstance(ndx, Key):
            tar = ndx.tar & aligned_scr.scoremask
            non = ndx.non & aligned_scr.scoremask

            missing = numpy.sum(ndx.tar) - numpy.sum(tar)
            if missing > 0:
                logging.info('%d of %d targets missing', missing, numpy.sum(ndx.tar))
            missing = numpy.sum(ndx.non) - numpy.sum(non)
            if missing > 0:
                logging.info('%d of %d non targets missing', missing, numpy.sum(ndx.non))

        else:
            mask = ndx.trialmask & aligned_scr.scoremask
            missing = numpy.sum(ndx.trialmask) - numpy.sum(mask)
            if missing > 0:
                logging.info('%d of %d trials missing', missing, numpy.sum(ndx.trialmask))

        assert all(numpy.isfinite(aligned_scr.scoremat[aligned_scr.scoremask])), \
            'Inifinite or Nan value in the scoremat'
        assert aligned_scr.validate(), 'Wrong Score format'
        return aligned_scr

    def set_missing_to_value(self, ndx, value):
        """Sets all scores for which the trialmask is true but the scoremask
        is false to the same value, supplied by the user.

        :param ndx: a Key or Ndx object.
        :param value: a value for the missing scores.

        :return: a Scores object (with the missing scores added and set
                    to value).
        """
        if isinstance(ndx, Key):
            ndx = ndx.to_ndx()

        new_scr = self.align_with_ndx(ndx)
        missing = ndx.trialmask & -new_scr.scoremask
        new_scr.scoremat[missing] = value
        new_scr.scoremask[missing] = True
        assert new_scr.validate(), "Wrong format of Scores"
        return new_scr

    def filter(self, modlist, seglist, keep):
        """Removes some of the information in a Scores object.  Useful for
        creating a gender specific score set from a pooled gender score
        set.  Depending on the value of \'keep\', the two input lists
        indicate the models and test segments (and their associated
        scores) to retain or discard.

        :param modlist: a list of strings which will be compared with
                the modelset of the current Scores object.
        :param seglist: a list of strings which will be compared with
                    the segset of \'inscr\'.
        :param  keep: a boolean indicating whether modlist and seglist are the
                models to keep or discard.

        :return: a filtered version of \'inscr\'.
        """
        if keep:
            keepmods = modlist
            keepsegs = seglist
        else:
            keepmods = diff(self.modelset, modlist)
            keepsegs = diff(self.segset, seglist)

        keepmodidx = numpy.array(ismember(self.modelset, keepmods))
        keepsegidx = numpy.array(ismember(self.segset, keepsegs))

        outscr = Scores()
        outscr.modelset = self.modelset[keepmodidx]
        outscr.segset = self.segset[keepsegidx]
        tmp = self.scoremat[numpy.array(keepmodidx), :]
        outscr.scoremat = tmp[:, numpy.array(keepsegidx)]
        tmp = self.scoremask[numpy.array(keepmodidx), :]
        outscr.scoremask = tmp[:, numpy.array(keepsegidx)]

        assert isinstance(outscr, Scores), 'Wrong Scores format'

        if self.modelset.shape[0] > outscr.modelset.shape[0]:
            logging.info('Number of models reduced from %d to %d', self.modelset.shape[0], outscr.modelset.shape[0])
        if self.segset.shape[0] > outscr.segset.shape[0]:
            logging.info('Number of test segments reduced from %d to %d', self.segset.shape[0], outscr.segset.shape[0])
        return outscr

    def validate(self):
        """Checks that an object of type Scores obeys certain rules that
        must always be true.

            :return: a boolean value indicating whether the object is valid.
        """
        ok = self.scoremat.shape == self.scoremask.shape
        ok &= (self.scoremat.shape[0] == self.modelset.shape[0])
        ok &= (self.scoremat.shape[1] == self.segset.shape[0])
        return ok

    @staticmethod
    def read(input_file_name):
        """Read a Scores object from information in a hdf5 file.

        :param input_file_name: name of the file to read from
        """
        with h5py.File(input_file_name, "r") as f:
            scores = Scores()

            scores.modelset = numpy.empty(f["modelset"].shape, dtype=f["modelset"].dtype)
            f["modelset"].read_direct(scores.modelset)
            scores.modelset = scores.modelset.astype('U100', copy=False)

            scores.segset = numpy.empty(f["segset"].shape, dtype=f["segset"].dtype)
            f["segset"].read_direct(scores.segset)
            scores.segset = scores.segset.astype('U100', copy=False)

            scores.scoremask = numpy.empty(f["score_mask"].shape, dtype=f["score_mask"].dtype)
            f["score_mask"].read_direct(scores.scoremask)
            scores.scoremask = scores.scoremask.astype('bool', copy=False)

            scores.scoremat = numpy.empty(f["scores"].shape, dtype=f["scores"].dtype)
            f["scores"].read_direct(scores.scoremat)

            assert scores.validate(), "Error: wrong Scores format"
            return scores

    @staticmethod
    def read_matlab(input_file_name):
        """Read a Scores object from information in a hdf5 file in Matlab BOSARIS format.

            :param input_file_name: name of the file to read from
        """
        with h5py.File(input_file_name, "r") as f:
            scores = Scores()

            scores.modelset = numpy.empty(f["ID/row_ids"].shape, dtype=f["ID/row_ids"].dtype)
            f["ID/row_ids"].read_direct(scores.modelset)
            scores.modelset = scores.modelset.astype('U100', copy=False)

            scores.segset = numpy.empty(f["ID/column_ids"].shape, dtype=f["ID/column_ids"].dtype)
            f["ID/column_ids"].read_direct(scores.segset)
            scores.segset = scores.segset.astype('U100', copy=False)

            scores.scoremask = numpy.empty(f["score_mask"].shape, dtype=f["score_mask"].dtype)
            f["score_mask"].read_direct(scores.scoremask)
            scores.scoremask = scores.scoremask.astype('bool', copy=False)

            scores.scoremat = numpy.empty(f["scores"].shape, dtype=f["scores"].dtype)
            f["scores"].read_direct(scores.scoremat)

            assert scores.validate(), "Error: wrong Scores format"
            return scores

    @classmethod
    @check_path_existance
    def read_txt(cls, input_file_name):
        """Creates a Scores object from information stored in a text file.

        :param input_file_name: name of the file to read from
        """
        s = Scores()
        with open(input_file_name, 'r') as fid:
            lines = [l.rstrip().split() for l in fid]

        models = numpy.array([], '|O')
        models.resize(len(lines))
        testsegs = numpy.array([], '|O')
        testsegs.resize(len(lines))
        scores = numpy.array([])
        scores.resize(len(lines))

        for ii in range(len(lines)):
            models[ii] = lines[ii][0]
            testsegs[ii] = lines[ii][1]
            scores[ii] = float(lines[ii][2])

        modelset = numpy.unique(models)
        segset = numpy.unique(testsegs)

        scoremask = numpy.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
        scoremat = numpy.zeros((modelset.shape[0], segset.shape[0]))
        for m in range(modelset.shape[0]):
            segs = testsegs[numpy.array(ismember(models, modelset[m]))]
            scrs = scores[numpy.array(ismember(models, modelset[m]))]
            idx = segs.argsort()
            segs = segs[idx]
            scrs = scrs[idx]
            scoremask[m, ] = ismember(segset, segs)
            scoremat[m, numpy.array(ismember(segset, segs))] = scrs

        s.modelset = modelset
        s.segset = segset
        s.scoremask = scoremask
        s.scoremat = scoremat
        assert s.validate(), "Wrong Scores format"
        s.sort()
        return s

    def merge(self, score_list):
        """Merges a list of Scores objects into the current one.
        The resulting must have all models and segment in the input
        Scores (only once) and the union of all the scoremasks.
        It is an error if two of the input Scores objects have a
        score for the same trial.

        :param score_list: the list of Scores object to merge
        """
        assert isinstance(score_list, list), "Input is not a list"
        for scr in score_list:
            assert isinstance(score_list, list), \
                '{} {} {}'.format("Element ", scr, " is not a Score")

        self.validate()
        for scr2 in score_list:
            scr_new = Scores()
            scr1 = self
            scr1.sort()
            scr2.sort()

            # create new scr with empty matrices
            scr_new.modelset = numpy.union1d(scr1.modelset, scr2.modelset)
            scr_new.segset = numpy.union1d(scr1.segset, scr2.segset)

            # expand scr1 matrices
            scoremat_1 = numpy.zeros((scr_new.modelset.shape[0], scr_new.segset.shape[0]))
            scoremask_1 = numpy.zeros((scr_new.modelset.shape[0], scr_new.segset.shape[0]), dtype='bool')
            model_index_a = numpy.argwhere(numpy.in1d(scr_new.modelset, scr1.modelset))
            model_index_b = numpy.argwhere(numpy.in1d(scr1.modelset, scr_new.modelset))
            seg_index_a = numpy.argwhere(numpy.in1d(scr_new.segset, scr1.segset))
            seg_index_b = numpy.argwhere(numpy.in1d(scr1.segset, scr_new.segset))
            scoremat_1[model_index_a[:, None], seg_index_a] = scr1.scoremat[model_index_b[:, None], seg_index_b]
            scoremask_1[model_index_a[:, None], seg_index_a] = scr1.scoremask[model_index_b[:, None], seg_index_b]

            # expand scr2 matrices
            scoremat_2 = numpy.zeros((scr_new.modelset.shape[0], scr_new.segset.shape[0]))
            scoremask_2 = numpy.zeros((scr_new.modelset.shape[0], scr_new.segset.shape[0]), dtype='bool')
            model_index_a = numpy.argwhere(numpy.in1d(scr_new.modelset, scr2.modelset))
            model_index_b = numpy.argwhere(numpy.in1d(scr2.modelset, scr_new.modelset))
            seg_index_a = numpy.argwhere(numpy.in1d(scr_new.segset, scr2.segset))
            seg_index_b = numpy.argwhere(numpy.in1d(scr2.segset, scr_new.segset))
            scoremat_2[model_index_a[:, None], seg_index_a] = scr2.scoremat[model_index_b[:, None], seg_index_b]
            scoremask_2[model_index_a[:, None], seg_index_a] = scr2.scoremask[model_index_b[:, None], seg_index_b]

            # check for clashes
            assert numpy.sum(scoremask_1 & scoremask_2) == 0, "Conflict in the new scoremask"

            # merge masks
            self.scoremat = scoremat_1 + scoremat_2
            self.scoremask = scoremask_1 | scoremask_2
            self.modelset = scr_new.modelset
            self.segset = scr_new.segset
            assert self.validate(), 'Wrong Scores format'

    def sort(self):
        """Sort models and segments"""
        sort_model_idx = numpy.argsort(self.modelset)
        sort_seg_idx = numpy.argsort(self.segset)
        sort_mask = self.scoremask[sort_model_idx[:, None], sort_seg_idx]
        sort_mat = self.scoremat[sort_model_idx[:, None], sort_seg_idx]
        self.modelset.sort()
        self.segset.sort()
        self.scoremat = sort_mat
        self.scoremask = sort_mask
    
    def get_score(self, modelID, segID):
        """return a score given a model and segment identifiers
        raise an error if the trial does not exist
        :param modelID: id of the model
        :param segID: id of the test segment
        """
        model_idx = numpy.argwhere(self.modelset == modelID)
        seg_idx = numpy.argwhere(self.segset == segID)
        if model_idx.shape[0] == 0:
            raise Exception('No such model as: %s', modelID)
        elif seg_idx.shape[0] == 0:
            raise Exception('No such segment as: %s', segID)
        else:
            return self.scoremat[model_idx, seg_idx]
