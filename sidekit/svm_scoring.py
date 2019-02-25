# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2019 Anthony Larcher

:mod:`svm_scoring` provides functions to perform speaker verification 
by using Support Vector Machines.
"""
import ctypes
import numpy
import multiprocessing
import logging
import sidekit.sv_utils
from sidekit.bosaris import Ndx
from sidekit.bosaris import Scores
from sidekit.statserver import StatServer


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def svm_scoring_singleThread(svm_filename_structure, test_sv, ndx, score, seg_idx=None):
    """Compute scores for SVM verification on a single thread
    (two classes only as implementeed at the moment)
     
    :param svm_filename_structure: structure of the filename where to load the SVM models
    :param test_sv: StatServer object of super-vectors. stat0 are set to 1 and stat1 are the super-vector to classify
    :param ndx: Ndx object of the trials to perform
    :param score: Scores object to fill
    :param seg_idx: list of segments to classify. Classify all if the list is empty.
    """
    assert isinstance(test_sv, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'

    if seg_idx is None:
        seg_idx = range(ndx.segset.shape[0])

    # Load SVM models
    Msvm = numpy.zeros((ndx.modelset.shape[0], test_sv.stat1.shape[1]))
    bsvm = numpy.zeros(ndx.modelset.shape[0])
    for m in range(ndx.modelset.shape[0]):
        svm_file_name = svm_filename_structure.format(ndx.modelset[m])
        w, b = sidekit.sv_utils.read_svm(svm_file_name)
        Msvm[m, :] = w
        bsvm[m] = b

    # Compute scores against all test segments
    for ts in seg_idx:
        logging.info('Compute trials involving test segment %d/%d', ts + 1, ndx.segset.shape[0])

        # Select the models to test with the current segment
        models = ndx.modelset[ndx.trialmask[:, ts]]
        ind_dict = dict((k, i) for i, k in enumerate(ndx.modelset))
        inter = set(ind_dict.keys()).intersection(models)
        idx_ndx = numpy.array([ind_dict[x] for x in inter])

        scores = numpy.dot(Msvm[idx_ndx, :], test_sv.stat1[ts, :]) + bsvm[idx_ndx]

        # Fill the score matrix
        score.scoremat[idx_ndx, ts] = scores


def svm_scoring(svm_filename_structure, test_sv, ndx, num_thread=1):
    """Compute scores for SVM verification on multiple threads
    (two classes only as implementeed at the moment)
    
    :param svm_filename_structure: structure of the filename where to load the SVM models
    :param test_sv: StatServer object of super-vectors. stat0 are set to 1 and stat1
          are the super-vector to classify
    :param ndx: Ndx object of the trials to perform
    :param num_thread: number of thread to launch in parallel
    
    :return: a Score object.
    """
    # Remove missing models and test segments
    existing_models, model_idx = sidekit.sv_utils.check_file_list(ndx.modelset, svm_filename_structure)
    clean_ndx = ndx.filter(existing_models, test_sv.segset, True)

    score = Scores()
    score.scoremat = numpy.zeros(clean_ndx.trialmask.shape)
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask

    tmp = multiprocessing.Array(ctypes.c_double, score.scoremat.size)
    score.scoremat = numpy.ctypeslib.as_array(tmp.get_obj())
    score.scoremat = score.scoremat.reshape(score.modelset.shape[0], score.segset.shape[0])

    # Split the list of segment to process for multi-threading
    los = numpy.array_split(numpy.arange(clean_ndx.segset.shape[0]), num_thread)

    jobs = []
    for idx in los:
        p = multiprocessing.Process(target=svm_scoring_singleThread,
                                    args=(svm_filename_structure, test_sv, clean_ndx, score, idx))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    return score
