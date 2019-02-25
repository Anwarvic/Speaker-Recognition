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
Copyright 2014-2019 Anthony Larcher and Sylvain Meignier

    :mod:`features_server` provides methods to test gmm models

"""
import numpy
import warnings
import multiprocessing
import ctypes
import logging

import sidekit.sv_utils
import sidekit.frontend
from sidekit.mixture import Mixture
from sidekit.statserver import StatServer
from sidekit.features_server import FeaturesServer
from sidekit.bosaris import Ndx
from sidekit.bosaris import Scores


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def gmm_scoring_singleThread(ubm, enroll, ndx, feature_server, score_mat, seg_idx=None):
    """Compute log-likelihood ratios for sequences of acoustic feature 
    frames between a Universal Background Model (UBM) and a list of Gaussian
    Mixture Models (GMMs) which only mean vectors differ from the UBM.
    
    :param ubm: a Mixture object used to compute the denominator 
        of the likelihood ratios
    :param enroll: a StatServer object which stat1 attribute contains mean 
        super-vectors of the GMMs to use to compute the numerator of the 
        likelihood ratios.
    :param ndx: an Ndx object which define the list of trials to compute
    :param feature_server: sidekit.FeaturesServer used to load the acoustic parameters
    :param score_mat: a ndarray of scores to fill
    :param seg_idx: the list of unique test segments to process.
        Those test segments should belong to the list of test segments 
        in the ndx object. By setting seg_idx=None, all test segments
        from the ndx object will be processed
    
    """
    assert isinstance(ubm, Mixture), 'First parameter should be a Mixture'
    assert isinstance(enroll, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be a Ndx'
    assert isinstance(feature_server, FeaturesServer), 'Fourth parameter should be a FeatureServer'
    
    if seg_idx is None:
        seg_idx = range(ndx.segset.shape[0])

    for ts in seg_idx:
        logging.info('Compute trials involving test segment %d/%d', ts + 1, ndx.segset.shape[0])

        # Select the models to test with the current segment
        models = ndx.modelset[ndx.trialmask[:, ts]]
        ind_dict = dict((k, i) for i, k in enumerate(ndx.modelset))
        inter = set(ind_dict.keys()).intersection(models)
        idx_ndx = [ind_dict[x] for x in inter]
        ind_dict = dict((k, i) for i, k in enumerate(enroll.modelset))
        inter = set(ind_dict.keys()).intersection(models)
        idx_enroll = [ind_dict[x] for x in inter]

        # Load feature file
        cep, _ = feature_server.load(ndx.segset[ts])
        
        llr = numpy.zeros(numpy.array(idx_enroll).shape)
        for m in range(llr.shape[0]):
            # Compute llk for the current model
            if ubm.invcov.ndim == 2:
                lp = ubm.compute_log_posterior_probabilities(cep, enroll.stat1[idx_enroll[m], :])
            elif ubm.invcov.ndim == 3:
                lp = ubm.compute_log_posterior_probabilities_full(cep, enroll.stat1[idx_enroll[m], :])
            pp_max = numpy.max(lp, axis=1)
            log_lk = pp_max + numpy.log(numpy.sum(numpy.exp((lp.transpose() - pp_max).transpose()), axis=1))
            llr[m] = log_lk.mean()
       
        # Compute and substract llk for the ubm
        if ubm.invcov.ndim == 2:
            lp = ubm.compute_log_posterior_probabilities(cep)
        elif ubm.invcov.ndim == 3:
            lp = ubm.compute_log_posterior_probabilities_full(cep)
        ppMax = numpy.max(lp, axis=1)
        loglk = ppMax + numpy.log(numpy.sum(numpy.exp((lp.transpose() - ppMax).transpose()), axis=1))
        llr = llr - loglk.mean()

        # Fill the score matrix
        score_mat[idx_ndx, ts] = llr


def gmm_scoring(ubm, enroll, ndx, feature_server, num_thread=1):
    """Compute log-likelihood ratios for sequences of acoustic feature 
    frames between a Universal Background Model (UBM) and a list of 
    Gaussian Mixture Models (GMMs) which only mean vectors differ 
    from the UBM.
    
    :param ubm: a Mixture object used to compute the denominator of the 
        likelihood ratios
    :param enroll: a StatServer object which stat1 attribute contains 
        mean super-vectors of the GMMs to use to compute the numerator 
        of the likelihood ratios.
    :param ndx: an Ndx object which define the list of trials to compute
    :param feature_server: a FeatureServer object to load the features
    :param num_thread: number of thread to launch in parallel
    
    :return: a Score object.
    
    """
    assert isinstance(ubm, Mixture), 'First parameter should be a Mixture'
    assert isinstance(enroll, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be a Ndx'
    assert isinstance(feature_server, FeaturesServer), 'Fourth parameter should be a FeatureServer'

    # Remove missing models and test segments
    if feature_server.features_extractor is None:
        existing_test_seg, test_seg_idx = sidekit.sv_utils.check_file_list(ndx.segset,
                                                                           feature_server.feature_filename_structure)
        clean_ndx = ndx.filter(enroll.modelset, existing_test_seg, True)
    elif feature_server.features_extractor.audio_filename_structure is not None:
        existing_test_seg, test_seg_idx = \
            sidekit.sv_utils.check_file_list(ndx.segset, feature_server.features_extractor.audio_filename_structure)
        clean_ndx = ndx.filter(enroll.modelset, existing_test_seg, True)
    else:
        clean_ndx = ndx

    s = numpy.zeros(clean_ndx.trialmask.shape)
    dims = s.shape
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        tmp_stat1 = multiprocessing.Array(ctypes.c_double, s.size)
        s = numpy.ctypeslib.as_array(tmp_stat1.get_obj())
        s = s.reshape(dims)

    # Split the list of segment to process for multi-threading
    los = numpy.array_split(numpy.arange(clean_ndx.segset.shape[0]), num_thread)
    jobs = []
    for idx in los:
        p = multiprocessing.Process(target=gmm_scoring_singleThread, args=(ubm, enroll, ndx, feature_server, s, idx))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    score = Scores()
    score.scoremat = s
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask
    return score
