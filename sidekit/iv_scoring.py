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

    :mod:`iv_scoring` provides methods to compare i-vectors
"""
import copy
import logging
import numpy
import scipy
from sidekit.bosaris import Ndx
from sidekit.bosaris import Scores
from sidekit.statserver import StatServer

import sys
if sys.version_info.major > 2:
    from functools import reduce


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def _check_missing_model(enroll, test, ndx):
    # Remove missing models and test segments
    clean_ndx = ndx.filter(enroll.modelset, test.segset, True)

    # Align StatServers to match the clean_ndx
    enroll.align_models(clean_ndx.modelset)
    test.align_segments(clean_ndx.segset)

    return clean_ndx


def cosine_scoring(enroll, test, ndx, wccn=None, check_missing=True):
    """Compute the cosine similarities between to sets of vectors. The list of 
    trials to perform is given in an Ndx object.
    
    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param wccn: numpy.ndarray, if provided, the i-vectors are normalized by using a Within Class Covariance Matrix
    :param check_missing: boolean, if True, check that all models and segments exist
    
    :return: a score object
    """
    assert isinstance(enroll, StatServer), 'First parameter should be a StatServer'
    assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    enroll_copy = copy.deepcopy(enroll)
    test_copy = copy.deepcopy(test)

    # If models are not unique, compute the mean per model, display a warning
    #if not numpy.unique(enroll_copy.modelset).shape == enroll_copy.modelset.shape:
    #    logging.warning("Enrollment models are not unique, average i-vectors")
    #    enroll_copy = enroll_copy.mean_stat_per_model()

    # Remove missing models and test segments
    if check_missing:
        clean_ndx = _check_missing_model(enroll_copy, test_copy, ndx)
    else:
        clean_ndx = ndx

    if wccn is not None:
        enroll_copy.rotate_stat1(wccn)
        if enroll_copy != test_copy:
            test_copy.rotate_stat1(wccn)

    # Cosine scoring
    enroll_copy.norm_stat1()
    if enroll_copy != test_copy:
        test_copy.norm_stat1()
    s = numpy.dot(enroll_copy.stat1, test_copy.stat1.transpose())

    score = Scores()
    score.scoremat = s
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask
    return score


def mahalanobis_scoring(enroll, test, ndx, m, check_missing=True):
    """Compute the mahalanobis distance between to sets of vectors. The list of 
    trials to perform is given in an Ndx object.
    
    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param m: mahalanobis matrix as a ndarray
    :param check_missing: boolean, default is True, set to False not to check missing models
    
    :return: a score object
    """
    assert isinstance(enroll, StatServer), 'First parameter should be a StatServer'
    assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    assert enroll.stat1.shape[1] == test.stat1.shape[1], 'I-vectors dimension mismatch'
    assert enroll.stat1.shape[1] == m.shape[0], 'I-vectors and Mahalanobis matrix dimension mismatch'

    # If models are not unique, compute the mean per model, display a warning
    if not numpy.unique(enroll.modelset).shape == enroll.modelset.shape:
        logging.warning("Enrollment models are not unique, average i-vectors")
        enroll = enroll.mean_stat_per_model()

    # Remove missing models and test segments
    if check_missing:
        clean_ndx = _check_missing_model(enroll, test, ndx)
    else:
        clean_ndx = ndx

    # Mahalanobis scoring
    s = numpy.zeros((enroll.modelset.shape[0], test.segset.shape[0]))
    for i in range(enroll.modelset.shape[0]):
        diff = enroll.stat1[i, :] - test.stat1
        s[i, :] = -0.5 * numpy.sum(numpy.dot(diff, m) * diff, axis=1)

    score = Scores()
    score.scoremat = s
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask
    return score


def two_covariance_scoring(enroll, test, ndx, W, B, check_missing=True):
    """Compute the 2-covariance scores between to sets of vectors. The list of 
    trials to perform is given in an Ndx object. Within and between class 
    co-variance matrices have to be pre-computed.
    
    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param W: the within-class co-variance matrix to consider
    :param B: the between-class co-variance matrix to consider
    :param check_missing: boolean, default is True, set to False not to check missing models
      
    :return: a score object
    """
    assert isinstance(enroll, StatServer), 'First parameter should be a directory'
    assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    assert enroll.stat1.shape[1] == test.stat1.shape[1], 'I-vectors dimension mismatch'
    assert enroll.stat1.shape[1] == W.shape[0], 'I-vectors and co-variance matrix dimension mismatch'
    assert enroll.stat1.shape[1] == B.shape[0], 'I-vectors and co-variance matrix dimension mismatch'

    # If models are not unique, compute the mean per model, display a warning
    if not numpy.unique(enroll.modelset).shape == enroll.modelset.shape:
        logging.warning("Enrollment models are not unique, average i-vectors")
        enroll = enroll.mean_stat_per_model()

    # Remove missing models and test segments
    if check_missing:
        clean_ndx = _check_missing_model(enroll, test, ndx)
    else:
        clean_ndx = ndx

    # Two covariance scoring scoring
    S = numpy.zeros((enroll.modelset.shape[0], test.segset.shape[0]))
    iW = scipy.linalg.inv(W)
    iB = scipy.linalg.inv(B)

    G = reduce(numpy.dot, [iW, scipy.linalg.inv(iB + 2*iW), iW])
    H = reduce(numpy.dot, [iW, scipy.linalg.inv(iB + iW), iW])

    s2 = numpy.sum(numpy.dot(enroll.stat1, H) * enroll.stat1, axis=1)
    s3 = numpy.sum(numpy.dot(test.stat1, H) * test.stat1, axis=1)

    for ii in range(enroll.modelset.shape[0]):
        A = enroll.stat1[ii, :] + test.stat1
        s1 = numpy.sum(numpy.dot(A, G) * A, axis=1)
        S[ii, :] = s1 - s3 - s2[ii]

    score = Scores()
    score.scoremat = S
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask
    return score


def PLDA_scoring(enroll,
                 test,
                 ndx,
                 mu,
                 F,
                 G,
                 Sigma,
                 test_uncertainty=None,
                 Vtrans=None,
                 p_known=0.0,
                 scaling_factor=1.,

                 full_model=False):
    """Compute the PLDA scores between to sets of vectors. The list of
    trials to perform is given in an Ndx object. PLDA matrices have to be
    pre-computed. i-vectors are supposed to be whitened before.

    Implements the appraoch described in [Lee13]_ including scoring
    for partially open-set identification

    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param mu: the mean vector of the PLDA gaussian
    :param F: the between-class co-variance matrix of the PLDA
    :param G: the within-class co-variance matrix of the PLDA
    :param Sigma: the residual covariance matrix
    :param p_known: probability of having a known speaker for open-set
        identification case (=1 for the verification task and =0 for the
        closed-set case)
    :param scaling_factor: scaling factor to be multiplied by the sufficient statistics
    :param full_model: boolean, set to True when using a complete PLDA model (including within class covariance matrix)

    :return: a score object
    """
    assert isinstance(enroll, StatServer), 'First parameter should be a StatServer'
    assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    assert enroll.stat1.shape[1] == test.stat1.shape[1], 'I-vectors dimension mismatch'
    assert enroll.stat1.shape[1] == F.shape[0], 'I-vectors and co-variance matrix dimension mismatch'
    assert enroll.stat1.shape[1] == G.shape[0], 'I-vectors and co-variance matrix dimension mismatch'

    if not full_model:
        return fast_PLDA_scoring(enroll,
                                 test,
                                 ndx,
                                 mu,
                                 F,
                                 Sigma,
                                 test_uncertainty,
                                 Vtrans,
                                 p_known=p_known,
                                 scaling_factor=scaling_factor,
                                 check_missing=True)
    else:
        return full_PLDA_scoring(enroll, test, ndx, mu, F, G, Sigma, p_known=p_known, scaling_factor=scaling_factor)


def full_PLDA_scoring(enroll, test, ndx, mu, F, G, Sigma, p_known=0.0, scaling_factor=1., check_missing=True):
    """Compute PLDA scoring

    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param mu: the mean vector of the PLDA gaussian
    :param F: the between-class co-variance matrix of the PLDA
    :param G: the within-class co-variance matrix of the PLDA
    :param Sigma: the residual covariance matrix
    :param p_known: probability of having a known speaker for open-set
        identification case (=1 for the verification task and =0 for the
        closed-set case)
    :param check_missing: boolean, default is True, set to False not to check missing models

    """

    enroll_copy = copy.deepcopy(enroll)
    test_copy = copy.deepcopy(test)

    # If models are not unique, compute the mean per model, display a warning
    # if not numpy.unique(enroll_copy.modelset).shape == enroll_copy.modelset.shape:
    #    logging.warning("Enrollment models are not unique, average i-vectors")
    #    enroll_copy = enroll_copy.mean_stat_per_model()

    # Remove missing models and test segments
    if check_missing:
        clean_ndx = _check_missing_model(enroll_copy, test_copy, ndx)
    else:
        clean_ndx = ndx

    # Center the i-vectors around the PLDA mean
    enroll_copy.center_stat1(mu)
    test_copy.center_stat1(mu)

    # Compute temporary matrices
    invSigma = scipy.linalg.inv(Sigma)
    I_iv = numpy.eye(mu.shape[0], dtype='float')
    I_ch = numpy.eye(G.shape[1], dtype='float')
    I_spk = numpy.eye(F.shape[1], dtype='float')
    A = numpy.linalg.inv(G.T.dot(invSigma * scaling_factor).dot(G) + I_ch)  # keep numpy as interface are different
    B = F.T.dot(invSigma * scaling_factor).dot(I_iv - G.dot(A).dot(G.T).dot(invSigma * scaling_factor))
    K = B.dot(F)
    K1 = scipy.linalg.inv(K + I_spk)
    K2 = scipy.linalg.inv(2 * K + I_spk)

    # Compute the Gaussian distribution constant
    alpha1 = numpy.linalg.slogdet(K1)[1]
    alpha2 = numpy.linalg.slogdet(K2)[1]
    constant = alpha2 / 2.0 - alpha1

    # Compute verification scores
    score = Scores()
    score.scoremat = numpy.zeros(clean_ndx.trialmask.shape)
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask

    # Project data in the space that maximizes the speaker separability
    test_tmp = B.dot(test_copy.stat1.T)
    enroll_tmp = B.dot(enroll_copy.stat1.T)

    # score qui ne dépend que du segment
    tmp1 = test_tmp.T.dot(K1)

    # Compute the part of the score that is only dependent on the test segment
    S1 = numpy.empty(test_copy.segset.shape[0])
    for seg_idx in range(test_copy.segset.shape[0]):
        S1[seg_idx] = tmp1[seg_idx, :].dot(test_tmp[:, seg_idx])/2.

    # Compute the part of the score that depends only on the model (S2) and on both model and test segment
    S2 = numpy.empty(enroll_copy.modelset.shape[0])

    for model_idx in range(enroll_copy.modelset.shape[0]):
        mod_plus_test_seg = test_tmp + numpy.atleast_2d(enroll_tmp[:, model_idx]).T
        tmp2 = mod_plus_test_seg.T.dot(K2)

        S2[model_idx] = enroll_tmp[:, model_idx].dot(K1).dot(enroll_tmp[:, model_idx])/2.
        score.scoremat[model_idx, :] = numpy.einsum("ij, ji->i", tmp2, mod_plus_test_seg)/2.

    score.scoremat += constant - (S1 + S2[:, numpy.newaxis])
    score.scoremat *= scaling_factor

    # Case of open-set identification, we compute the log-likelihood
    # by taking into account the probability of having a known impostor
    # or an out-of set class
    if p_known != 0:
        N = score.scoremat.shape[0]
        open_set_scores = numpy.empty(score.scoremat.shape)
        tmp = numpy.exp(score.scoremat)
        for ii in range(N):
            # open-set term
            open_set_scores[ii, :] = score.scoremat[ii, :] \
                - numpy.log(p_known * tmp[~(numpy.arange(N) == ii)].sum(axis=0) / (N - 1) + (1 - p_known))
        score.scoremat = open_set_scores

    return score

def fast_PLDA_scoring(enroll,
                      test,
                      ndx,
                      mu,
                      F,
                      Sigma,
                      test_uncertainty=None,
                      Vtrans=None,
                      p_known=0.0,
                      scaling_factor=1.,
                      check_missing=True):
    """Compute the PLDA scores between to sets of vectors. The list of
    trials to perform is given in an Ndx object. PLDA matrices have to be
    pre-computed. i-vectors are supposed to be whitened before.

    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param mu: the mean vector of the PLDA gaussian
    :param F: the between-class co-variance matrix of the PLDA
    :param Sigma: the residual covariance matrix
    :param p_known: probability of having a known speaker for open-set
        identification case (=1 for the verification task and =0 for the
        closed-set case)
    :param check_missing: boolean, if True, check that all models and segments exist

    :return: a score object
    """
    # assert isinstance(enroll, StatServer), 'First parameter should be a StatServer'
    # assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    # assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    # assert enroll.stat1.shape[1] == test.stat1.shape[1], 'I-vectors dimension mismatch'
    # assert enroll.stat1.shape[1] == F.shape[0], 'I-vectors and co-variance matrix dimension mismatch'
    # assert enroll.stat1.shape[1] == G.shape[0], 'I-vectors and co-variance matrix dimension mismatch'

    enroll_ctr = copy.deepcopy(enroll)
    test_ctr = copy.deepcopy(test)

    # If models are not unique, compute the mean per model, display a warning
    if not numpy.unique(enroll_ctr.modelset).shape == enroll_ctr.modelset.shape:
        logging.warning("Enrollment models are not unique, average i-vectors")
        enroll_ctr = enroll_ctr.mean_stat_per_model()

    # Remove missing models and test segments
    if check_missing:
        clean_ndx = _check_missing_model(enroll_ctr, test_ctr, ndx)
    else:
        clean_ndx = ndx

    # Center the i-vectors around the PLDA mean
    enroll_ctr.center_stat1(mu)
    test_ctr.center_stat1(mu)

    # If models are not unique, compute the mean per model, display a warning
    if not numpy.unique(enroll_ctr.modelset).shape == enroll_ctr.modelset.shape:
        logging.warning("Enrollment models are not unique, average i-vectors")
        enroll_ctr = enroll_ctr.mean_stat_per_model()

    # Compute constant component of the PLDA distribution
    invSigma = scipy.linalg.inv(Sigma)
    I_spk = numpy.eye(F.shape[1], dtype='float')

    K = F.T.dot(invSigma * scaling_factor).dot(F)
    K1 = scipy.linalg.inv(K + I_spk)
    K2 = scipy.linalg.inv(2 * K + I_spk)

    # Compute the Gaussian distribution constant
    alpha1 = numpy.linalg.slogdet(K1)[1]
    alpha2 = numpy.linalg.slogdet(K2)[1]
    plda_cst = alpha2 / 2.0 - alpha1

    # Compute intermediate matrices
    Sigma_ac = numpy.dot(F, F.T)
    Sigma_tot = Sigma_ac + Sigma
    Sigma_tot_inv = scipy.linalg.inv(Sigma_tot)

    Tmp = numpy.linalg.inv(Sigma_tot - Sigma_ac.dot(Sigma_tot_inv).dot(Sigma_ac))
    Phi = Sigma_tot_inv - Tmp
    Psi = Sigma_tot_inv.dot(Sigma_ac).dot(Tmp)

    # Compute the different parts of PLDA score
    model_part = 0.5 * numpy.einsum('ij, ji->i', enroll_ctr.stat1.dot(Phi), enroll_ctr.stat1.T)
    seg_part = 0.5 * numpy.einsum('ij, ji->i', test_ctr.stat1.dot(Phi), test_ctr.stat1.T)

    # Compute verification scores
    score = Scores()
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask

    score.scoremat = model_part[:, numpy.newaxis] + seg_part + plda_cst
    score.scoremat += enroll_ctr.stat1.dot(Psi).dot(test_ctr.stat1.T)
    score.scoremat *= scaling_factor

    # Case of open-set identification, we compute the log-likelihood
    # by taking into account the probability of having a known impostor
    # or an out-of set class
    if p_known != 0:
        N = score.scoremat.shape[0]
        open_set_scores = numpy.empty(score.scoremat.shape)
        tmp = numpy.exp(score.scoremat)
        for ii in range(N):
            # open-set term
            open_set_scores[ii, :] = score.scoremat[ii, :] \
                - numpy.log(p_known * tmp[~(numpy.arange(N) == ii)].sum(axis=0) / (N - 1) + (1 - p_known))
        score.scoremat = open_set_scores

    return score


#IL FAUT RAJOUTER LA GESTION DES MULTI-SESSIONS (voir fonction PLDA_scoring_with_test_uncertainty_by_the_book de Themos)
#IMPLEMENTER LA VERSION "BY THE BOOK" pour ne pas utiliser la moyenne des i-vecteurs
def PLDA_scoring_uncertainty(enroll,
                             test,
                             ndx,
                             mu,
                             F,
                             Sigma,
                             p_known=0.0,
                             scaling_factor=1.,
                             test_uncertainty=None,
                             Vtrans=None,
                             check_missing=True):
    """

    :param enroll:
    :param test:
    :param ndx:
    :param mu:
    :param F:
    :param Sigma:
    :param p_known:
    :param scaling_factor:
    :param test_uncertainty:
    :param Vtrans:
    :param check_missing:
    :return:
    """
    assert isinstance(enroll, StatServer), 'First parameter should be a StatServer'
    assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    assert enroll.stat1.shape[1] == test.stat1.shape[1], 'I-vectors dimension mismatch'
    assert enroll.stat1.shape[1] == F.shape[0], 'I-vectors and co-variance matrix dimension mismatch'
    assert enroll.stat1.shape[1] == G.shape[0], 'I-vectors and co-variance matrix dimension mismatch'

    enroll_ctr = copy.deepcopy(enroll)
    test_ctr = copy.deepcopy(test)

    # Remove missing models and test segments
    if check_missing:
        clean_ndx = _check_missing_model(enroll_ctr, test_ctr, ndx)
    else:
        clean_ndx = ndx

    # Center the i-vectors around the PLDA mean
    enroll_ctr.center_stat1(mu)
    test_ctr.center_stat1(mu)

    # Align StatServers to match the clean_ndx
    enroll_ctr.align_models_average(clean_ndx.modelset)
    test_ctr.align_segments(clean_ndx.segset)

    # Compute constant component of the PLDA distribution
    scoremat = numpy.zeros((enroll_ctr.stat1.shape[0], test_ctr.stat1.shape[0]), dtype='float')
    invSigma = scipy.linalg.inv(Sigma)

    K1 = scipy.linalg.inv(numpy.eye(F.shape[1]) + F.T.dot(invSigma * scaling_factor).dot(F))

    FK1Ft = F.dot(K1).dot(F.T)
    Xtilda_e = FK1Ft.dot(invSigma * scaling_factor).dot(enroll_ctr.stat1.T).T

    for t in range(test_ctr.stat1.shape[0]):
        xt = test_ctr.stat1[t,:]
        Pr = numpy.eye(F.shape[1]) - numpy.outer(test.stat1[t, :],test.stat1[t, :])
        Cunc = Pr.dot(Vtrans.transpose()).dot(numpy.diag(test_uncertainty[t, :]).dot(Vtrans)).dot(Pr)
        prec_den = scipy.linalg.inv(F.dot(F.T) + Sigma + Cunc)

        denom = -0.5 * xt.dot(prec_den).dot(xt) +0.5 * numpy.linalg.slogdet(prec_den)[1]
        prec_num = scipy.linalg.inv(FK1Ft+Sigma+Cunc)
        Xec = Xtilda_e - xt
        numer = -0.5 * numpy.einsum('ij, ji->i', Xec.dot(prec_num), Xec.T) + 0.5 * numpy.linalg.slogdet(prec_num)[1]
        scoremat[:, t] = scaling_factor * (numer - denom)

    # Compute verification scores
    score = Scores()
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask

    score.scoremat = scoremat

    # Case of open-set identification, we compute the log-likelihood
    # by taking into account the probability of having a known impostor
    # or an out-of set class
    if p_known != 0:
        N = score.scoremat.shape[0]
        open_set_scores = numpy.empty(score.scoremat.shape)
        tmp = numpy.exp(score.scoremat)
        for ii in range(N):
            # open-set term
            open_set_scores[ii, :] = score.scoremat[ii, :] \
                - numpy.log(p_known * tmp[~(numpy.arange(N) == ii)].sum(axis=0) / (N - 1) + (1 - p_known))
        score.scoremat = open_set_scores

    return score
