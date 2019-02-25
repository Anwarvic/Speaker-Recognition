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

    :mod:`jfa_scoring` provides methods to score using JFA model

"""
import copy
from sidekit.mixture import Mixture
from sidekit.statserver import StatServer
from sidekit.bosaris import Ndx
from sidekit.bosaris import Scores


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


def jfa_scoring(ubm, enroll, test, ndx, mean, sigma, V, U, D, batch_size=100, num_thread=1, check_missing=True):
    """Compute a verification score as a channel point estimate 
    of the log-likelihood ratio. Detail of this scoring can be found in 
    [Glembeck09].
    
    [Glembek09] Ondrej Glembek, Lukas Burget, Najim Dehak, Niko Brummer, 
    and Patrick Kenny, "Comparison of scoring methods used in speaker 
    recognition with joint factor analysis." 
    in Acoustics, Speech and Signal Processing, 2009. ICASSP 2009. 
    IEEE International Conference on. IEEE, 2009.

    Note that input statistics should not be whitened as
        it is done within this function.
    
    :param ubm: the Mixture object used to compute sufficient statistics
    :param enroll: a StatServer object which contains zero- and first-order
        statistics.
    :param test: a StatServer object which contains zero- and first-order
        statistics.
    :param ndx: an Ndx object which trial mask will be copied into the output
        Scores object
    :param mean: mean vector of the JFA model
    :param sigma: residual covariance vector of the JFA model
    :param V: between class covariance matrix of the JFA model
    :param U: within class covariance matrix of the JFA model
    :param D: MAP covariance matrix for the JFA model
    :param batch_size: size of the batch to reduce memory footprint
    :param num_thread: number of parallel process to run
    :param check_missing: boolean, if True, check that all model exist

    :return: a Scores object
    """
    assert isinstance(ubm, Mixture), '1st parameter must be a Mixture'
    assert isinstance(enroll, StatServer), '2nd parameter must be a StatServer'
    assert isinstance(test, StatServer), '3rd parameter must be a StatServer'
    assert isinstance(ndx, Ndx), '4th parameter shomustuld be a Ndx'

    # Remove missing models and test segments
    if check_missing:
        clean_ndx = _check_missing_model(enroll, test, ndx)
    else:
        clean_ndx = ndx

    print("taille de clean_ndx.trial_mask = {}".format(clean_ndx.trialmask.shape))

    # Sum enrolment statistics per model in case of multi-session
    enroll = enroll.sum_stat_per_model()[0]
    
    # Whiten enroll and test statistics
    enroll.whiten_stat1(ubm.get_mean_super_vector(), ubm.get_invcov_super_vector())
    test.whiten_stat1(ubm.get_mean_super_vector(), ubm.get_invcov_super_vector())
    
    # Estimate Vy and DZ from the enrollment
    trn_y, trn_x, trn_z = enroll.estimate_hidden(mean, sigma, V, U, D, batch_size=batch_size, num_thread=num_thread)
    M = ((trn_y.stat1.dot(V.T)) + (trn_z.stat1 * D))
    
    # Estimate Ux from the test
    tmp = copy.deepcopy(test)
    test_y, test_x, test_z = tmp.estimate_hidden(mean, sigma, None, U, None, batch_size, num_thread)
    
    # remove Ux weighted from the test statistics
    Ux = copy.deepcopy(test)
    Ux.stat1 = test_x.stat1.dot(U.T)
    test = test.subtract_weighted_stat1(Ux)
    
    # sum zero-order statistics for each test segment
    test_stat0_sum = test.stat0.sum(axis=1)
    
    # Compute score as the dot product of the enrollment supervector and the first 
    # order statistics divided by the sum of the zero-order stats
    scores = Scores()
    scores.modelset = enroll.modelset
    scores.segset = test.segset
    scores.scoremask = clean_ndx.trialmask
    scores.scoremat = M.dot((test.stat1 / test_stat0_sum[:, None]).T)    
    
    return scores
