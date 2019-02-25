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

:mod:`statserver` provides methods to manage zero and first statistics.
"""
import copy
import ctypes
import h5py
import logging
import multiprocessing
import numpy
import os
import scipy
import sys
import warnings

from sidekit.bosaris import IdMap
from sidekit.mixture import Mixture
from sidekit.features_server import FeaturesServer
from sidekit.sidekit_wrappers import process_parallel_lists, deprecated, check_path_existance
import sidekit.frontend
from sidekit import STAT_TYPE


ct = ctypes.c_double
if STAT_TYPE == numpy.float32:
    ct = ctypes.c_float


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def compute_llk(stat, V, sigma, U=None):
    # Compute Likelihood
    (n, d) = stat.stat1.shape
    centered_data = stat.stat1 - stat.get_mean_stat1()
    
    if sigma.ndim == 2:
        sigma_tot = numpy.dot(V, V.T) + sigma
    else:
        sigma_tot = numpy.dot(V, V.T) + numpy.diag(sigma)
    if U is not None:
        sigma_tot += numpy.dot(U, U.T)
    
    E, junk = scipy.linalg.eigh(sigma_tot)
    log_det = numpy.sum(numpy.log(E))

    return (-0.5 * (n * d * numpy.log(2 * numpy.pi) + n * log_det +
                    numpy.sum(numpy.sum(numpy.dot(centered_data,
                                                  scipy.linalg.inv(sigma_tot)) * centered_data, axis=1))))


def sum_log_probabilities(lp):
    """Sum log probabilities in a secure manner to avoid extreme values

    :param lp: ndarray of log-probabilities to sum
    """
    pp_max = numpy.max(lp, axis=1)
    log_lk = pp_max \
        + numpy.log(numpy.sum(numpy.exp((lp.transpose() - pp_max).transpose()), axis=1))
    ind = ~numpy.isfinite(pp_max)
    if sum(ind) != 0:
        log_lk[ind] = pp_max[ind]
    pp = numpy.exp((lp.transpose() - log_lk).transpose())
    return pp, log_lk


@process_parallel_lists
def fa_model_loop(batch_start,
                  mini_batch_indices,
                  r,
                  phi_white,
                  phi,
                  sigma,
                  stat0,
                  stat1,
                  e_h,
                  e_hh,
                  num_thread=1):
    """
    :param batch_start: index to start at in the list
    :param mini_batch_indices: indices of the elements in the list (should start at zero)
    :param r: rank of the matrix
    :param phi_white: whitened version of the factor matrix
    :param phi: non-whitened version of the factor matrix
    :param sigma: covariance matrix
    :param stat0: matrix of zero order statistics
    :param stat1: matrix of first order statistics
    :param e_h: accumulator
    :param e_hh: accumulator
    :param num_thread: number of parallel process to run
    """
    if sigma.ndim == 2:
        A = phi.T.dot(scipy.linalg.solve(sigma, phi)).astype(dtype=STAT_TYPE)

    tmp = numpy.zeros((phi.shape[1], phi.shape[1]), dtype=STAT_TYPE)

    for idx in mini_batch_indices:
        if sigma.ndim == 1:
            inv_lambda = scipy.linalg.inv(numpy.eye(r) + (phi_white.T * stat0[idx + batch_start, :]).dot(phi_white))
        else:
            inv_lambda = scipy.linalg.inv(stat0[idx + batch_start, 0] * A + numpy.eye(A.shape[0]))

        Aux = phi_white.T.dot(stat1[idx + batch_start, :])
        numpy.dot(Aux, inv_lambda, out=e_h[idx])
        e_hh[idx] = inv_lambda + numpy.outer(e_h[idx], e_h[idx], tmp)


@process_parallel_lists
def fa_distribution_loop(distrib_indices, _A, stat0, batch_start, batch_stop, e_hh, num_thread=1):
    """
    :param distrib_indices: indices of the distributions to iterate on
    :param _A: accumulator
    :param stat0: matrix of zero order statistics
    :param batch_start: index of the first session to process
    :param batch_stop: index of the last session to process
    :param e_hh: accumulator
    :param num_thread: number of parallel process to run
    """
    tmp = numpy.zeros((e_hh.shape[1], e_hh.shape[1]), dtype=STAT_TYPE)
    for c in distrib_indices:
        _A[c] += numpy.einsum('ijk,i->jk', e_hh, stat0[batch_start:batch_stop, c], out=tmp)
        # The line abov is equivalent to the two lines below:
        # tmp = (E_hh.T * stat0[batch_start:batch_stop, c]).T
        # _A[c] += numpy.sum(tmp, axis=0)


def load_existing_statistics_hdf5(statserver, statserver_file_name):
    """Load required statistics into the StatServer by reading from a file
        in hdf5 format.

    :param statserver: sidekit.StatServer to fill
    :param statserver_file_name: name of the file to read from
    """
    assert os.path.isfile(statserver_file_name), "statserver_file_name does not exist"

    # Load the StatServer
    ss = StatServer(statserver_file_name)

    # Check dimension consistency with current Stat_Server
    ok = True
    if statserver.stat0.shape[0] > 0:
        ok &= (ss.stat0.shape[0] == statserver.stat0.shape[1])
        ok &= (ss.stat1.shape[0] == statserver.stat1.shape[1])
    else:
        statserver.stat0 = numpy.zeros((statserver. modelset.shape[0], ss.stat0.shape[1]), dtype=STAT_TYPE)
        statserver.stat1 = numpy.zeros((statserver. modelset.shape[0], ss.stat1.shape[1]), dtype=STAT_TYPE)

    if ok:
        # For each segment, load statistics if they exist
        # Get the lists of existing segments
        seg_idx = [i for i in range(statserver. segset.shape[0]) if statserver.segset[i] in ss.segset]
        stat_idx = [numpy.where(ss.segset == seg)[0][0] for seg in statserver.segset if seg in ss.segset]

        # Copy statistics
        statserver.stat0[seg_idx, :] = ss.stat0[stat_idx, :]
        statserver.stat1[seg_idx, :] = ss.stat1[stat_idx, :]
    else:
        raise Exception('Mismatched statistic dimensions')


class StatServer:
    """A class for statistic storage and processing

    :attr modelset: list of model IDs for each session as an array of strings
    :attr segset: the list of session IDs as an array of strings
    :attr start: index of the first frame of the segment
    :attr stop: index of the last frame of the segment
    :attr stat0: a ndarray of float64. Each line contains 0-order statistics 
        from the corresponding session
    :attr stat1: a ndarray of float64. Each line contains 1-order statistics 
        from the corresponding session
    
    """

    #def __init__(self, statserver_file_name=None, ubm=None, index=None):$
    def __init__(self, statserver_file_name=None, distrib_nb=0, feature_size=0, index=None, ubm=None):
        """Initialize an empty StatServer or load a StatServer from an existing
        file.

        :param statserver_file_name: name of the file to read from. If filename
                is an empty string, the StatServer is initialized empty. 
                If filename is an IdMap object, the StatServer is initialized 
                to match the structure of the IdMap.
        """
        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.start = numpy.empty(0, dtype="|O")
        self.stop = numpy.empty(0, dtype="|O")
        self.stat0 = numpy.array([], dtype=STAT_TYPE)
        self.stat1 = numpy.array([], dtype=STAT_TYPE)

        if ubm is not None:
            distrib_nb = ubm.w.shape[0]
            feature_size = ubm.mu.shape[1]

        if statserver_file_name is None:
            pass
        # initialize
        elif isinstance(statserver_file_name, IdMap):
            self.modelset = statserver_file_name.leftids
            self.segset = statserver_file_name.rightids
            self.start = statserver_file_name.start
            self.stop = statserver_file_name.stop
            self.stat0 = numpy.empty((self.segset.shape[0], distrib_nb), dtype=STAT_TYPE)
            self.stat1 = numpy.empty((self.segset.shape[0], distrib_nb * feature_size), dtype=STAT_TYPE)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                tmp_stat0 = multiprocessing.Array(ct, self.stat0.size)
                self.stat0 = numpy.ctypeslib.as_array(tmp_stat0.get_obj())
                self.stat0 = self.stat0.reshape(self.segset.shape[0], distrib_nb)

                tmp_stat1 = multiprocessing.Array(ct, self.stat1.size)
                self.stat1 = numpy.ctypeslib.as_array(tmp_stat1.get_obj())
                self.stat1 = self.stat1.reshape(self.segset.shape[0], distrib_nb * feature_size)

        # initialize by reading an existing StatServer
        elif isinstance(statserver_file_name, str) and index is None:
            tmp = StatServer.read(statserver_file_name)
            self.modelset = tmp.modelset
            self.segset = tmp.segset
            self.start = tmp.start
            self.stop = tmp.stop
            self.stat0 = tmp.stat0.astype(STAT_TYPE)
            self.stat1 = tmp.stat1.astype(STAT_TYPE)

            with warnings.catch_warnings():
                size = self.stat0.shape
                warnings.simplefilter('ignore', RuntimeWarning)
                tmp_stat0 = multiprocessing.Array(ct, self.stat0.size)
                self.stat0 = numpy.ctypeslib.as_array(tmp_stat0.get_obj())
                self.stat0 = self.stat0.reshape(size)

                size = self.stat1.shape
                tmp_stat1 = multiprocessing.Array(ct, self.stat1.size)
                self.stat1 = numpy.ctypeslib.as_array(tmp_stat1.get_obj())
                self.stat1 = self.stat1.reshape(size)

            self.stat0 = copy.deepcopy(tmp.stat0).astype(STAT_TYPE)
            self.stat1 = copy.deepcopy(tmp.stat1).astype(STAT_TYPE)


        elif isinstance(statserver_file_name, str) and index is not None:
            tmp = StatServer.read_subset(statserver_file_name, index)
            self.modelset = tmp.modelset
            self.segset = tmp.segset
            self.start = tmp.start
            self.stop = tmp.stop
            self.stat0 = tmp.stat0.astype(STAT_TYPE)
            self.stat1 = tmp.stat1.astype(STAT_TYPE)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                tmp_stat0 = multiprocessing.Array(ct, self.stat0.size)
                self.stat0 = numpy.ctypeslib.as_array(tmp_stat0.get_obj())
                self.stat0 = self.stat0.reshape(self.segset.shape[0], distrib_nb)

                tmp_stat1 = multiprocessing.Array(ct, self.stat1.size)
                self.stat1 = numpy.ctypeslib.as_array(tmp_stat1.get_obj())
                self.stat1 = self.stat1.reshape(self.segset.shape[0], feature_size * distrib_nb)

            self.stat0 = copy.deepcopy(tmp.stat0).astype(STAT_TYPE)
            self.stat1 = copy.deepcopy(tmp.stat1).astype(STAT_TYPE)


    def __repr__(self):
        ch = '-' * 30 + '\n'
        ch += 'modelset: ' + self.modelset.__repr__() + '\n'
        ch += 'segset: ' + self.segset.__repr__() + '\n'
        ch += 'seg start:' + self.start.__repr__() + '\n'
        ch += 'seg stop:' + self.stop.__repr__() + '\n'
        ch += 'stat0:' + self.stat0.__repr__() + '\n'
        ch += 'stat1:' + self.stat1.__repr__() + '\n'
        ch += '-' * 30 + '\n'
        return ch

    def validate(self, warn=False):
        """Validate the structure and content of the StatServer. 
        Check consistency between the different attributes of 
        the StatServer:
        - dimension of the modelset
        - dimension of the segset
        - length of the modelset and segset
        - consistency of stat0 and stat1
        
        :param warn: bollean optional, if True, display possible warning
        """
        ok = self.modelset.ndim == 1 \
            and (self.modelset.shape == self.segset.shape == self.start.shape == self.stop.shape) \
            and (self.stat0.shape[0] == self.stat1.shape[0] == self.modelset.shape[0]) \
            and (not bool(self.stat1.shape[1] % self.stat0.shape[1]))

        if warn and (self.segset.shape != numpy.unique(self.segset).shape):
                logging.warning('Duplicated segments in StatServer')
        return ok

    def merge(*arg):
        """
        Merge a variable number of StatServers into one.
        If a pair segmentID is duplicated, keep ony one
        of them and raises a WARNING
        """
        line_number = 0
        for idx, ss in enumerate(arg):
            assert(isinstance(ss, sidekit.StatServer) and ss.validate()), "Arguments must be proper StatServers"
            
            # Check consistency of StatServers (dimension of the stat0 and stat1)
            if idx == 0:
                dim_stat0 = ss.stat0.shape[1]
                dim_stat1 = ss.stat1.shape[1]            
            else:
                assert(dim_stat0 == ss.stat0.shape[1] and 
                       dim_stat1 == ss.stat1.shape[1]), "Stat dimensions are not consistent"
    
            line_number += ss.modelset.shape[0]
    
        # Get a list of unique modelID-segmentID    
        id_list = []
        for ss in arg:
            id_list += list(ss.segset)
        id_set = set(id_list)
        if line_number != len(id_set):
            print("WARNING: duplicated segmentID in input StatServers")
        
        # Initialize the new StatServer with unique set of segmentID
        new_stat_server = sidekit.StatServer()
        new_stat_server.modelset = numpy.empty(len(id_set), dtype='object')
        new_stat_server.segset = numpy.array(list(id_set))
        new_stat_server.start = numpy.empty(len(id_set), 'object')
        new_stat_server.stop = numpy.empty(len(id_set), dtype='object')
        new_stat_server.stat0 = numpy.zeros((len(id_set), dim_stat0), dtype=STAT_TYPE)
        new_stat_server.stat1 = numpy.zeros((len(id_set), dim_stat1), dtype=STAT_TYPE)
        
        for ss in arg:
            for idx, segment in enumerate(ss.segset):
                new_idx = numpy.argwhere(new_stat_server.segset == segment)
                new_stat_server.modelset[new_idx] = ss.modelset[idx]
                new_stat_server.start[new_idx] = ss.start[idx]
                new_stat_server.stop[new_idx] = ss.stop[idx]
                new_stat_server.stat0[new_idx, :] = ss.stat0[idx, :].astype(STAT_TYPE)
                new_stat_server.stat1[new_idx, :] = ss.stat1[idx, :].astype(STAT_TYPE)
                
        assert(new_stat_server.validate()), "Problem in StatServer Merging"
        return new_stat_server

    @staticmethod
    def read(statserver_file_name, prefix=''):
        """Read StatServer in hdf5 format
        
        :param statserver_file_name: name of the file to read from
        :param prefix: prefixe of the dataset to read from in HDF5 file
        """
        with h5py.File(statserver_file_name, "r") as f:
            statserver = StatServer()
            statserver.modelset = f.get(prefix+"modelset").value
            statserver.segset = f.get(prefix+"segset").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                statserver.modelset = statserver.modelset.astype('U', copy=False)
                statserver.segset = statserver.segset.astype('U', copy=False)

            tmpstart = f.get(prefix+"start").value
            tmpstop = f.get(prefix+"stop").value
            statserver.start = numpy.empty(f[prefix+"start"].shape, '|O')
            statserver.stop = numpy.empty(f[prefix+"stop"].shape, '|O')
            statserver.start[tmpstart != -1] = tmpstart[tmpstart != -1]
            statserver.stop[tmpstop != -1] = tmpstop[tmpstop != -1]

            statserver.stat0 = f.get(prefix+"stat0").value.astype(dtype=STAT_TYPE)
            statserver.stat1 = f.get(prefix+"stat1").value.astype(dtype=STAT_TYPE)

            assert statserver.validate(), "Error: wrong StatServer format"
            return statserver

    @check_path_existance
    def write(self, output_file_name, prefix='', mode='w'):
        """Write the StatServer to disk in hdf5 format.
        
        :param output_file_name: name of the file to write in.
        :param prefix:
        """
        assert self.validate(), "Error: wrong StatServer format"

        file_already_exist = os.path.exists(output_file_name)

        start = copy.deepcopy(self.start)
        start[numpy.isnan(self.start.astype('float'))] = -1
        start = start.astype('int8', copy=False)

        stop = copy.deepcopy(self.stop)
        stop[numpy.isnan(self.stop.astype('float'))] = -1
        stop = stop.astype('int8', copy=False)

        with h5py.File(output_file_name, mode) as f:

            # If the file doesn't exist before, create it
            if mode == "w" or not file_already_exist:

                f.create_dataset(prefix+"modelset", data=self.modelset.astype('S'),
                                 maxshape=(None,),
                                 compression="gzip",
                                 fletcher32=True)
                f.create_dataset(prefix+"segset", data=self.segset.astype('S'),
                                 maxshape=(None,),
                                 compression="gzip",
                                 fletcher32=True)
                f.create_dataset(prefix+"stat0", data=self.stat0.astype(numpy.float32),
                                 maxshape=(None, self.stat0.shape[1]),
                                 compression="gzip",
                                 fletcher32=True)
                f.create_dataset(prefix+"stat1", data=self.stat1.astype(numpy.float32),
                                 maxshape=(None, self.stat1.shape[1]),
                                 compression="gzip",
                                 fletcher32=True)

                f.create_dataset(prefix+"start", data=start,
                                 maxshape=(None,),
                                 compression="gzip",
                                 fletcher32=True)
                f.create_dataset(prefix+"stop", data=stop,
                                 maxshape=(None,),
                                 compression="gzip",
                                 fletcher32=True)

            # If the file already exist, we extend all datasets and add the new data
            else:

                previous_size = f[prefix+"modelset"].shape[0]

                # Extend the size of each dataset
                f[prefix+"modelset"].resize((previous_size + self.modelset.shape[0],))
                f[prefix+"segset"].resize((previous_size + self.segset.shape[0],))
                f[prefix+"start"].resize((previous_size + start.shape[0],))
                f[prefix+"stop"].resize((previous_size + stop.shape[0],))
                f[prefix+"stat0"].resize((previous_size + self.stat0.shape[0], self.stat0.shape[1]))
                f[prefix+"stat1"].resize((previous_size + self.stat1.shape[0], self.stat1.shape[1]))

                # add the new data; WARNING: no check is done on the new data, beware of duplicated entries
                f[prefix+"modelset"][previous_size:] = self.modelset.astype('S')
                f[prefix+"segset"][previous_size:] = self.segset.astype('S')
                f[prefix+"start"][previous_size:] = start
                f[prefix+"stop"][previous_size:] = stop
                f[prefix+"stat0"][previous_size:, :] = self.stat0.astype(STAT_TYPE)
                f[prefix+"stat1"][previous_size:, :] = self.stat1.astype(STAT_TYPE)

    def get_model_stat0(self, mod_id):
        """Return zero-order statistics of a given model
        
        :param mod_id: ID of the model which stat0 will be returned
          
        :return: a matrix of zero-order statistics as a ndarray
        """
        S = self.stat0[self. modelset == mod_id, :]
        return S

    def get_model_stat1(self, mod_id):
        """Return first-order statistics of a given model
        
        :param mod_id: string, ID of the model which stat1 will be returned
          
        :return: a matrix of first-order statistics as a ndarray
        """
        return self.stat1[self.modelset == mod_id, :]

    def get_model_stat0_by_index(self, mod_idx):
        """Return zero-order statistics of model number modIDX
        
        :param mod_idx: integer, index of the unique model which stat0 will be
            returned
        
        :return: a matrix of zero-order statistics as a ndarray
        """
        return self.stat0[(self.modelset == numpy.unique(self.modelset)[mod_idx]), :]

    def get_model_stat1_by_index(self, mod_idx):
        """Return first-order statistics of model number modIDX
        
        :param mod_idx: integer, index of the unique model which stat1 will be
              returned
        
        :return: a matrix of first-order statistics as a ndarray
        """
        selectSeg = (self.modelset == numpy.unique(self.modelset)[mod_idx])
        return self.stat1[selectSeg, :]

    def get_segment_stat0(self, seg_id):
        """Return zero-order statistics of segment which ID is segID
        
        :param seg_id: string, ID of the segment which stat0 will be
              returned
        
        :return: a matrix of zero-order statistics as a ndarray
        """
        return self.stat0[self.segset == seg_id, :]

    def get_segment_stat1(self, seg_id):
        """Return first-order statistics of segment which ID is segID
        
        :param seg_id: string, ID of the segment which stat1 will be
              returned
        
        :return: a matrix of first-order statistics as a ndarray
        """
        return self.stat1[self.segset == seg_id, :]

    def get_segment_stat0_by_index(self, seg_idx):
        """Return zero-order statistics of segment number segIDX
        
        :param seg_idx: integer, index of the unique segment which stat0 will be
              returned
        
        :return: a matrix of zero-order statistics as a ndarray
        """
        return self.stat0[seg_idx, :]

    def get_segment_stat1_by_index(self, seg_idx):
        """Return first-order statistics of segment number segIDX
        
        :param seg_idx: integer, index of the unique segment which stat1 will be
              returned
        
        :return: a matrix of first-order statistics as a ndarray
        """
        return self.stat1[seg_idx, :]

    def get_model_segments(self, mod_id):
        """Return the list of segments belonging to model modID
        
        :param mod_id: string, ID of the model which belonging segments will be
              returned
        
        :return: a list of segments belonging to the model
        """
        return self.segset[self.modelset == mod_id]

    def get_model_segments_by_index(self, mod_idx):
        """Return the list of segments belonging to model number modIDX
        
        :param mod_idx: index of the model which list of segments will be
            returned
        
        :return: a list of segments belonging to the model
        """
        select_seg = (self.modelset == numpy.unique(self.modelset)[mod_idx])
        return self.segset[select_seg, :]

    def align_segments(self, segment_list):
        """Align segments of the current StatServer to match a list of segment 
            provided as input parameter. The size of the StatServer might be 
            reduced to match the input list of segments.
        
        :param segment_list: ndarray of strings, list of segments to match
        """
        indx = numpy.array([numpy.argwhere(self.segset == v)[0][0] for v in segment_list])
        self.segset = self.segset[indx]
        self.modelset = self.modelset[indx]
        self.start = self.start[indx]
        self.stop = self.stop[indx]
        self.stat0 = self.stat0[indx, :]
        self.stat1 = self.stat1[indx, :]
        
    def align_models(self, model_list):
        """Align models of the current StatServer to match a list of models 
            provided as input parameter. The size of the StatServer might be 
            reduced to match the input list of models.
        
        :param model_list: ndarray of strings, list of models to match
        """
        indx = numpy.array([numpy.argwhere(self.modelset == v)[0][0] for v in model_list])
        self.segset = self.segset[indx]
        self.modelset = self.modelset[indx]
        self.start = self.start[indx]
        self.stop = self.stop[indx]
        self.stat0 = self.stat0[indx, :]
        self.stat1 = self.stat1[indx, :]

    @process_parallel_lists
    def accumulate_stat(self, ubm, feature_server, seg_indices=None, channel_extension=("", "_b"), num_thread=1):
        """Compute statistics for a list of sessions which indices 
            are given in segIndices.
        
        :param ubm: a Mixture object used to compute the statistics
        :param feature_server: featureServer object
        :param seg_indices: list of indices of segments to process
              if segIndices is an empty list, process all segments.
        :param channel_extension: tuple of strings, extension of first and second channel for stereo files, default
        is ("", "_b")
        :param num_thread: number of parallel process to run
        """
        assert isinstance(ubm, Mixture), 'First parameter has to be a Mixture'
        assert isinstance(feature_server, FeaturesServer), 'Second parameter has to be a FeaturesServer'

        if (seg_indices is None) \
                or (self.stat0.shape[0] != self.segset.shape[0]) \
                or (self.stat1.shape[0] != self.segset.shape[0]):
            self.stat0 = numpy.zeros((self.segset.shape[0], ubm.distrib_nb()), dtype=STAT_TYPE)
            self.stat1 = numpy.zeros((self.segset.shape[0], ubm.sv_size()), dtype=STAT_TYPE)
            seg_indices = range(self.segset.shape[0])
        feature_server.keep_all_features = True

        for count, idx in enumerate(seg_indices):
            logging.debug('Compute statistics for {}'.format(self.segset[idx]))

            show = self.segset[idx]

            # If using a FeaturesExtractor, get the channel number by checking the extension of the show
            channel = 0
            if feature_server.features_extractor is not None and show.endswith(channel_extension[1]):
                channel = 1
            show = show[:show.rfind(channel_extension[channel])]

            #ANWAR(ADD)
            cep, vad = feature_server.load(show, channel=channel, input_feature_filename=feature_server.feature_filename_structure)
            # cep, vad = feature_server.load(show, channel=channel)
            #END
            # cep, vad = feature_server.load(show, channel=channel)
            stop = vad.shape[0] if self.stop[idx] is None else min(self.stop[idx], vad.shape[0])
            logging.info('{} start: {} stop: {}'.format(show, self.start[idx], stop))
            data = cep[self.start[idx]:stop, :]
            data = data[vad[self.start[idx]:stop], :]

            # Verify that frame dimension is equal to gmm dimension
            if not ubm.dim() == data.shape[1]:
                raise Exception('dimension of ubm and features differ: {:d} / {:d}'.format(ubm.dim(), data.shape[1]))
            else:
                if ubm.invcov.ndim == 2:
                    lp = ubm.compute_log_posterior_probabilities(data)
                else:
                    lp = ubm.compute_log_posterior_probabilities_full(data)
                pp, foo = sum_log_probabilities(lp)
                # Compute 0th-order statistics
                self.stat0[idx, :] = pp.sum(0)
                # Compute 1st-order statistics
                self.stat1[idx, :] = numpy.reshape(numpy.transpose(
                        numpy.dot(data.transpose(), pp)), ubm.sv_size()).astype(STAT_TYPE)

    def get_mean_stat1(self):
        """Return the mean of first order statistics
        
        return: the mean array of the first order statistics.
        """
        mu = numpy.mean(self.stat1, axis=0)
        return mu

    def norm_stat1(self):
        """Divide all first-order statistics by their euclidian norm."""
        vect_norm = numpy.clip(numpy.linalg.norm(self.stat1, axis=1), 1e-08, numpy.inf)
        self.stat1 = (self.stat1.transpose() / vect_norm).transpose()

    def rotate_stat1(self, R):
        """Rotate first-order statistics by a right-product.
        
        :param R: ndarray, matrix to use for right product on the first order 
            statistics.
        """
        self.stat1 = numpy.dot(self.stat1, R)

    def center_stat1(self, mu):
        """Center first order statistics.
        
        :param mu: array to center on.
        """
        dim = self.stat1.shape[1] / self.stat0.shape[1]
        index_map = numpy.repeat(numpy.arange(self.stat0.shape[1]), dim)
        self.stat1 = self.stat1 - (self.stat0[:, index_map] * mu.astype(STAT_TYPE))

    def subtract_weighted_stat1(self, sts):
        """Subtract the stat1 from from the sts StatServer to the stat1 of 
        the current StatServer after multiplying by the zero-order statistics
        from the current statserver
        
        :param sts: a StatServer
        
        :return: a new StatServer
        """
        new_sts = copy.deepcopy(self)
        
        # check the compatibility of the two statservers
        #   exact match of the sessions and dimensions of the stat0 and stat1
        if all(numpy.sort(sts.modelset) == numpy.sort(self.modelset))and \
                all(numpy.sort(sts.segset) == numpy.sort(self.segset)) and \
                (sts.stat0.shape == self.stat0.shape) and \
                (sts.stat1.shape == self.stat1.shape):
    
            # align sts according to self.segset
            idx = self.segset.argsort()
            idx_sts = sts.segset.argsort()
            new_sts.stat1[idx, :] = sts.stat1[idx_sts, :]
            
            # Subtract the stat1
            dim = self.stat1.shape[1] / self.stat0.shape[1]
            index_map = numpy.repeat(numpy.arange(self.stat0.shape[1]), dim)
            new_sts.stat1 = self.stat1 - (self.stat0[:, index_map] * new_sts.stat1)
            
        else:
            raise Exception('Statserver are not compatible')
        
        return new_sts

    def whiten_stat1(self, mu, sigma, isSqrInvSigma=False):
        """Whiten first-order statistics
        If sigma.ndim == 1, case of a diagonal covariance
        If sigma.ndim == 2, case of a single Gaussian with full covariance
        If sigma.ndim == 3, case of a full covariance UBM
        
        :param mu: array, mean vector to be subtracted from the statistics
        :param sigma: narray, co-variance matrix or covariance super-vector
        :param isSqrInvSigma: boolean, True if the input Sigma matrix is the inverse of the square root of a covariance
         matrix
        """
        if sigma.ndim == 1:
            self.center_stat1(mu)
            self.stat1 = self.stat1 / numpy.sqrt(sigma.astype(STAT_TYPE))

        elif sigma.ndim == 2:
            # Compute the inverse square root of the co-variance matrix Sigma
            sqr_inv_sigma = sigma
            
            if not isSqrInvSigma:
                eigen_values, eigen_vectors = scipy.linalg.eigh(sigma)
                ind = eigen_values.real.argsort()[::-1]
                eigen_values = eigen_values.real[ind]
                eigen_vectors = eigen_vectors.real[:, ind]
            
                sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
                sqr_inv_sigma = numpy.dot(eigen_vectors, numpy.diag(sqr_inv_eval_sigma))
            else:
                pass

            # Whitening of the first-order statistics
            self.center_stat1(mu)
            self.rotate_stat1(sqr_inv_sigma)

        elif sigma.ndim == 3:
            # we assume that sigma is a 3D ndarray of size D x n x n
            # where D is the number of distributions and n is the dimension of a single distibution
            n = self.stat1.shape[1] // self.stat0.shape[1]
            sess_nb = self.stat0.shape[0]
            self.center_stat1(mu)
            self.stat1 = numpy.einsum("ikj,ikl->ilj",
                                      self.stat1.T.reshape(-1, n, sess_nb), sigma).reshape(-1, sess_nb).T

        else:
            raise Exception('Wrong dimension of Sigma, must be 1 or 2')
            
    def whiten_cholesky_stat1(self, mu, sigma):
        """Whiten first-order statistics by using Cholesky decomposition of 
        Sigma
        
        :param mu: array, mean vector to be subtracted from the statistics
        :param sigma: narray, co-variance matrix or covariance super-vector
        """
        if sigma.ndim == 2:
            # Compute the inverse square root of the co-variance matrix Sigma
            inv_sigma = scipy.linalg.inv(sigma)
            chol_invcov = scipy.linalg.cholesky(inv_sigma).T

            # Whitening of the first-order statistics
            self.center_stat1(mu)
            self.stat1 = self.stat1.dot(chol_invcov)            

        elif sigma.ndim == 1:
            self.center_stat1(mu)
            self.stat1 = self.stat1 / numpy.sqrt(sigma)
        else:
            raise Exception('Wrong dimension of Sigma, must be 1 or 2')

    def get_total_covariance_stat1(self):
        """Compute and return the total covariance matrix of the first-order 
            statistics.
        
        :return: the total co-variance matrix of the first-order statistics 
                as a ndarray.
        """
        C = self.stat1 - self.stat1.mean(axis=0)
        return numpy.dot(C.transpose(), C) / self.stat1.shape[0]

    def get_within_covariance_stat1(self):
        """Compute and return the within-class covariance matrix of the 
            first-order statistics.
        
        :return: the within-class co-variance matrix of the first-order statistics 
              as a ndarray.
        """
        vect_size = self.stat1.shape[1]
        unique_speaker = numpy.unique(self.modelset)
        W = numpy.zeros((vect_size, vect_size))

        for speakerID in unique_speaker:
            spk_ctr_vec = self.get_model_stat1(speakerID) \
                        - numpy.mean(self.get_model_stat1(speakerID), axis=0)
            W += numpy.dot(spk_ctr_vec.transpose(), spk_ctr_vec)
        W /= self.stat1.shape[0]
        return W

    def get_between_covariance_stat1(self):
        """Compute and return the between-class covariance matrix of the 
            first-order statistics.
        
        :return: the between-class co-variance matrix of the first-order 
            statistics as a ndarray.
        """
        vect_size = self.stat1.shape[1]
        unique_speaker = numpy.unique(self.modelset)
        B = numpy.zeros((vect_size, vect_size))

        # Compute overall mean first-order statistics
        mu = self.get_mean_stat1()

        # Compute and accumulate mean first-order statistics for each class
        for speaker_id in unique_speaker:
            spk_sessions = self.get_model_stat1(speaker_id)
            tmp = numpy.mean(spk_sessions, axis=0) - mu
            B += (spk_sessions.shape[0] * numpy.outer(tmp, tmp))
        B /= self.stat1.shape[0]
        return B

    def get_lda_matrix_stat1(self, rank):
        """Compute and return the Linear Discriminant Analysis matrix 
            on the first-order statistics. Columns of the LDA matrix are ordered
            according to the corresponding eigenvalues in descending order.
        
        :param rank: integer, rank of the LDA matrix to return
        
        :return: the LDA matrix of rank "rank" as a ndarray
        """
        vect_size = self.stat1.shape[1]
        unique_speaker = numpy.unique(self.modelset)

        mu = self.get_mean_stat1()

        class_means = numpy.zeros((unique_speaker.shape[0], vect_size))
        Sw = numpy.zeros((vect_size, vect_size))

        spk_idx = 0
        for speaker_id in unique_speaker:
            spk_sessions = self.get_model_stat1(speaker_id) \
                        - numpy.mean(self.get_model_stat1(speaker_id), axis=0)
            Sw += numpy.dot(spk_sessions.transpose(), spk_sessions) / spk_sessions.shape[0]
            class_means[spk_idx, :] = numpy.mean(self.get_model_stat1(speaker_id), axis=0)
            spk_idx += 1

        # Compute Between-class scatter matrix
        class_means = class_means - mu
        Sb = numpy.dot(class_means.transpose(), class_means)

        # Compute the Eigenvectors & eigenvalues of the discrimination matrix
        DiscriminationMatrix = numpy.dot(Sb, scipy.linalg.inv(Sw)).transpose()
        eigen_values, eigen_vectors = scipy.linalg.eigh(DiscriminationMatrix)
        eigen_values = eigen_values.real
        eigen_vectors = eigen_vectors.real

        # Rearrange the eigenvectors according to decreasing eigenvalues
        # get indexes of the rank top eigen values
        idx = eigen_values.real.argsort()[-rank:][::-1]
        L = eigen_vectors[:, idx]
        return L

    def get_mahalanobis_matrix_stat1(self):
        """Compute and return Mahalanobis matrix of first-order statistics.
        
        :return: the mahalanobis matrix computed on the first-order 
            statistics as a ndarray
        """
        W = self.get_within_covariance_stat1()
        M = scipy.linalg.inv(W)
        return M

    def get_wccn_choleski_stat1(self):
        """Compute and return the lower Cholesky decomposition matrix of the
            Within Class Co-variance Normalization matrix on the first-order
            statistics.
        
        :return: the lower Choleski decomposition of the WCCN matrix 
            as a ndarray
        """
        vect_size = self.stat1.shape[1]
        unique_speaker = numpy.unique(self.modelset)
        WCCN = numpy.zeros((vect_size, vect_size))

        for speaker_id in unique_speaker:
            spk_ctr_vec = self.get_model_stat1(speaker_id) \
                      - numpy.mean(self.get_model_stat1(speaker_id), axis=0)
            #WCCN += numpy.dot(spk_ctr_vec.transpose(), spk_ctr_vec)
            WCCN += numpy.dot(spk_ctr_vec.transpose(), spk_ctr_vec) / spk_ctr_vec.shape[0]

        #WCCN /= self.stat1.shape[0]
        WCCN = WCCN / unique_speaker.shape[0]

        # Choleski decomposition of the WCCN matrix
        invW = scipy.linalg.inv(WCCN)
        W = scipy.linalg.cholesky(invW).T
        return W

    def get_nap_matrix_stat1(self, co_rank):
        """Compute return the Nuisance Attribute Projection matrix
            from first-order statistics.
        
        :param co_rank: co-rank of the Nuisance Attribute Projection matrix
        
        :return: the NAP matrix of rank "coRank"
        """
        vectSize = self.stat1.shape[1]
        W = numpy.dot(self.stat1, self.stat1.transpose()) / vectSize
        eigenValues, eigenVectors = scipy.linalg.eigh(W)

        # Rearrange the eigenvectors according to decreasing eigenvalues
        # get indexes of the rank top eigen values
        idx = eigenValues.real.argsort()[-co_rank:][::-1]
        N = numpy.dot(self.stat1.transpose(), eigenVectors[:, idx])
        N = numpy.dot(N, numpy.diag(1 / numpy.sqrt(vectSize * eigenValues.real[idx])))
        return N

    def adapt_mean_map(self, ubm, r=16, norm=False):
        """Maximum A Posteriori adaptation of the mean super-vector of ubm,
            train one model per segment.
        
        :param ubm: a Mixture object to adapt
        :param r: float, the relevant factor for MAP adaptation
        :param norm: boolean, normalize by using the UBM co-variance. 
            Default is False
          
        :return: a StatServer with 1 as stat0 and the MAP adapted super-vectors 
              as stat1
        """
        gsv_statserver = StatServer()
        gsv_statserver.modelset = self.modelset
        gsv_statserver.segset = self.segset
        gsv_statserver.start = self.start
        gsv_statserver.stop = self.stop
        gsv_statserver.stat0 = numpy.ones((self.segset. shape[0], 1), dtype=STAT_TYPE)

        index_map = numpy.repeat(numpy.arange(ubm.distrib_nb()), ubm.dim())

        # Adapt mean vectors
        alpha = (self.stat0 + numpy.finfo(numpy.float32).eps) / (self.stat0 + numpy.finfo(numpy.float32).eps + r)
        M = self.stat1 / self.stat0[:, index_map]
        M[numpy.isnan(M)] = 0  # Replace NaN due to divide by zeros
        M = alpha[:, index_map] * M + (1 - alpha[:, index_map]) * \
                                      numpy.tile(ubm.get_mean_super_vector(), (M.shape[0], 1))

        if norm:
            if ubm.invcov.ndim == 2:
                # Normalization corresponds to KL divergence
                w = numpy.repeat(ubm.w, ubm.dim())
                KLD = numpy.sqrt(w * ubm.get_invcov_super_vector())

            M = M * KLD

        gsv_statserver.stat1 = M.astype(dtype=STAT_TYPE)
        gsv_statserver.validate()
        return gsv_statserver

    def adapt_mean_map_multisession(self, ubm, r=16, norm=False):
        """Maximum A Posteriori adaptation of the mean super-vector of ubm,
            train one model per model in the modelset by summing the statistics
            of the multiple segments.
        
        :param ubm: a Mixture object to adapt 
        :param r: float, the relevant factor for MAP adaptation
        :param norm: boolean, normalize by using the UBM co-variance. 
            Default is False
          
        :return: a StatServer with 1 as stat0 and the MAP adapted super-vectors 
              as stat1
        """
        gsv_statserver = StatServer()
        gsv_statserver.modelset = numpy.unique(self.modelset)
        gsv_statserver.segset = numpy.unique(self.modelset)
        gsv_statserver.start = numpy.empty(gsv_statserver.modelset.shape, dtype="|O")
        gsv_statserver.stop = numpy.empty(gsv_statserver.modelset.shape, dtype="|O")
        gsv_statserver.stat0 = numpy.ones((numpy.unique(self.modelset).shape[0], 1), dtype=STAT_TYPE)

        index_map = numpy.repeat(numpy.arange(ubm.distrib_nb()), ubm.dim())

        # Sum the statistics per model
        modelStat = self.sum_stat_per_model()[0]
        
        # Adapt mean vectors
        alpha = modelStat.stat0 / (modelStat.stat0 + r)
        M = modelStat.stat1 / modelStat.stat0[:, index_map]
        M[numpy.isnan(M)] = 0  # Replace NaN due to divide by zeros
        M = alpha[:, index_map] * M \
            + (1 - alpha[:, index_map]) * numpy.tile(ubm.get_mean_super_vector(), (M.shape[0], 1))

        if norm:
            if ubm.invcov.ndim == 2:
                # Normalization corresponds to KL divergence
                w = numpy.repeat(ubm.w, ubm.dim())
                KLD = numpy.sqrt(w * ubm.get_invcov_super_vector())

            M = M * KLD

        gsv_statserver.stat1 = M.astype(dtype=STAT_TYPE)
        gsv_statserver.validate()
        return gsv_statserver

    def precompute_svm_kernel_stat1(self):
        """Pre-compute the Kernel for SVM training and testing,
            the output parameter is a matrix that only contains the impostor
            part of the Kernel. This one has to be completed by the
            target-dependent part during training and testing.
        
        :return: the impostor part of the SVM Graam matrix as a ndarray
        """
        K = numpy.dot(self.stat1, self.stat1.transpose())
        return K

    def ivector_extraction_weight(self, ubm, W, Tnorm, delta=numpy.array([])):
        """Compute i-vectors using the ubm weight approximation.
            For more information, refers to:
            
            Glembeck, O.; Burget, L.; Matejka, P.; Karafiat, M. & Kenny, P. 
            "Simplification and optimization of I-Vector extraction," 
            in IEEE International Conference on Acoustics, Speech, and Signal 
            Processing, ICASSP, 2011, 4516-4519
        
        :param ubm: a Mixture used as UBM for i-vector estimation
        :param W: fix matrix pre-computed using the weights from 
            the UBM and the total variability matrix
        :param Tnorm: total variability matrix pre-normalized using 
                the co-variance of the UBM
        :param delta: men vector if re-estimated using minimum divergence 
                criteria
        
        :return: a StatServer which zero-order statistics are 1 
                and first-order statistics are approximated i-vectors.
        """
        # check consistency of dimensions for delta, Tnorm, W, ubm
        assert ubm.get_invcov_super_vector().shape[0] == Tnorm.shape[0], \
            'UBM and TV matrix dimension are not consistent'
        if delta.shape == (0, ):
            delta = numpy.zeros(ubm.get_invcov_super_vector().shape)
        assert ubm.get_invcov_super_vector().shape[0] == delta.shape[0],\
            'Minimum divergence mean and TV matrix dimension not consistent'
        assert W.shape[0] == Tnorm.shape[1], 'W and TV matrix dimension are not consistent'
        ivector_size = Tnorm.shape[1]
    
        # Sum stat0
        sumStat0 = self.stat0.sum(axis=1)
    
        # Center and normalize first-order statistics 
        # for the case of diagonal covariance UBM
        self.whiten_stat1(delta, 1./ubm.get_invcov_super_vector())
    
        X = numpy.dot(self.stat1, Tnorm)
    
        enroll_iv = StatServer()
        enroll_iv.modelset = self.modelset
        enroll_iv.segset = self.segset
        enroll_iv.stat0 = numpy.ones((enroll_iv.segset.shape[0], 1))
        enroll_iv.stat1 = numpy.zeros((enroll_iv.segset.shape[0], ivector_size))
        for iv in range(self.stat0.shape[0]):  # loop on i-vector
            logging.debug('Estimate i-vector [ %d / %d ]', iv + 1, self.stat0.shape[0])
            # Compute precision matrix
            L = numpy.eye(ivector_size) + sumStat0[iv] * W
            # Estimate i-vector
            enroll_iv.stat1[iv, :] = scipy.linalg.solve(L, X[iv, :])

        return enroll_iv

    def ivector_extraction_eigen_decomposition(self,
                                               ubm,
                                               Q,
                                               D_bar_c,
                                               Tnorm,
                                               delta=numpy.array([])):
        """Compute i-vectors using the eigen decomposition approximation.
            For more information, refers to[Glembeck09]_
        
        :param ubm: a Mixture used as UBM for i-vector estimation
        :param Q: Q matrix as described in [Glembeck11]
        :param D_bar_c: matrices as described in [Glembeck11]
        :param Tnorm: total variability matrix pre-normalized using 
                the co-variance of the UBM
        :param delta: men vector if re-estimated using minimum divergence 
                criteria
        
        :return: a StatServer which zero-order statistics are 1 
                and first-order statistics are approximated i-vectors.
        """
        # check consistency of dimensions for delta, Tnorm, Q, D_bar_c, ubm
        assert ubm.get_invcov_super_vector().shape[0] == Tnorm.shape[0], \
            'UBM and TV matrix dimension not consistent'
        if delta.shape == (0, ):
            delta = numpy.zeros(ubm.get_invcov_super_vector().shape)
        assert ubm.get_invcov_super_vector().shape[0] == delta.shape[0], \
            'Minimum divergence mean and TV matrix dimension not consistent'
        assert D_bar_c.shape[1] == Tnorm.shape[1], \
            'D_bar_c and TV matrix dimension are not consistent'
        assert D_bar_c.shape[0] == ubm.w.shape[0], \
            'D_bar_c and UBM dimension are not consistent'
    
        ivector_size = Tnorm.shape[1]

        # Center and normalize first-order statistics 
        # for the case of diagonal covariance UBM
        self.whiten_stat1(delta, 1./ubm.get_invcov_super_vector())
    
        X = numpy.dot(self.stat1, Tnorm)
    
        enroll_iv = StatServer()
        enroll_iv.modelset = self.modelset
        enroll_iv.segset = self.segset
        enroll_iv.stat0 = numpy.ones((enroll_iv.segset.shape[0], 1), dtype=STAT_TYPE)
        enroll_iv.stat1 = numpy.zeros((enroll_iv.segset.shape[0], ivector_size), dtype=STAT_TYPE)
        for iv in range(self.stat0.shape[0]):  # loop on i-vector
            logging.debug('Estimate i-vector [ %d / %d ]', iv + 1, self.stat0.shape[0])

            # Compute precision matrix
            diag_L = 1 + numpy.sum(numpy.dot(numpy.diag(self.stat0[iv, :]), D_bar_c), axis=0)

            # Estimate i-vector
            enroll_iv.stat1[iv, :] = X[iv, :].dot(Q).dot(numpy.diag(1/diag_L)).dot(Q.transpose())

        return enroll_iv

    def estimate_spectral_norm_stat1(self, it=1, mode='efr'):
        """Compute meta-parameters for Spectral Normalization as described
            in [Bousquet11]_
            
            Can be used to perform Eigen Factor Radial or Spherical Nuisance
            Normalization. Default behavior is equivalent to Length Norm as 
            described in [Garcia-Romero11]_
            
            Statistics are transformed while the meta-parameters are 
            estimated.
        
        :param it: integer, number of iterations to perform
        :param mode: string, can be 
                - efr for Eigen Factor Radial
                - sphNorm, for Spherical Nuisance Normalization
                  
        :return: a tupple of two lists:
                - a list of mean vectors
                - a list of co-variance matrices as ndarrays
        """
        spectral_norm_mean = []
        spectral_norm_cov = []
        tmp_iv = copy.deepcopy(self)
        
        for i in range(it):
            # estimate mean and covariance matrix
            spectral_norm_mean.append(tmp_iv.get_mean_stat1())
            
            if mode == 'efr':
                spectral_norm_cov.append(tmp_iv.get_total_covariance_stat1())
            elif mode == 'sphNorm':
                spectral_norm_cov.append(tmp_iv.get_within_covariance_stat1())

            # Center and whiten the statistics
            tmp_iv.whiten_stat1(spectral_norm_mean[i], spectral_norm_cov[i])
            tmp_iv.norm_stat1()
        return spectral_norm_mean, spectral_norm_cov

    def spectral_norm_stat1(self, spectral_norm_mean, spectral_norm_cov, is_sqr_inv_sigma=False):
        """Apply Spectral Sormalization to all first order statistics.
            See more details in [Bousquet11]_
            
            The number of iterations performed is equal to the length of the
            input lists.
        
        :param spectral_norm_mean: a list of mean vectors
        :param spectral_norm_cov: a list of co-variance matrices as ndarrays
        :param is_sqr_inv_sigma: boolean, True if
        """
        assert len(spectral_norm_mean) == len(spectral_norm_cov), \
            'Number of mean vectors and covariance matrices is different'

        for mu, Cov in zip(spectral_norm_mean, spectral_norm_cov):
            self.whiten_stat1(mu, Cov, is_sqr_inv_sigma)
            self.norm_stat1()

    def sum_stat_per_model(self):
        """Sum the zero- and first-order statistics per model and store them 
        in a new StatServer.        
        
        :return: a StatServer with the statistics summed per model
        """
        sts_per_model = sidekit.StatServer()
        sts_per_model.modelset = numpy.unique(self.modelset)
        sts_per_model.segset = sts_per_model.modelset
        sts_per_model.stat0 = numpy.zeros((sts_per_model.modelset.shape[0], self.stat0.shape[1]), dtype=STAT_TYPE)
        sts_per_model.stat1 = numpy.zeros((sts_per_model.modelset.shape[0], self.stat1.shape[1]), dtype=STAT_TYPE)
        sts_per_model.start = numpy.empty(sts_per_model.segset.shape, '|O')
        sts_per_model.stop = numpy.empty(sts_per_model.segset.shape, '|O')
        
        session_per_model = numpy.zeros(numpy.unique(self.modelset).shape[0])

        for idx, model in enumerate(sts_per_model.modelset):
            sts_per_model.stat0[idx, :] = self.get_model_stat0(model).sum(axis=0)
            sts_per_model.stat1[idx, :] = self.get_model_stat1(model).sum(axis=0)
            session_per_model[idx] += self.get_model_stat1(model).shape[0]
        return sts_per_model, session_per_model

    def mean_stat_per_model(self):
        """Average the zero- and first-order statistics per model and store them
        in a new StatServer.

        :return: a StatServer with the statistics averaged per model
        """
        sts_per_model = sidekit.StatServer()
        sts_per_model.modelset = numpy.unique(self.modelset)
        sts_per_model.segset = sts_per_model.modelset
        sts_per_model.stat0 = numpy.zeros((sts_per_model.modelset.shape[0], self.stat0.shape[1]), dtype=STAT_TYPE)
        sts_per_model.stat1 = numpy.zeros((sts_per_model.modelset.shape[0], self.stat1.shape[1]), dtype=STAT_TYPE)
        sts_per_model.start = numpy.empty(sts_per_model.segset.shape, '|O')
        sts_per_model.stop = numpy.empty(sts_per_model.segset.shape, '|O')

        for idx, model in enumerate(sts_per_model.modelset):
            sts_per_model.stat0[idx, :] = self.get_model_stat0(model).mean(axis=0)
            sts_per_model.stat1[idx, :] = self.get_model_stat1(model).mean(axis=0)
        return sts_per_model

    def _expectation(self, phi, mean, sigma, session_per_model, batch_size=100, num_thread=1):
        """
        dans cette version, on considère que les stats NE sont PAS blanchis avant
        """
        warnings.warn("deprecated, use FactorAnalyser module", DeprecationWarning)

        r = phi.shape[-1]
        d = int(self.stat1.shape[1] / self.stat0.shape[1])
        C = self.stat0.shape[1]

        """Whiten the statistics and multiply the covariance matrix by the 
        square root of the inverse of the residual covariance"""
        self.whiten_stat1(mean, sigma)
        phi_white = copy.deepcopy(phi)
        if sigma.ndim == 2:
            eigen_values, eigen_vectors = scipy.linalg.eigh(sigma)
            ind = eigen_values.real.argsort()[::-1]
            eigen_values = eigen_values.real[ind]
            eigen_vectors = eigen_vectors.real[:, ind]
            sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
            sqr_inv_sigma = numpy.dot(eigen_vectors, numpy.diag(sqr_inv_eval_sigma))
            phi_white = sqr_inv_sigma.T.dot(phi)
        elif sigma.ndim == 1:
            sqr_inv_sigma = 1/numpy.sqrt(sigma)
            phi_white = phi * sqr_inv_sigma[:, None]
            
        # Replicate self.stat0
        index_map = numpy.repeat(numpy.arange(C), d)
        _stat0 = self.stat0[:, index_map]

        # Create accumulators for the list of models to process
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _A = numpy.zeros((C, r, r), dtype=STAT_TYPE)
            tmp_A = multiprocessing.Array(ct, _A.size)
            _A = numpy.ctypeslib.as_array(tmp_A.get_obj())
            _A = _A.reshape(C, r, r)

        _C = numpy.zeros((r, d * C), dtype=STAT_TYPE)
        
        _R = numpy.zeros((r, r), dtype=STAT_TYPE)
        _r = numpy.zeros(r, dtype=STAT_TYPE)

        # Process in batches in order to reduce the memory requirement
        batch_nb = int(numpy.floor(self.segset.shape[0]/float(batch_size) + 0.999))
        
        for batch in range(batch_nb):
            batch_start = batch * batch_size
            batch_stop = min((batch + 1) * batch_size, self.segset.shape[0])
            batch_len = batch_stop - batch_start

            # Allocate the memory to save time
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                e_h = numpy.zeros((batch_len, r), dtype=STAT_TYPE)
                tmp_e_h = multiprocessing.Array(ct, e_h.size)
                e_h = numpy.ctypeslib.as_array(tmp_e_h.get_obj())
                e_h = e_h.reshape(batch_len, r)

                e_hh = numpy.zeros((batch_len, r, r), dtype=STAT_TYPE)
                tmp_e_hh = multiprocessing.Array(ct, e_hh.size)
                e_hh = numpy.ctypeslib.as_array(tmp_e_hh.get_obj())
                e_hh = e_hh.reshape(batch_len, r, r)

            # loop on model id's
            fa_model_loop(batch_start=batch_start, mini_batch_indices=numpy.arange(batch_len),
                          r=r, phi_white=phi_white, phi=phi, sigma=sigma,
                          stat0=_stat0, stat1=self.stat1,
                          e_h=e_h, e_hh=e_hh, num_thread=num_thread)
            
            # Accumulate for minimum divergence step
            _r += numpy.sum(e_h * session_per_model[batch_start:batch_stop, None], axis=0)
            _R += numpy.sum(e_hh, axis=0)

            if sqr_inv_sigma.ndim == 2:
                _C += e_h.T.dot(self.stat1[batch_start:batch_stop, :]).dot(scipy.linalg.inv(sqr_inv_sigma))
            elif sqr_inv_sigma.ndim == 1:
                _C += e_h.T.dot(self.stat1[batch_start:batch_stop, :]) / sqr_inv_sigma
 
            # Parallelized loop on the model id's
            fa_distribution_loop(distrib_indices=numpy.arange(C),
                                 _A=_A,
                                 stat0=self.stat0,
                                 batch_start=batch_start,
                                 batch_stop=batch_stop,
                                 e_hh=e_hh,
                                 num_thread=num_thread)

        _r /= session_per_model.sum()
        _R /= session_per_model.shape[0]
        
        return _A, _C, _R  

    def _maximization(self, phi, _A, _C, _R=None, sigma_obs=None, session_number=None):
        """
        """
        warnings.warn("deprecated, use FactorAnalyser module", DeprecationWarning)
        d = self.stat1.shape[1] // self.stat0.shape[1]
        C = self.stat0.shape[1]
    
        for c in range(C):
            distrib_idx = range(c * d, (c+1) * d)
            phi[distrib_idx, :] = scipy.linalg.solve(_A[c], _C[:, distrib_idx]).T

        # Update the residual covariance if needed 
        # (only for full co-variance case of PLDA
        sigma = None
        if sigma_obs is not None:
            sigma = sigma_obs - phi.dot(_C) / session_number

        # MINIMUM DIVERGENCE STEP
        if _R is not None:
            print('applyminDiv reestimation')
            ch = scipy.linalg.cholesky(_R)
            phi = phi.dot(ch)

        return phi, sigma

    def estimate_between_class(self,
                               itNb,
                               V,
                               mean,
                               sigma_obs,
                               batch_size=100,
                               Ux=None,
                               Dz=None,
                               minDiv=True,
                               num_thread=1,
                               re_estimate_residual=False,
                               save_partial=False):
        """
        Estimate the factor loading matrix for the between class covariance

        :param itNb:
        :param V: initial between class covariance matrix
        :param mean: global mean vector
        :param sigma_obs: covariance matrix of the input data
        :param batch_size: size of the batches to process one by one to reduce the memory usage
        :param Ux: statserver of supervectors
        :param Dz: statserver of supervectors
        :param minDiv: boolean, if True run the minimum divergence step after maximization
        :param num_thread: number of parallel process to run
        :param re_estimate_residual: boolean, if True the residual covariance matrix is re-estimated (for PLDA)
        :param save_partial: boolean, if True, save FA model for each iteration
        :return: the within class factor loading matrix
        """
        warnings.warn("deprecated, use FactorAnalyser module", DeprecationWarning)
        # Initialize the covariance
        sigma = sigma_obs

        # Estimate F by iterating the EM algorithm
        for it in range(itNb):
            logging.info('Estimate between class covariance, it %d / %d',
                         it + 1, itNb)

            # Dans la fonction estimate_between_class
            model_shifted_stat = copy.deepcopy(self)
        
            # subtract channel effect, Ux, if already estimated 
            if Ux is not None:
                model_shifted_stat = model_shifted_stat.subtract_weighted_stat1(Ux)
            
            # Sum statistics per speaker
            model_shifted_stat, session_per_model = model_shifted_stat.sum_stat_per_model()
            # subtract residual, Dz, if already estimated
            if Dz is not None:
                model_shifted_stat = model_shifted_stat.subtract(Dz)                     
                    
            # E-step
            print("E_step")
            _A, _C, _R = model_shifted_stat._expectation(V, mean, sigma, session_per_model, batch_size, num_thread)
        
            if not minDiv:
                _R = None
            
            # M-step
            print("M_step")
            if re_estimate_residual:
                V, sigma = model_shifted_stat._maximization(V, _A, _C, _R, sigma_obs, session_per_model.sum())
            else:
                V = model_shifted_stat._maximization(V, _A, _C, _R)[0]

            if sigma.ndim == 2:
                logging.info('Likelihood after iteration %d / %f', it + 1, compute_llk(self, V, sigma))
            
            del model_shifted_stat

            if save_partial:
                sidekit.sidekit_io.write_fa_hdf5((mean, V, None, None, sigma),
                                                 save_partial + "_{}_between_class.h5".format(it))

        return V, sigma

    def estimate_within_class(self,
                              it_nb,
                              U,
                              mean,
                              sigma_obs,
                              batch_size=100,
                              Vy=None,
                              Dz=None,
                              min_div=True,
                              num_thread=1,
                              save_partial=False):
        """
        Estimate the factor loading matrix for the within class covariance

        :param it_nb: number of iterations to estimate the within class covariance matrix
        :param U: initial within class covariance matrix
        :param mean: mean of the input data
        :param sigma_obs: co-variance matrix of the input data
        :param batch_size: number of sessions to process per batch to optimize memory usage
        :param Vy: statserver of supervectors
        :param Dz: statserver of supervectors
        :param min_div: boolean, if True run the minimum divergence step after maximization
        :param num_thread: number of parallel process to run
        :param save_partial: boolean, if True, save FA model for each iteration
        :return: the within class factor loading matrix
        """
        warnings.warn("deprecated, use FactorAnalyser module", DeprecationWarning)
        session_shifted_stat = copy.deepcopy(self)
        
        session_per_model = numpy.ones(session_shifted_stat.modelset.shape[0])
        # Estimate F by iterating the EM algorithm
        for it in range(it_nb):
            logging.info('Estimate between class covariance, it %d / %d',
                         it + 1, it_nb)

            session_shifted_stat = self
            # subtract channel effect, Ux,  if already estimated 
            # and sum per speaker
            if Vy is not None:
                session_shifted_stat = session_shifted_stat.subtract_weighted_stat1(Vy)
                # session_shifted_stat = self.subtract_weighted_stat1(Vy)

            # subtract residual, Dz, if already estimated
            if Dz is not None:
                session_shifted_stat = session_shifted_stat.subtract_weighted_stat1(Dz)
        
            # E step
            A, C, R = session_shifted_stat._expectation(U, mean, sigma_obs,
                                                        session_per_model,
                                                        batch_size, num_thread)

            # M step
            if not min_div:
                R = None
            U = session_shifted_stat._maximization(U, A, C, R)[0]

            if save_partial:
                sidekit.sidekit_io.write_fa_hdf5((None, None, U, None, None), save_partial + "_{}_within_class.h5")

        return U
        
    def estimate_map(self, itNb, D, mean, Sigma, Vy=None, Ux=None, num_thread=1, save_partial=False):
        """
        
        :param itNb: number of iterations to estimate the MAP covariance matrix
        :param D: Maximum a Posteriori marix to estimate
        :param mean: mean of the input parameters
        :param Sigma: residual covariance matrix
        :param Vy: statserver of supervectors
        :param Ux: statserver of supervectors
        :param num_thread: number of parallel process to run
        :param save_partial: boolean, if True save MAP matrix after each iteration
        
        :return: the MAP covariance matrix into a vector as it is diagonal
        """
        warnings.warn("deprecated, use FactorAnalyser module", DeprecationWarning)
        model_shifted_stat = copy.deepcopy(self)
        
        logging.info('Estimate MAP matrix')
        # subtract speaker and channel if already estimated
        model_shifted_stat.center_stat1(mean)
        if Vy is not None:
            model_shifted_stat = model_shifted_stat.subtract_weighted_stat1(Vy)
        if Ux is not None:
            model_shifted_stat = model_shifted_stat.subtract_weighted_stat1(Ux)

        # Sum statistics per speaker
        model_shifted_stat = model_shifted_stat.sum_stat_per_model()[0]

        d = model_shifted_stat.stat1.shape[1] / model_shifted_stat.stat0.shape[1]
        C = model_shifted_stat.stat0.shape[1]

        # Replicate self.stat0
        index_map = numpy.repeat(numpy.arange(C), d)
        _stat0 = model_shifted_stat.stat0[:, index_map]
        
        # Estimate D by iterating the EM algorithm
        for it in range(itNb):
            logging.info('Estimate MAP covariance, it %d / %d', it + 1, itNb)

            # E step
            e_h = numpy.zeros(model_shifted_stat.stat1.shape, dtype=STAT_TYPE)
            _A = numpy.zeros(D.shape, dtype=STAT_TYPE)
            _C = numpy.zeros(D.shape, dtype=STAT_TYPE)
            for idx in range(model_shifted_stat.modelset.shape[0]):
                Lambda = numpy.ones(D.shape) + (_stat0[idx, :] * D**2 / Sigma)
                e_h[idx] = model_shifted_stat.stat1[idx] * D / (Lambda * Sigma)
                _A = _A + (1/Lambda + e_h[idx]**2) * _stat0[idx, :]
                _C = _C + e_h[idx] * model_shifted_stat.stat1[idx]

            # M step
            D = _C / _A

            if save_partial:
                sidekit.sidekit_io.write_fa_hdf5((None, None, None, D, None), save_partial + "_{}_map.h5")
            
        return D
               
    def estimate_hidden(self, mean, sigma, V=None, U=None, D=None, batch_size=100, num_thread=1):
        """
        Assume that the statistics have not been whitened
        :param mean: global mean of the data to subtract
        :param sigma: residual covariance matrix of the Factor Analysis model
        :param V: between class covariance matrix
        :param U: within class covariance matrix
        :param D: MAP covariance matrix
        :param batch_size: size of the batches used to reduce memory footprint
        :param num_thread: number of parallel process to run
        """
        warnings.warn("deprecated, use FactorAnalyser module", DeprecationWarning)
        if V is None:
            V = numpy.zeros((self.stat1.shape[1], 0), dtype=STAT_TYPE)
        if U is None:
            U = numpy.zeros((self.stat1.shape[1], 0), dtype=STAT_TYPE)
        W = numpy.hstack((V, U))
        
        # Estimate yx    
        r = W.shape[1]
        d = int(self.stat1.shape[1] / self.stat0.shape[1])
        C = self.stat0.shape[1]

        self.whiten_stat1(mean, sigma)
        W_white = copy.deepcopy(W)
        if sigma.ndim == 2:
            eigenvalues, eigenvectors = scipy.linalg.eigh(sigma)
            ind = eigenvalues.real.argsort()[::-1]
            eigenvalues = eigenvalues.real[ind]
            eigenvectors = eigenvectors.real[:, ind]
            sqr_inv_eval_sigma = 1 / numpy.sqrt(eigenvalues.real)
            sqr_inv_sigma = numpy.dot(eigenvectors, numpy.diag(sqr_inv_eval_sigma))
            W_white = sqr_inv_sigma.T.dot(W)
        elif sigma.ndim == 1:
            sqr_inv_sigma = 1/numpy.sqrt(sigma)
            W_white = W * sqr_inv_sigma[:, None]

        # Replicate self.stat0
        index_map = numpy.repeat(numpy.arange(C), d)
        _stat0 = self.stat0[:, index_map]

        y = sidekit.StatServer()
        y.modelset = copy.deepcopy(self.modelset)
        y.segset = copy.deepcopy(self.segset)
        y.start = copy.deepcopy(self.start)
        y.stop = copy.deepcopy(self.stop)
        y.stat0 = numpy.ones((self.modelset.shape[0], 1))
        y.stat1 = numpy.ones((self.modelset.shape[0], V.shape[1]))

        x = sidekit.StatServer()
        x.modelset = copy.deepcopy(self.modelset)
        x.segset = copy.deepcopy(self.segset)
        x.start = copy.deepcopy(self.start)
        x.stop = copy.deepcopy(self.stop)
        x.stat0 = numpy.ones((self.modelset.shape[0], 1))
        x.stat1 = numpy.ones((self.modelset.shape[0], U.shape[1]))

        z = sidekit.StatServer()
        if D is not None:
            z.modelset = copy.deepcopy(self.modelset)
            z.segset = copy.deepcopy(self.segset)
            z.stat0 = numpy.ones((self.modelset.shape[0], 1), dtype=STAT_TYPE)
            z.stat1 = numpy.ones((self.modelset.shape[0], D.shape[0]), dtype=STAT_TYPE)

            VUyx = copy.deepcopy(self)

        # Process in batches in order to reduce the memory requirement
        batch_nb = int(numpy.floor(self.segset.shape[0]/float(batch_size) + 0.999))

        for batch in range(batch_nb):
            batch_start = batch * batch_size
            batch_stop = min((batch + 1) * batch_size, self.segset.shape[0])
            batch_len = batch_stop - batch_start

            # Allocate the memory to save time
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                e_h = numpy.zeros((batch_len, r), dtype=STAT_TYPE)
                tmp_e_h = multiprocessing.Array(ct, e_h.size)
                e_h = numpy.ctypeslib.as_array(tmp_e_h.get_obj())
                e_h = e_h.reshape(batch_len, r)

                e_hh = numpy.zeros((batch_len, r, r), dtype=STAT_TYPE)
                tmp_e_hh = multiprocessing.Array(ct, e_hh.size)
                e_hh = numpy.ctypeslib.as_array(tmp_e_hh.get_obj())
                e_hh = e_hh.reshape(batch_len, r, r)

            # Parallelized loop on the model id's
            fa_model_loop(batch_start=batch_start, mini_batch_indices=numpy.arange(batch_len),
                          r=r, phi_white=W_white, phi=W, sigma=sigma,
                          stat0=_stat0, stat1=self.stat1,
                          e_h=e_h, e_hh=e_hh, num_thread=num_thread)

            y.stat1[batch_start:batch_start + batch_len, :] = e_h[:, :V.shape[1]]
            x.stat1[batch_start:batch_start + batch_len, :] = e_h[:, V.shape[1]:]

            if D is not None:
                # subtract Vy + Ux from the first-order statistics
                VUyx.stat1[batch_start:batch_start + batch_len, :] = e_h.dot(W.T)

        if D is not None:
            # subtract Vy + Ux from the first-order statistics
            self = self.subtract_weighted_stat1(VUyx)

            # estimate z
            for idx in range(self.modelset.shape[0]):
                Lambda = numpy.ones(D.shape, dtype=STAT_TYPE) + (_stat0[idx, :] * D**2)
                z.stat1[idx] = self.stat1[idx] * D / Lambda
         
        return y, x, z

    def factor_analysis(self, rank_f, rank_g=0, rank_h=None, re_estimate_residual=False,
                        it_nb=(10, 10, 10), min_div=True, ubm=None,
                        batch_size=100, num_thread=1, save_partial=False, init_matrices=(None, None, None)):
        """        
        :param rank_f: rank of the between class variability matrix
        :param rank_g: rank of the within  class variab1ility matrix
        :param rank_h: boolean, if True, estimate the residual covariance
            matrix. Default is False
        :param re_estimate_residual: boolean, if True, the residual covariance matrix is re-estimated (use for PLDA)
        :param it_nb: tupple of three integers; number of iterations to run
            for F, G, H estimation
        :param min_div: boolean, if True, re-estimate the covariance matrices
            according to the minimum divergence criteria
        :param batch_size: number of sessions to process in one batch or memory optimization
        :param num_thread: number of thread to run in parallel
        :param ubm: origin of the space; should be None for PLDA and be a 
            Mixture object for JFA or TV
        :param save_partial: name of the file to save intermediate models,
               if True, save before each split of the distributions
        :param init_matrices: tuple of three optional matrices to initialize the model, default is (None, None, None)

        :return: three matrices, the between class factor loading matrix,
            the within class factor loading matrix the diagonal MAP matrix 
            (as a vector) and the residual covariance matrix
        """
        warnings.warn("deprecated, use FactorAnalyser module", DeprecationWarning)

        (F_init, G_init, H_init) = init_matrices
        """ not true anymore, stats are not whiten"""
        # Whiten the statistics around the UBM.mean or, 
        # if there is no UBM, around the effective mean

        vect_size = self.stat1.shape[1]
        if ubm is None:
            mean = self.stat1.mean(axis=0)
            Sigma_obs = self.get_total_covariance_stat1()
            if F_init is None:
                evals, evecs = scipy.linalg.eigh(Sigma_obs)
                idx = numpy.argsort(evals)[::-1]
                evecs = evecs[:, idx]
                F_init = evecs[:, :rank_f]
        else:
            mean = ubm.get_mean_super_vector()
            Sigma_obs = 1. / ubm.get_invcov_super_vector()
            if F_init is None:
                F_init = numpy.random.randn(vect_size, rank_f).astype(dtype=STAT_TYPE)

        if G_init is None:
            G_init = numpy.random.randn(vect_size, rank_g)
        # rank_H = 0
        if rank_h is not None:  # H is empty or full-rank
            rank_h = vect_size
        else:
            rank_h = 0
        if H_init is None:
            H_init = numpy.random.randn(rank_h).astype(dtype=STAT_TYPE) * Sigma_obs.mean()

        # Estimate the between class variability matrix
        if rank_f == 0 or it_nb[0] == 0:
            F = F_init
            sigma = Sigma_obs
        else:
            # Modify the StatServer for the Total Variability estimation
            # each session is considered a class.
            if rank_g == rank_h == 0 and not re_estimate_residual:
                modelset_backup = copy.deepcopy(self.modelset)
                self.modelset = self.segset            
            
            F, sigma = self.estimate_between_class(it_nb[0],
                                                   F_init,
                                                   mean,
                                                   Sigma_obs,
                                                   batch_size,
                                                   None,
                                                   None,
                                                   min_div,
                                                   num_thread,
                                                   re_estimate_residual,
                                                   save_partial)

            if rank_g == rank_h == 0 and not re_estimate_residual:
                            self.modelset = modelset_backup

        # Estimate the within class variability matrix
        if rank_g == 0 or it_nb[2] == 0:
            G = G_init
        else:
            # Estimate Vy per model (not per session)
            Gtmp = numpy.random.randn(vect_size, 0)
            model_shifted_stat = self.sum_stat_per_model()[0]
            y, x, z = model_shifted_stat.estimate_hidden(mean, Sigma_obs,
                                                         F, Gtmp, None,
                                                         num_thread)
                        
            """ Here we compute Vy for each  session so we duplicate first 
            the Y computed per model for each session corresponding to 
            this model and then multiply by V.
            We subtract then a weighted version of Vy from the statistics."""
            duplicate_y = numpy.zeros((self.modelset.shape[0], rank_f), dtype=STAT_TYPE)
            for idx, mod in enumerate(y.modelset):
                duplicate_y[self.modelset == mod] = y.stat1[idx]
            Vy = copy.deepcopy(self)
            Vy.stat1 = duplicate_y.dot(F.T)

            # Estimate G
            G = self.estimate_within_class(it_nb[1],
                                           G_init,
                                           mean,
                                           Sigma_obs,
                                           batch_size,
                                           Vy,
                                           None,
                                           min_div,
                                           num_thread,
                                           save_partial)

        # Estimate the MAP covariance matrix
        if rank_h == 0 or it_nb[2] == 0:
            H = H_init
        else:
            # Estimate Vy per model (not per session)
            empty = numpy.random.randn(vect_size, 0)
            tmp_stat = self.sum_stat_per_model()[0]
            y, x, z = tmp_stat.estimate_hidden(mean, Sigma_obs, F, empty, None, num_thread)
                        
            """ Here we compute Vy for each  session so we duplicate first 
            the Y computed per model for each session corresponding to 
            this model and then multiply by V.
            We subtract then a weighted version of Vy from the statistics."""
            duplicate_y = numpy.zeros((self.modelset.shape[0], rank_f), dtype=STAT_TYPE)
            for idx, mod in enumerate(y.modelset):
                duplicate_y[self.modelset == mod] = y.stat1[idx]
            Vy = copy.deepcopy(self)
            Vy.stat1 = duplicate_y.dot(F.T)
            
            # Estimate Ux per session
            tmp_stat = copy.deepcopy(self)
            tmp_stat = tmp_stat.subtract_weighted_stat1(Vy)
            y, x, z = tmp_stat.estimate_hidden(mean, Sigma_obs, empty, G, None, num_thread)
            
            Ux = copy.deepcopy(self)
            Ux.stat1 = x.stat1.dot(G.T)
            
            # Estimate H
            H = self.estimate_map(it_nb[2], H_init,
                                  mean,
                                  Sigma_obs,
                                  Vy,
                                  Ux,
                                  save_partial)

        return mean, F, G, H, sigma

    # @staticmethod
    # def read_subset(statserver_filename, idmap, prefix=''):
    #     """
    #     Given a statserver in HDF5 format stored on disk and an IdMap,
    #     create a StatServer object filled with sessions corresponding to the IdMap.
    #
    #     :param statserver_filename: name of the statserver in hdf5 format to read from
    #     :param idmap: the IdMap of sessions to load
    #     :param prefix: prefix of the group in HDF5 file
    #     :return: a StatServer
    #     """
    #     with h5py.File(statserver_filename, 'r') as h5f:
    #
    #         # create tuples of (model,seg) for both HDF5 and IdMap for quick comparaison
    #         sst = [(mod, seg) for mod, seg in zip(h5f[prefix+"modelset"].value.astype('U', copy=False),
    #                                               h5f[prefix+"segset"].value.astype('U', copy=False))]
    #         imt = [(mod, seg) for mod, seg in zip(idmap.leftids, idmap.rightids)]
    #
    #         # Get indices of existing sessions
    #         existing_sessions = set(sst).intersection(set(imt))
    #         idx = numpy.sort(numpy.array([sst.index(session) for session in existing_sessions]))
    #
    #         # Create the new StatServer by loading the correct sessions
    #         statserver = sidekit.StatServer()
    #         statserver.modelset = h5f[prefix+"modelset"].value[idx].astype('U', copy=False)
    #         statserver.segset = h5f[prefix+"segset"].value[idx].astype('U', copy=False)
    #
    #         tmpstart = h5f.get(prefix+"start").value[idx]
    #         tmpstop = h5f.get(prefix+"stop").value[idx]
    #         statserver.start = numpy.empty(idx.shape, '|O')
    #         statserver.stop = numpy.empty(idx.shape, '|O')
    #         statserver.start[tmpstart != -1] = tmpstart[tmpstart != -1]
    #         statserver.stop[tmpstop != -1] = tmpstop[tmpstop != -1]
    #
    #         statserver.stat0 = h5f[prefix+"stat0"].value[idx, :]
    #         statserver.stat1 = h5f[prefix+"stat1"].value[idx, :]
    #
    #         return statserver

    def generator(self):
        """
        Create a generator which yield stat0, stat1, of one session at a time
        """
        i = 0
        while i < self.stat0.shape[0]:
            yield self.stat0[i, :], self.stat1[i, :]
            i += 1

    @staticmethod
    def read_subset(statserver_filename, index, prefix=''):
        """
        Given a statserver in HDF5 format stored on disk and an IdMap,
        create a StatServer object filled with sessions corresponding to the IdMap.

        :param statserver_filename: name of the statserver in hdf5 format to read from
        :param index: the IdMap of sessions to load or an array of index to load
        :param prefix: prefix of the group in HDF5 file
        :return: a StatServer
        """
        with h5py.File(statserver_filename, 'r') as h5f:

            if isinstance(index, sidekit.IdMap):
                # create tuples of (model,seg) for both HDF5 and IdMap for quick comparaison
                sst = [(mod, seg) for mod, seg in zip(h5f[prefix+"modelset"].value.astype('U', copy=False),
                                                      h5f[prefix+"segset"].value.astype('U', copy=False))]
                imt = [(mod, seg) for mod, seg in zip(index.leftids, index.rightids)]

                # Get indices of existing sessions
                existing_sessions = set(sst).intersection(set(imt))
                idx = numpy.sort(numpy.array([sst.index(session) for session in existing_sessions]).astype(int))

            else:
                idx = numpy.array(index)
                # If some indices are higher than the size of the StatServer, they are replace by the last index
                idx = numpy.array([min(len(h5f[prefix+"modelset"]) - 1, idx[ii]) for ii in range(len(idx))])

            # Create the new StatServer by loading the correct sessions
            statserver = sidekit.StatServer()
            statserver.modelset = h5f[prefix+"modelset"].value[idx].astype('U', copy=False)
            statserver.segset = h5f[prefix+"segset"].value[idx].astype('U', copy=False)

            tmpstart = h5f.get(prefix+"start").value[idx]
            tmpstop = h5f.get(prefix+"stop").value[idx]
            statserver.start = numpy.empty(idx.shape, '|O')
            statserver.stop = numpy.empty(idx.shape, '|O')
            statserver.start[tmpstart != -1] = tmpstart[tmpstart != -1]
            statserver.stop[tmpstop != -1] = tmpstop[tmpstop != -1]

            statserver.stat0 = h5f[prefix+"stat0"].value[idx, :]
            statserver.stat1 = h5f[prefix+"stat1"].value[idx, :]

            return statserver


