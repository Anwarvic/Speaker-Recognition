# -* coding: utf-8 -*-
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
Copyright 2014-2019 Sylvain Meignier and Anthony Larcher

    :mod:`factor_analyser` provides methods to train different types of factor analysers

"""
import copy
import numpy
import multiprocessing
import logging
import h5py
import scipy
import warnings
import ctypes
from sidekit.sv_utils import serialize
from sidekit.statserver import StatServer
from sidekit.mixture import Mixture
from sidekit.sidekit_wrappers import process_parallel_lists, deprecated, check_path_existance
from sidekit import STAT_TYPE

__license__ = "LGPL"
__author__ = "Anthony Larcher & Sylvain Meignier"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'

def e_on_batch(stat0, stat1, ubm, F):
    """
    Compute statistics for the Expectation step on a batch of data

    :param stat0: matrix of zero-order statistics (1 session per line)
    :param stat1: matrix of first-order statistics (1 session per line)
    :param ubm: Mixture object
    :param F: factor loading matrix
    :return: first and second order statistics
    """
    tv_rank = F.shape[1]
    nb_distrib = stat0.shape[1]
    feature_size = ubm.mu.shape[1]
    index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)
    upper_triangle_indices = numpy.triu_indices(tv_rank)

    gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

    # Allocate the memory to save
    session_nb = stat0.shape[0]
    e_h = numpy.zeros((session_nb, tv_rank), dtype=STAT_TYPE)
    e_hh = numpy.zeros((session_nb, tv_rank * (tv_rank + 1) // 2), dtype=STAT_TYPE)

    # Whiten the statistics for diagonal or full models
    stat1 -= stat0[:, index_map] * ubm.get_mean_super_vector()
 
    if gmm_covariance == "diag":
        stat1 *= numpy.sqrt(ubm.get_invcov_super_vector())
    elif gmm_covariance == "full":
        stat1 = numpy.einsum("ikj,ikl->ilj",
                             stat1.T.reshape(-1, nb_distrib, session_nb),
                             ubm.invchol
                             ).reshape(-1, session_nb).T

    for idx in range(session_nb):
        inv_lambda = scipy.linalg.inv(numpy.eye(tv_rank) + (F.T * stat0[idx, index_map]).dot(F))
        aux = F.T.dot(stat1[idx, :])
        e_h[idx] = numpy.dot(aux, inv_lambda)
        e_hh[idx] = (inv_lambda + numpy.outer(e_h[idx], e_h[idx]))[upper_triangle_indices]

    return e_h, e_hh


def e_worker(arg, q):
    """
    Encapsulates the method that compute statistics for expectation step

    :param arg: a tuple that should include
        a matrix of zero-order statistics (1 session per line)
        a matrix of first-order statistics (1 session per line)
        a Mixture object
        a factor loading matrix
    :param q: output queue (a multiprocessing.Queue object)
    """
    q.put(arg[:2] + e_on_batch(*arg))


def e_gather(arg, q):
    """
    Consumer that sums accumulators stored in the memory

    :param arg: a tuple of input parameters including three accumulators for the estimation of Factor Analysis matrix
    :param q: input queue that is filled by the producers and emptied in this function (a multiprocessing.Queue object)
    :return: the three accumulators
    """
    _A, _C, _R = arg

    while True:

        stat0, stat1, e_h, e_hh = q.get()
        if e_h is None:
            break
        _A += stat0.T.dot(e_hh)
        _C += e_h.T.dot(stat1)
        _R += numpy.sum(e_hh, axis=0)

    return _A, _C, _R


def iv_extract_on_batch(arg, q):
    """
    Extract i-vectors for a batch of sessions (shows)

    :param arg: a tuple of inputs that includes a list of batch_indices, a matrix of zero-order statistics
        a matrix of first order statistics, a Mixture model and loading factor matrix
    :param q: the output queue to fill (a multiprocessing.Queue object)
    """
    batch_indices, stat0, stat1, ubm, F = arg
    E_h, E_hh = e_on_batch(stat0, stat1, ubm, F)
    tv_rank = E_h.shape[1]
    q.put((batch_indices,) + (E_h, E_hh[:, numpy.array([i * tv_rank-((i*(i-1))//2) for i in range(tv_rank)])]))


def iv_collect(arg, q):
    """
    Consumer method that takes inputs from a queue and fill matrices with i-vectors
    and uncertainty matrices (diagonal version only)

    :param arg:a tuple of inputs including a matrix to store i-vectors and a matrix to store uncertainty matrices
    :param q: the input queue (a multiprocessing.Queue object)
    :return: the matrices of i-vectors and uncertainty matrices
    """
    iv, iv_sigma = arg

    while True:

        batch_idx, e_h, e_hh = q.get()
        if e_h is None:
            break
        iv[batch_idx, :] = e_h
        iv_sigma[batch_idx, :] = e_hh

    return iv, iv_sigma


@process_parallel_lists
def fa_model_loop(batch_start,
                  mini_batch_indices,
                  factor_analyser,
                  stat0,
                  stat1,
                  e_h,
                  e_hh,
                  num_thread=1):
    """
    Methods that is called for PLDA estimation for parallelization on classes

    :param batch_start: index to start at in the list
    :param mini_batch_indices: indices of the elements in the list (should start at zero)
    :param factor_analyser: FactorAnalyser object
    :param stat0: matrix of zero order statistics
    :param stat1: matrix of first order statistics
    :param e_h: accumulator
    :param e_hh: accumulator
    :param num_thread: number of parallel process to run
    """
    rank = factor_analyser.F.shape[1]
    if factor_analyser.Sigma.ndim == 2:
        A = factor_analyser.F.T.dot(factor_analyser.F)
        inv_lambda_unique = dict()
        for sess in numpy.unique(stat0[:, 0]):
            inv_lambda_unique[sess] = scipy.linalg.inv(sess * A + numpy.eye(A.shape[0]))

    tmp = numpy.zeros((factor_analyser.F.shape[1], factor_analyser.F.shape[1]), dtype=STAT_TYPE)

    for idx in mini_batch_indices:
        if factor_analyser.Sigma.ndim == 1:
            inv_lambda = scipy.linalg.inv(numpy.eye(rank) +
                                          (factor_analyser.F.T * stat0[idx + batch_start, :]).dot(factor_analyser.F))
        else:
            inv_lambda = inv_lambda_unique[stat0[idx + batch_start, 0]]

        aux = factor_analyser.F.T.dot(stat1[idx + batch_start, :])
        numpy.dot(aux, inv_lambda, out=e_h[idx])
        e_hh[idx] = inv_lambda + numpy.outer(e_h[idx], e_h[idx], tmp)


class FactorAnalyser:
    """
    A class to train factor analyser such as total variability models and Probabilistic
    Linear Discriminant Analysis (PLDA).

    :attr mean: mean vector
    :attr F: between class matrix
    :attr G: within class matrix
    :attr H: MAP covariance matrix (for Joint Factor Analysis only)
    :attr Sigma: residual covariance matrix
    """

    def __init__(self,
                 input_file_name=None,
                 mean=None,
                 F=None,
                 G=None,
                 H=None,
                 Sigma=None):
        """
        Initialize a Factor Analyser object to None or by reading from an HDF5 file.
        When loading from a file, other parameters can be provided to overwrite each of the component.

        :param input_file_name: name of the HDF5 file to read from, default is nNone
        :param mean: the mean vector
        :param F: between class matrix
        :param G: within class matrix
        :param H: MAP covariance matrix
        :param Sigma: residual covariance matrix
        """
        if input_file_name is not None:
            fa = FactorAnalyser.read(input_file_name)
            self.mean = fa.mean
            self.F = fa.F
            self.G = fa.G
            self.H = fa.H
            self.Sigma = fa.Sigma
        else:
            self.mean = None
            self.F = None
            self.G = None
            self.H = None
            self.Sigma = None

        if mean is not None:
            self.mean = mean
        if F is not None:
            self.F = F
        if G is not None:
            self.G = G
        if H is not None:
            self.H = H
        if Sigma is not None:
            self.Sigma = Sigma

    @check_path_existance
    def write(self, output_file_name):
        """
        Write a FactorAnalyser object into HDF5 file

        :param output_file_name: the name of the file to write to
        """
        with h5py.File(output_file_name, "w") as fh:
            kind = numpy.zeros(5, dtype="int16")  # FA with 5 matrix
            if self.mean is not None:
                kind[0] = 1
                fh.create_dataset("fa/mean", data=self.mean,
                                  compression="gzip",
                                  fletcher32=True)
            if self.F is not None:
                kind[1] = 1
                fh.create_dataset("fa/f", data=self.F,
                                  compression="gzip",
                                  fletcher32=True)
            if self.G is not None:
                kind[2] = 1
                fh.create_dataset("fa/g", data=self.G,
                                  compression="gzip",
                                  fletcher32=True)
            if self.H is not None:
                kind[3] = 1
                fh.create_dataset("fa/h", data=self.H,
                                  compression="gzip",
                                  fletcher32=True)
            if self.Sigma is not None:
                kind[4] = 1
                fh.create_dataset("fa/sigma", data=self.Sigma,
                                  compression="gzip",
                                  fletcher32=True)
            fh.create_dataset("fa/kind", data=kind,
                              compression="gzip",
                              fletcher32=True)

    @staticmethod
    def read(input_filename):
        """
         Read a generic FactorAnalyser model from a HDF5 file

        :param input_filename: the name of the file to read from

        :return: a FactorAnalyser object
        """
        fa = FactorAnalyser()
        with h5py.File(input_filename, "r") as fh:
            kind = fh.get("fa/kind").value
            if kind[0] != 0:
                fa.mean = fh.get("fa/mean").value
            if kind[1] != 0:
                fa.F = fh.get("fa/f").value
            if kind[2] != 0:
                fa.G = fh.get("fa/g").value
            if kind[3] != 0:
                fa.H = fh.get("fa/h").value
            if kind[4] != 0:
                fa.Sigma = fh.get("fa/sigma").value
        return fa

    def total_variability_raw(self,
                              stat_server,
                              ubm,
                              tv_rank,
                              nb_iter=20,
                              min_div=True,
                              tv_init=None,
                              save_init=False,
                              output_file_name=None):
        """
        Train a total variability model using a single process on a single node.
        This method is provided for didactic purpose and should not be used as it uses 
        to much memory and is to slow. If you want to use a single process
        run: "total_variability_single"

        :param stat_server: the StatServer containing data to train the model
        :param ubm: a Mixture object
        :param tv_rank: rank of the total variability model
        :param nb_iter: number of EM iteration
        :param min_div: boolean, if True, apply minimum divergence re-estimation
        :param tv_init: initial matrix to start the EM iterations with
        :param save_init: boolean, if True, save the initial matrix
        :param output_file_name: name of the file where to save the matrix
        """
        assert(isinstance(stat_server, StatServer) and stat_server.validate()), \
            "First argument must be a proper StatServer"
        assert(isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"
        assert(isinstance(tv_rank, int) and (0 < tv_rank <= min(stat_server.stat1.shape))), \
            "tv_rank must be a positive integer less than the dimension of the statistics"
        assert(isinstance(nb_iter, int) and (0 < nb_iter)), "nb_iter must be a positive integer"
 
        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full" 

        # Set useful variables
        nb_sessions, sv_size = stat_server.stat1.shape
        feature_size = ubm.mu.shape[1]
        nb_distrib = ubm.w.shape[0]    

        # Whiten the statistics for diagonal or full models
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # Initialize TV from given data or randomly
        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.F = numpy.random.randn(sv_size, tv_rank) if tv_init is None else tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # Save init if required
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        if save_init:
            self.write(output_file_name + "_init.h5")

        # Estimate  TV iteratively
        for it in range(nb_iter):
            # Create accumulators for the list of models to process
            _A = numpy.zeros((nb_distrib, tv_rank, tv_rank), dtype=STAT_TYPE)
            _C = numpy.zeros((tv_rank, feature_size * nb_distrib), dtype=STAT_TYPE)
        
            _R = numpy.zeros((tv_rank, tv_rank), dtype=STAT_TYPE)

            # E-step:
            index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)

            for sess in range(stat_server.segset.shape[0]):

                inv_lambda = scipy.linalg.inv(numpy.eye(tv_rank)
                                              + (self.F.T * stat_server.stat0[sess, index_map]).dot(self.F))
                Aux = self.F.T.dot(stat_server.stat1[sess, :])
                e_h = Aux.dot(inv_lambda)
                e_hh = inv_lambda + numpy.outer(e_h, e_h)
                
                # Accumulate for minimum divergence step
                _R += e_hh

                # Accumulate for M-step
                _C += numpy.outer(e_h, stat_server.stat1[sess, :])
                _A += e_hh * stat_server.stat0[sess][:, numpy.newaxis, numpy.newaxis]

            _R /= nb_sessions

            # M-step ( + MinDiv si _R n'est pas None)
            for g in range(nb_distrib):
                distrib_idx = range(g * feature_size, (g + 1) * feature_size)
                self.F[distrib_idx, :] = scipy.linalg.solve(_A[g], _C[:, distrib_idx]).T

            # MINIMUM DIVERGENCE STEP
            if min_div:
                ch = scipy.linalg.cholesky(_R)
                self.F = self.F.dot(ch)

            # Save the FactorAnalyser
            if it < nb_iter - 1:
                self.write(output_file_name + "_it-{}.h5".format(it))
            else:
                self.write(output_file_name + ".h5")

    def total_variability_single(self,
                                 stat_server_filename,
                                 ubm,
                                 tv_rank,
                                 nb_iter=20,
                                 min_div=True,
                                 tv_init=None,
                                 batch_size=300,
                                 save_init=False,
                                 output_file_name=None):
        """
        Train a total variability model using a single process on a single node.
        Use this method to run a single process on a single node with optimized code.

        Optimization:
            Only half of symmetric matrices are stored here
            process sessions per batch in order to control the memory footprint

        :param stat_server_filename: the name of the file for StatServer, containing data to train the model
        :param ubm: a Mixture object
        :param tv_rank: rank of the total variability model
        :param nb_iter: number of EM iteration
        :param min_div: boolean, if True, apply minimum divergence re-estimation
        :param tv_init: initial matrix to start the EM iterations with
        :param batch_size: number of sessions to process at once to reduce memory footprint
        :param save_init: boolean, if True, save the initial matrix
        :param output_file_name: name of the file where to save the matrix
        """
        assert (isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"
        assert (isinstance(nb_iter, int) and (0 < nb_iter)), "nb_iter must be a positive integer"

        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        with h5py.File(stat_server_filename, 'r') as fh:
            nb_sessions, sv_size = fh["stat1"].shape
            feature_size = ubm.mu.shape[1]
            nb_distrib = ubm.w.shape[0]
            sv_size = nb_distrib * feature_size

        # Initialize TV from given data or randomly
        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.F = numpy.random.randn(sv_size, tv_rank) if tv_init is None else tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # Save init if required
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        if save_init:
            self.write(output_file_name + "_init.h5")

        # Create index to replicate self.stat0 and save only upper triangular coefficients of symmetric matrices
        index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)
        upper_triangle_indices = numpy.triu_indices(tv_rank)

        # Open the StatServer file
        with h5py.File(stat_server_filename, 'r') as fh:
            nb_sessions, sv_size = fh['stat1'].shape
            batch_nb = int(numpy.floor(fh['segset'].shape[0] / float(batch_size) + 0.999))
            batch_indices = numpy.array_split(numpy.arange(nb_sessions), batch_nb)

            # Estimate  TV iteratively
            for it in range(nb_iter):

                # Create accumulators for the list of models to process
                _A = numpy.zeros((nb_distrib, tv_rank * (tv_rank + 1) // 2), dtype=STAT_TYPE)
                _C = numpy.zeros((tv_rank, feature_size * nb_distrib), dtype=STAT_TYPE)
                _R = numpy.zeros((tv_rank * (tv_rank + 1) // 2), dtype=STAT_TYPE)

                # Load data per batch to reduce the memory footprint
                for batch_idx in batch_indices:

                    stat0 = fh['stat0'][batch_idx, :]
                    stat1 = fh['stat1'][batch_idx, :]

                    e_h, e_hh = e_on_batch(stat0, stat1, ubm, self.F)

                    _R += numpy.sum(e_hh, axis=0)
                    _C += e_h.T.dot(stat1)
                    _A += stat0.T.dot(e_hh)

                _R /= nb_sessions

                # M-step
                _A_tmp = numpy.zeros((tv_rank, tv_rank), dtype=STAT_TYPE)
                for c in range(nb_distrib):
                    distrib_idx = range(c * feature_size, (c + 1) * feature_size)
                    _A_tmp[upper_triangle_indices] = _A_tmp.T[upper_triangle_indices] = _A[c, :]
                    self.F[distrib_idx, :] = scipy.linalg.solve(_A_tmp, _C[:, distrib_idx]).T

                # Minimum divergence
                if min_div:
                    _R_tmp = numpy.zeros((tv_rank, tv_rank), dtype=STAT_TYPE)
                    _R_tmp[upper_triangle_indices] = _R_tmp.T[upper_triangle_indices] = _R
                    ch = scipy.linalg.cholesky(_R_tmp)
                    self.F = self.F.dot(ch)

                # Save the current FactorAnalyser
                if output_file_name is not None:
                    if it < nb_iter - 1:
                        self.write(output_file_name + "_it-{}.h5".format(it))
                    else:
                        self.write(output_file_name + ".h5")

    def total_variability(self,
                          stat_server_filename,
                          ubm,
                          tv_rank,
                          nb_iter=20,
                          min_div=True,
                          tv_init=None,
                          batch_size=300,
                          save_init=False,
                          output_file_name=None,
                          num_thread=1):
        """
        Train a total variability model using multiple process on a single node.
        this method is the recommended one to train a Total Variability matrix.

        Optimization:
            Only half of symmetric matrices are stored here
            process sessions per batch in order to control the memory footprint
            Batches are processed by a pool of workers running in different process
            The implementation is based on a multiple producers / single consumer approach

        :param stat_server_filename: a list of StatServer file names to process
        :param ubm: a Mixture object
        :param tv_rank: rank of the total variability model
        :param nb_iter: number of EM iteration
        :param min_div: boolean, if True, apply minimum divergence re-estimation
        :param tv_init: initial matrix to start the EM iterations with
        :param batch_size: size of batch to load in memory for each worker
        :param save_init: boolean, if True, save the initial matrix
        :param output_file_name: name of the file where to save the matrix
        :param num_thread: number of process to run in parallel
        """
        if not isinstance(stat_server_filename, list):
            stat_server_filename = [stat_server_filename]

        assert (isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"
        assert (isinstance(nb_iter, int) and (0 < nb_iter)), "nb_iter must be a positive integer"

        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        with h5py.File(stat_server_filename[0], 'r') as fh:  # open the first StatServer to get size
            _, sv_size = fh['stat1'].shape
            feature_size = fh['stat1'].shape[1] // fh['stat0'].shape[1]
            distrib_nb = fh['stat0'].shape[1]

        upper_triangle_indices = numpy.triu_indices(tv_rank)

        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape, dtype=STAT_TYPE)
        self.F = serialize(numpy.zeros((sv_size, tv_rank)).astype(STAT_TYPE))
        if tv_init is None:
            self.F = numpy.random.randn(sv_size, tv_rank).astype(STAT_TYPE)
        else:
            self.F = tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape, dtype=STAT_TYPE)

        # Save init if required
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        if save_init:
            self.write(output_file_name + "_init.h5")

        # Estimate  TV iteratively
        for it in range(nb_iter):

            # Create serialized accumulators for the list of models to process
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                _A = serialize(numpy.zeros((distrib_nb, tv_rank * (tv_rank + 1) // 2), dtype=STAT_TYPE))
                _C = serialize(numpy.zeros((tv_rank, sv_size), dtype=STAT_TYPE))
                _R = serialize(numpy.zeros((tv_rank * (tv_rank + 1) // 2), dtype=STAT_TYPE))

            total_session_nb = 0

            # E-step
            # Accumulate statistics for each StatServer from the list
            for stat_server_file in stat_server_filename:

                # get info from the current StatServer
                with h5py.File(stat_server_file, 'r') as fh:
                    nb_sessions = fh["modelset"].shape[0]
                    total_session_nb += nb_sessions
                    batch_nb = int(numpy.floor(nb_sessions / float(batch_size) + 0.999))
                    batch_indices = numpy.array_split(numpy.arange(nb_sessions), batch_nb)

                    manager = multiprocessing.Manager()
                    q = manager.Queue()
                    pool = multiprocessing.Pool(num_thread + 2)

                    # put Consumer to work first
                    watcher = pool.apply_async(e_gather, ((_A, _C, _R), q))
                    # fire off workers
                    jobs = []

                    # Load data per batch to reduce the memory footprint
                    for batch_idx in batch_indices:

                        # Create list of argument for a process
                        arg = fh["stat0"][batch_idx, :], fh["stat1"][batch_idx, :], ubm, self.F
                        job = pool.apply_async(e_worker, (arg, q))
                        jobs.append(job)

                    # collect results from the workers through the pool result queue
                    for job in jobs:
                        job.get()

                    # now we are done, kill the consumer
                    q.put((None, None, None, None))
                    pool.close()

                    _A, _C, _R = watcher.get()

            _R /= total_session_nb

            # M-step
            _A_tmp = numpy.zeros((tv_rank, tv_rank), dtype=STAT_TYPE)
            for c in range(distrib_nb):
                distrib_idx = range(c * feature_size, (c + 1) * feature_size)
                _A_tmp[upper_triangle_indices] = _A_tmp.T[upper_triangle_indices] = _A[c, :]
                self.F[distrib_idx, :] = scipy.linalg.solve(_A_tmp, _C[:, distrib_idx]).T

            # Minimum divergence
            if min_div:
                _R_tmp = numpy.zeros((tv_rank, tv_rank), dtype=STAT_TYPE)
                _R_tmp[upper_triangle_indices] = _R_tmp.T[upper_triangle_indices] = _R
                ch = scipy.linalg.cholesky(_R_tmp)
                self.F = self.F.dot(ch)

            # Save the current FactorAnalyser
            if output_file_name is not None:
                if it < nb_iter - 1:
                    self.write(output_file_name + "_it-{}.h5".format(it))
                else:
                    self.write(output_file_name + ".h5")

    def extract_ivectors_single(self,
                                ubm,
                                stat_server,
                                uncertainty=False):
        """
        Estimate i-vectors for a given StatServer using single process on a single node.

        :param stat_server: sufficient statistics stored in a StatServer
        :param ubm: Mixture object (the UBM)
        :param uncertainty: boolean, if True, return an additional matrix with uncertainty matrices (diagonal of the matrices)

        :return: a StatServer with i-vectors in the stat1 attribute and a matrix of uncertainty matrices (optional)
        """
        assert(isinstance(stat_server, StatServer) and stat_server.validate()), \
            "First argument must be a proper StatServer"
        assert(isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"

        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        tv_rank = self.F.shape[1]
        feature_size = ubm.mu.shape[1]
        nb_distrib = ubm.w.shape[0]

        # Whiten the statistics for diagonal or full models
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        # Extract i-vectors
        iv_stat_server = StatServer()
        iv_stat_server.modelset = copy.deepcopy(stat_server.modelset)
        iv_stat_server.segset = copy.deepcopy(stat_server.segset)
        iv_stat_server.start = copy.deepcopy(stat_server.start)
        iv_stat_server.stop = copy.deepcopy(stat_server.stop)
        iv_stat_server.stat0 = numpy.ones((stat_server.modelset.shape[0], 1))
        iv_stat_server.stat1 = numpy.ones((stat_server.modelset.shape[0], tv_rank))

        iv_sigma = numpy.ones((stat_server.modelset.shape[0], tv_rank))

        # Replicate self.stat0
        index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)

        for sess in range(stat_server.segset.shape[0]):

            inv_lambda = scipy.linalg.inv(numpy.eye(tv_rank) + (self.F.T *
                                                                stat_server.stat0[sess, index_map]).dot(self.F))
            Aux = self.F.T.dot(stat_server.stat1[sess, :])
            iv_stat_server.stat1[sess, :] = Aux.dot(inv_lambda)
            iv_sigma[sess, :] = numpy.diag(inv_lambda + numpy.outer(iv_stat_server.stat1[sess, :],
                                                                    iv_stat_server.stat1[sess, :]))

        if uncertainty:
            return iv_stat_server, iv_sigma
        else:
            return iv_stat_server

    def extract_ivectors(self,
                         ubm,
                         stat_server_filename,
                         prefix='',
                         batch_size=300,
                         uncertainty=False,
                         num_thread=1):
        """
        Parallel extraction of i-vectors using multiprocessing module

        :param ubm: Mixture object (the UBM)
        :param stat_server_filename: name of the file from which the input StatServer is read
        :param prefix: prefix used to store the StatServer in its file
        :param batch_size: number of sessions to process in a batch
        :param uncertainty: a boolean, if True, return the diagonal of the uncertainty matrices
        :param num_thread: number of process to run in parallel
        :return: a StatServer with i-vectors in the stat1 attribute and a matrix of uncertainty matrices (optional)
        """
        assert (isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"

        tv_rank = self.F.shape[1]

        # Set useful variables
        with h5py.File(stat_server_filename, 'r') as fh:  # open the first statserver to get size
            _, sv_size = fh[prefix + 'stat1'].shape
            nb_sessions = fh[prefix + "modelset"].shape[0]

            iv_server = StatServer()
            iv_server.modelset = fh.get(prefix + 'modelset').value
            iv_server.segset = fh.get(prefix + 'segset').value

            tmpstart = fh.get(prefix+"start").value
            tmpstop = fh.get(prefix+"stop").value
            iv_server.start = numpy.empty(fh[prefix+"start"].shape, '|O')
            iv_server.stop = numpy.empty(fh[prefix+"stop"].shape, '|O')
            iv_server.start[tmpstart != -1] = tmpstart[tmpstart != -1]
            iv_server.stop[tmpstop != -1] = tmpstop[tmpstop != -1]

            iv_server.stat0 = numpy.ones((nb_sessions, 1), dtype=STAT_TYPE)
            with warnings.catch_warnings():
                iv_server.stat1 = serialize(numpy.zeros((nb_sessions, tv_rank)))
                iv_sigma = serialize(numpy.zeros((nb_sessions, tv_rank)))

            nb_sessions = iv_server.modelset.shape[0]
            batch_nb = int(numpy.floor(nb_sessions / float(batch_size) + 0.999))
            batch_indices = numpy.array_split(numpy.arange(nb_sessions), batch_nb)

            manager = multiprocessing.Manager()
            q = manager.Queue()
            pool = multiprocessing.Pool(num_thread + 2)

            # put listener to work first
            watcher = pool.apply_async(iv_collect, ((iv_server.stat1, iv_sigma), q))
            # fire off workers
            jobs = []

            # Load data per batch to reduce the memory footprint
            for batch_idx in batch_indices:

                # Create list of argument for a process
                arg = batch_idx, fh["stat0"][batch_idx, :], fh["stat1"][batch_idx, :], ubm, self.F
                job = pool.apply_async(iv_extract_on_batch, (arg, q))
                jobs.append(job)

            # collect results from the workers through the pool result queue
            for job in jobs:
                job.get()

            # now we are done, kill the listener
            q.put((None, None, None))
            pool.close()
            
            iv_server.stat1, iv_sigma = watcher.get()
        if uncertainty:
            return iv_server, iv_sigma
        else:
            return iv_server

    def plda(self,
             stat_server,
             rank_f,
             nb_iter=10,
             scaling_factor=1.,
             output_file_name=None,
             save_partial=False):
        """
        Train a simplified Probabilistic Linear Discriminant Analysis model (no within class covariance matrix
        but full residual covariance matrix)

        :param stat_server: StatServer object with training statistics
        :param rank_f: rank of the between class covariance matrix
        :param nb_iter: number of iterations to run
        :param scaling_factor: scaling factor to downscale statistics (value bewteen 0 and 1)
        :param output_file_name: name of the output file where to store PLDA model
        :param save_partial: boolean, if True, save PLDA model after each iteration
        """
        vect_size = stat_server.stat1.shape[1]

        # Initialize mean and residual covariance from the training data
        self.mean = stat_server.get_mean_stat1()
        self.Sigma = stat_server.get_total_covariance_stat1()

        # Sum stat per model
        model_shifted_stat, session_per_model = stat_server.sum_stat_per_model()
        class_nb = model_shifted_stat.modelset.shape[0]

        # Multiply statistics by scaling_factor
        model_shifted_stat.stat0 *= scaling_factor
        model_shifted_stat.stat1 *= scaling_factor
        session_per_model *= scaling_factor

        # Compute Eigen Decomposition of Sigma in order to initialize the EigenVoice matrix
        sigma_obs = stat_server.get_total_covariance_stat1()
        evals, evecs = scipy.linalg.eigh(sigma_obs)
        idx = numpy.argsort(evals)[::-1]
        evecs = evecs.real[:, idx[:rank_f]]
        self.F = evecs[:, :rank_f]

        # Estimate PLDA model by iterating the EM algorithm
        for it in range(nb_iter):
            logging.info('Estimate between class covariance, it %d / %d', it + 1, nb_iter)

            # E-step
            print("E_step")

            # Copy stats as they will be whitened with a different Sigma for each iteration
            local_stat = copy.deepcopy(model_shifted_stat)

            # Whiten statistics (with the new mean and Sigma)
            local_stat.whiten_stat1(self.mean, self.Sigma)

            # Whiten the EigenVoice matrix
            eigen_values, eigen_vectors = scipy.linalg.eigh(self.Sigma)
            ind = eigen_values.real.argsort()[::-1]
            eigen_values = eigen_values.real[ind]
            eigen_vectors = eigen_vectors.real[:, ind]
            sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
            sqr_inv_sigma = numpy.dot(eigen_vectors, numpy.diag(sqr_inv_eval_sigma))
            self.F = sqr_inv_sigma.T.dot(self.F)

            # Replicate self.stat0
            index_map = numpy.zeros(vect_size, dtype=int)
            _stat0 = local_stat.stat0[:, index_map]

            e_h = numpy.zeros((class_nb, rank_f))
            e_hh = numpy.zeros((class_nb, rank_f, rank_f))

            # loop on model id's
            fa_model_loop(batch_start=0,
                          mini_batch_indices=numpy.arange(class_nb),
                          factor_analyser=self,
                          stat0=_stat0,
                          stat1=local_stat.stat1,
                          e_h=e_h,
                          e_hh=e_hh,
                          num_thread=1)

            # Accumulate for minimum divergence step
            _R = numpy.sum(e_hh, axis=0) / session_per_model.shape[0]

            _C = e_h.T.dot(local_stat.stat1).dot(scipy.linalg.inv(sqr_inv_sigma))
            _A = numpy.einsum('ijk,i->jk', e_hh, local_stat.stat0.squeeze())

            # M-step
            self.F = scipy.linalg.solve(_A, _C).T

            # Update the residual covariance
            self.Sigma = sigma_obs - self.F.dot(_C) / session_per_model.sum()

            # Minimum Divergence step
            self.F = self.F.dot(scipy.linalg.cholesky(_R))

            if output_file_name is None:
                output_file_name = "temporary_plda"

            if save_partial and it < nb_iter - 1:
                self.write(output_file_name + "_it-{}.h5".format(it))
            elif it == nb_iter - 1:
                self.write(output_file_name + ".h5")
