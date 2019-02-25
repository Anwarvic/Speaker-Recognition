
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

:mod:`mixture` provides methods to manage Gaussian mixture models

"""
import copy
import h5py
import numpy
import struct
import ctypes
import multiprocessing
import warnings
from sidekit.sidekit_wrappers import *
from sidekit.sv_utils import mean_std_many
import sys
from tqdm import tqdm

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def sum_log_probabilities(lp):
    """Sum log probabilities in a secure manner to avoid extreme values

    :param lp: numpy array of log-probabilities to sum
    """
    pp_max = numpy.max(lp, axis=1)
    log_lk = pp_max + numpy.log(numpy.sum(numpy.exp((lp.transpose() - pp_max).T), axis=1))
    ind = ~numpy.isfinite(pp_max)
    if sum(ind) != 0:
        log_lk[ind] = pp_max[ind]
    pp = numpy.exp((lp.transpose() - log_lk).transpose())
    llk = log_lk.sum()
    return pp, llk


class Mixture(object):
    """
    A class for Gaussian Mixture Model storage.
    For more details about Gaussian Mixture Models (GMM) you can refer to
    [Bimbot04]_.
    
    :attr w: array of weight parameters
    :attr mu: ndarray of mean parameters, each line is one distribution 
    :attr invcov: ndarray of inverse co-variance parameters, 2-dimensional 
        for diagonal co-variance distribution 3-dimensional for full co-variance
    :attr invchol: 3-dimensional ndarray containing upper cholesky
        decomposition of the inverse co-variance matrices
    :attr cst: array of constant computed for each distribution
    :attr det: array of determinant for each distribution
    
    """
    @staticmethod
    def read_alize(file_name):
        """

        :param file_name:
        :return:
        """
        """Read a Mixture in alize raw format

        :param mixtureFileName: name of the file to read from
        """
        mixture = Mixture()

        with open(file_name, 'rb') as f:
            distrib_nb = struct.unpack("I", f.read(4))[0]
            vect_size = struct.unpack("<I", f.read(4))[0]

            # resize all attributes
            mixture.w = numpy.zeros(distrib_nb, "d")
            mixture.invcov = numpy.zeros((distrib_nb, vect_size), "d")
            mixture.mu = numpy.zeros((distrib_nb, vect_size), "d")
            mixture.cst = numpy.zeros(distrib_nb, "d")
            mixture.det = numpy.zeros(distrib_nb, "d")

            for d in range(distrib_nb):
                mixture.w[d] = struct.unpack("<d", f.read(8))[0]
            for d in range(distrib_nb):
                mixture.cst[d] = struct.unpack("d", f.read(8))[0]
                mixture.det[d] = struct.unpack("d", f.read(8))[0]
                f.read(1)
                for c in range(vect_size):
                    mixture.invcov[d, c] = struct.unpack("d", f.read(8))[0]
                for c in range(vect_size):
                    mixture.mu[d, c] = struct.unpack("d", f.read(8))[0]
        mixture._compute_all()
        return mixture

    @staticmethod
    def read_htk(filename, begin_hmm=False, state2=False):
        """Read a Mixture in HTK format

        :param filename: name of the file to read from
        :param begin_hmm: boolean
        :param state2: boolean
        """
        mixture = Mixture()
        with open(filename, 'rb') as f:
            lines = [line.rstrip() for line in f]

        distrib = 0
        vect_size = 0
        for i in range(len(lines)):

            if lines[i] == '':
                break

            w = lines[i].split()

            if w[0] == '<NUMMIXES>':
                distrib_nb = int(w[1])
                mixture.w.resize(distrib_nb)
                mixture.cst.resize(distrib_nb)
                mixture.det.resize(distrib_nb)

            if w[0] == '<BEGINHMM>':
                begin_hmm = True

            if w[0] == '<STATE>':
                state2 = True

            if begin_hmm & state2:

                if w[0].upper() == '<MIXTURE>':
                    distrib = int(w[1]) - 1
                    mixture.w[distrib] = numpy.double(w[2])

                elif w[0].upper() == '<MEAN>':
                    if vect_size == 0:
                        vect_size = int(w[1])
                    mixture.mu.resize(distrib_nb, vect_size)
                    i += 1
                    mixture.mu[distrib, :] = numpy.double(lines[i].split())

                elif w[0].upper() == '<VARIANCE>':
                    if mixture.invcov.shape[0] == 0:
                        vect_size = int(w[1])
                    mixture.invcov.resize(distrib_nb, vect_size)
                    i += 1
                    C = numpy.double(lines[i].split())
                    mixture.invcov[distrib, :] = 1 / C

                elif w[0].upper() == '<INVCOVAR>':
                    raise Exception("we don't manage full covariance model")
                elif w[0].upper() == '<GCONST>':
                    mixture.cst[distrib] = numpy.exp(-.05 * numpy.double(w[1]))
        mixture._compute_all()
        return mixture

    def __init__(self,
                 mixture_file_name='',
                 name='empty'):
        """Initialize a Mixture from a file or as an empty Mixture.
        
        :param mixture_file_name: name of the file to read from, if empty, initialize
            an empty mixture
        """
        self.w = numpy.array([])
        self.mu = numpy.array([])
        self.invcov = numpy.array([])
        self.invchol = numpy.array([])
        self.cov_var_ctl = numpy.array([])
        self.cst = numpy.array([])
        self.det = numpy.array([])
        self.name = name
        self.A = 0

        if mixture_file_name != '':
            self.read(mixture_file_name)

    @accepts('Mixture', 'Mixture', debug=2)
    def __add__(self, other):
        """Overide the sum for a mixture.
        Weight, means and inv_covariances are added, det and cst are
        set to 0
        """
        new_mixture = Mixture()
        new_mixture.w = self.w + other.w
        new_mixture.mu = self.mu + other.mu
        new_mixture.invcov = self.invcov + other.invcov
        return new_mixture

    def init_from_diag(self, diag_mixture):
        """

        :param diag_mixture:
        """
        distrib_nb = diag_mixture.w.shape[0]
        dim = diag_mixture.mu.shape[1]

        self.w = diag_mixture.w
        self.cst = diag_mixture.cst
        self.det = diag_mixture.det
        self.mu = diag_mixture.mu

        self.invcov = numpy.empty((distrib_nb, dim, dim))
        self.invchol = numpy.empty((distrib_nb, dim, dim))
        for gg in range(distrib_nb):
            self.invcov[gg] = numpy.diag(diag_mixture.invcov[gg, :])
            self.invchol[gg] = numpy.linalg.cholesky(self.invcov[gg])
            self.cov_var_ctl = numpy.diag(diag_mixture.cov_var_ctl)
        self.name = diag_mixture.name
        self.A = numpy.zeros(self.cst.shape)  # we keep zero here as it is not used for full covariance distributions

    def _serialize(self):
        """
        Serialization is necessary to share the memomry when running multiprocesses
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            sh = self.w.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.w.size)
            self.w = numpy.ctypeslib.as_array(tmp.get_obj())
            self.w = self.w.reshape(sh)

            sh = self.mu.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.mu.size)
            self.mu = numpy.ctypeslib.as_array(tmp.get_obj())
            self.mu = self.mu.reshape(sh)

            sh = self.invcov.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.invcov.size)
            self.invcov = numpy.ctypeslib.as_array(tmp.get_obj())
            self.invcov = self.invcov.reshape(sh)

            sh = self.cov_var_ctl.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.cov_var_ctl.size)
            self.cov_var_ctl = numpy.ctypeslib.as_array(tmp.get_obj())
            self.cov_var_ctl = self.cov_var_ctl.reshape(sh)

            sh = self.cst.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.cst.size)
            self.cst = numpy.ctypeslib.as_array(tmp.get_obj())
            self.cst = self.cst.reshape(sh)

            sh = self.det.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.det.size)
            self.det = numpy.ctypeslib.as_array(tmp.get_obj())
            self.det = self.det.reshape(sh)

    def get_distrib_nb(self):
        """
        Return the number of Gaussian distributions in the mixture
        :return: then number of distributions
        """
        return self.w.shape[0]

    def read(self, mixture_file_name, prefix=''):
        """Read a Mixture in hdf5 format

        :param mixture_file_name: name of the file to read from
        :param prefix:
        """
        with h5py.File(mixture_file_name, 'r') as f:
            self.w = f.get(prefix+'w').value
            self.w.resize(numpy.max(self.w.shape))
            self.mu = f.get(prefix+'mu').value
            self.invcov = f.get(prefix+'invcov').value
            self.invchol = f.get(prefix+'invchol').value
            self.cov_var_ctl = f.get(prefix+'cov_var_ctl').value
            self.cst = f.get(prefix+'cst').value
            self.det = f.get(prefix+'det').value
            self.A = f.get(prefix+'a').value

    @check_path_existance
    def write_alize(self, mixture_file_name):
        """Save a mixture in alize raw format

        :param mixture_file_name: name of the file to write in
        """
        with open(mixture_file_name, 'wb') as of:
            # write the number of distributions per state
            of.write(struct.pack("<I", self.distrib_nb()))
            # Write the dimension of the features
            of.write(struct.pack("<I", self.dim()))
            # Weights
            of.write(struct.pack("<" + "d" * self.w.shape[0], *self.w))
            # For each distribution
            for d in range(self.distrib_nb()):
                # Write the constant
                of.write(struct.pack("<d", self.cst[d]))
                # Write the determinant
                of.write(struct.pack("<d", self.det[d]))
                # write a meaningless char for compatibility purpose
                of.write(struct.pack("<c", bytes(1)))
                # Covariance
                of.write(
                    struct.pack("<" + "d" * self.dim(), *self.invcov[d, :]))
                # Means
                of.write(struct.pack("<" + "d" * self.dim(), *self.mu[d, :]))

    @check_path_existance
    def write(self, mixture_file_name, prefix='', mode='w'):
        """Save a Mixture in hdf5 format

        :param mixture_file_name: the name of the file to write in
        :param prefix: prefix of the group in the HDF5 file
        :param mode: mode of the opening, default is "w"
        """
        f = h5py.File(mixture_file_name, mode)

        f.create_dataset(prefix+'w', self.w.shape, "d", self.w,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'mu', self.mu.shape, "d", self.mu,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'invcov', self.invcov.shape, "d", self.invcov,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'invchol', self.invchol.shape, "d", self.invchol,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'cov_var_ctl', self.cov_var_ctl.shape, "d",
                         self.cov_var_ctl,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'cst', self.cst.shape, "d", self.cst,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'det', self.det.shape, "d", self.det,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'a', self.A.shape, "d", self.A,
                         compression="gzip",
                         fletcher32=True)
        f.close()

    def distrib_nb(self):
        """Return the number of distribution of the Mixture
        
        :return: the number of distribution in the Mixture
        """
        return self.w.shape[0]

    def dim(self):
        """Return the dimension of distributions of the Mixture
        
        :return: an integer, size of the acoustic vectors
        """
        return self.mu.shape[1]

    def sv_size(self):
        """Return the dimension of the super-vector
        
        :return: an integer, size of the mean super-vector
        """
        return self.mu.shape[1] * self.w.shape[0]

    def _compute_all(self):
        """Compute determinant and constant values for each distribution"""
        if self.invcov.ndim == 2:  # for Diagonal covariance only
            self.det = 1.0 / numpy.prod(self.invcov, axis=1)
        elif self.invcov.ndim == 3:  # For full covariance dstributions
            for gg in range(self.mu.shape[0]):
                self.det[gg] = 1./numpy.linalg.det(self.invcov[gg])
                self.invchol[gg] = numpy.linalg.cholesky(self.invcov[gg])

        self.cst = 1.0 / (numpy.sqrt(self.det) * (2.0 * numpy.pi) ** (self.dim() / 2.0))

        if self.invcov.ndim == 2:
            self.A = (numpy.square(self.mu) * self.invcov).sum(1) - 2.0 * (numpy.log(self.w) + numpy.log(self.cst))
        elif self.invcov.ndim == 3:
            self.A = numpy.zeros(self.cst.shape)

    def validate(self):
        """Verify the format of the Mixture
        
        :return: a boolean giving the status of the Mixture
        """
        cov = 'diag'
        ok = (self.w.ndim == 1)
        ok &= (self.det.ndim == 1)
        ok &= (self.cst.ndim == 1)
        ok &= (self.mu.ndim == 2)
        if self.invcov.ndim == 3:
            cov = 'full'
        else:
            ok &= (self.invcov.ndim == 2)

        ok &= (self.w.shape[0] == self.mu.shape[0])
        ok &= (self.w.shape[0] == self.cst.shape[0])
        ok &= (self.w.shape[0] == self.det.shape[0])
        if cov == 'diag':
            ok &= (self.invcov.shape == self.mu.shape)
        else:
            ok &= (self.w.shape[0] == self.invcov.shape[0])
            ok &= (self.mu.shape[1] == self.invcov.shape[1])
            ok &= (self.mu.shape[1] == self.invcov.shape[2])
        return ok

    def get_mean_super_vector(self):
        """Return mean super-vector
        
        :return: an array, super-vector of the mean coefficients
        """
        sv = self.mu.flatten()
        return sv

    def get_invcov_super_vector(self):
        """Return Inverse covariance super-vector
        
        :return: an array, super-vector of the inverse co-variance coefficients
        """
        assert self.invcov.ndim == 2, 'Must be diagonal co-variance.'
        sv = self.invcov.flatten()
        return sv

    def compute_log_posterior_probabilities_full(self, cep, mu=None):
        """ Compute log posterior probabilities for a set of feature frames.

        :param cep: a set of feature frames in a ndarray, one feature per row
        :param mu: a mean super-vector to replace the ubm's one. If it is an empty
              vector, use the UBM

        :return: A ndarray of log-posterior probabilities corresponding to the
              input feature set.
        """
        if cep.ndim == 1:
            cep = cep[:, numpy.newaxis]
        if mu is None:
            mu = self.mu
        tmp = (cep - mu[:, numpy.newaxis, :])
        a = numpy.einsum('ijk,imk->ijm', tmp, self.invchol)
        lp = numpy.log(self.w[:, numpy.newaxis]) + numpy.log(self.cst[:, numpy.newaxis]) - 0.5 * (a * a).sum(-1)

        return lp.T

    def compute_log_posterior_probabilities(self, cep, mu=None):
        """ Compute log posterior probabilities for a set of feature frames.
        
        :param cep: a set of feature frames in a ndarray, one feature per row
        :param mu: a mean super-vector to replace the ubm's one. If it is an empty 
              vector, use the UBM
        
        :return: A ndarray of log-posterior probabilities corresponding to the 
              input feature set.
        """
        if cep.ndim == 1:
            cep = cep[numpy.newaxis, :]
        A = self.A
        if mu is None:
            mu = self.mu
        else:
            # for MAP, Compute the data independent term
            A = (numpy.square(mu.reshape(self.mu.shape)) * self.invcov).sum(1) \
               - 2.0 * (numpy.log(self.w) + numpy.log(self.cst))

        # Compute the data independent term
        B = numpy.dot(numpy.square(cep), self.invcov.T) \
            - 2.0 * numpy.dot(cep, numpy.transpose(mu.reshape(self.mu.shape) * self.invcov))

        # Compute the exponential term
        lp = -0.5 * (B + A)
        return lp

    @staticmethod
    def variance_control(cov, flooring, ceiling, cov_ctl):
        """variance_control for Mixture (florring and ceiling)

        :param cov: covariance to control
        :param flooring: float, florring value
        :param ceiling: float, ceiling value
        :param cov_ctl: co-variance to consider for flooring and ceiling
        """
        floor = flooring * cov_ctl
        ceil = ceiling * cov_ctl

        to_floor = numpy.less_equal(cov, floor)
        to_ceil = numpy.greater_equal(cov, ceil)

        cov[to_floor] = floor[to_floor]
        cov[to_ceil] = ceil[to_ceil]
        return cov

    def _reset(self):
        """Set all the Mixture values to ZERO"""
        self.cst.fill(0.0)
        self.det.fill(0.0)
        self.w.fill(0.0)
        self.mu.fill(0.0)
        self.invcov.fill(0.0)
        self.A = 0.0

    def _split_ditribution(self):
        """Split each distribution into two depending on the principal
            axis of variance."""
        sigma = 1.0 / self.invcov
        sig_max = numpy.max(sigma, axis=1)
        arg_max = numpy.argmax(sigma, axis=1)

        shift = numpy.zeros(self.mu.shape)
        for x, y, z in zip(range(arg_max.shape[0]), arg_max, sig_max):
            shift[x, y] = numpy.sqrt(z)

        self.mu = numpy.vstack((self.mu - shift, self.mu + shift))
        self.invcov = numpy.vstack((self.invcov, self.invcov))
        self.w = numpy.concatenate([self.w, self.w]) * 0.5
        self.cst = numpy.zeros(self.w.shape)
        self.det = numpy.zeros(self.w.shape)
        self.cov_var_ctl = numpy.vstack((self.cov_var_ctl, self.cov_var_ctl))

        self._compute_all()

    def _expectation(self, accum, cep):
        """Expectation step of the EM algorithm. Calculate the expected value 
            of the log likelihood function, with respect to the conditional 
            distribution.
        
        :param accum: a Mixture object to store the accumulated statistics
        :param cep: a set of input feature frames
        
        :return loglk: float, the log-likelihood computed over the input set of 
              feature frames.
        """
        if cep.ndim == 1:
            cep = cep[:, numpy.newaxis]
        if self.invcov.ndim == 2:
            lp = self.compute_log_posterior_probabilities(cep)
        elif self.invcov.ndim == 3:
            lp = self.compute_log_posterior_probabilities_full(cep)
        pp, loglk = sum_log_probabilities(lp)

        # zero order statistics
        accum.w += pp.sum(0)
        # first order statistics
        accum.mu += numpy.dot(cep.T, pp).T
        # second order statistics
        if self.invcov.ndim == 2:
            accum.invcov += numpy.dot(numpy.square(cep.T), pp).T  # version for diagonal covariance
        elif self.invcov.ndim == 3:
            tmp = numpy.einsum('ijk,ilk->ijl', cep[:, :, numpy.newaxis], cep[:, :, numpy.newaxis])
            accum.invcov += numpy.einsum('ijk,im->mjk', tmp, pp)

        # return the log-likelihood
        return loglk

    @process_parallel_lists
    def _expectation_list(self, stat_acc, feature_list, feature_server, llk_acc=numpy.zeros(1), num_thread=1):
        """
        Expectation step of the EM algorithm. Calculate the expected value
        of the log likelihood function, with respect to the conditional
        distribution.

        :param stat_acc:
        :param feature_list:
        :param feature_server:
        :param llk_acc:
        :param num_thread:
        :return:
        """
        stat_acc._reset()
        feature_server.keep_all_features = False
        for feat in feature_list:
            cep = feature_server.load(feat)[0]
            llk_acc[0] += self._expectation(stat_acc, cep)

    def _maximization(self, accum, ceil_cov=10, floor_cov=1e-2):
        """Re-estimate the parmeters of the model which maximize the likelihood
            on the data.
        
        :param accum: a Mixture in which statistics computed during the E step 
              are stored
        :param floor_cov: a constant; minimum bound to consider, default is 1e-200
        """
        self.w = accum.w / numpy.sum(accum.w)
        self.mu = accum.mu / accum.w[:, numpy.newaxis]
        if accum.invcov.ndim == 2:
            cov = accum.invcov / accum.w[:, numpy.newaxis] - numpy.square(self.mu)
            cov = Mixture.variance_control(cov, floor_cov, ceil_cov, self.cov_var_ctl)
            self.invcov = 1.0 / cov
        elif accum.invcov.ndim == 3:
            cov = accum.invcov / accum.w[:, numpy.newaxis, numpy.newaxis] \
                  - numpy.einsum('ijk,ilk->ijl', self.mu[:, :, numpy.newaxis], self.mu[:, :, numpy.newaxis])
            # ADD VARIANCE CONTROL
            for gg in range(self.w.shape[0]):
                self.invcov[gg] = numpy.linalg.inv(cov[gg])
                self.invchol[gg] = numpy.linalg.cholesky(self.invcov[gg]).T
        self._compute_all()

    def _init(self, features_server, feature_list, num_thread=1):
        """
        Initialize a Mixture as a single Gaussian distribution which
        mean and covariance are computed on a set of feature frames

        :param features_server:
        :param feature_list:
        :param num_thread:
        :return:
        """

        # Init using all data
        #ANWAR (ADD)
        # if num_thread <= 1:
        #     num_thread = 1
        #END
        features = features_server.stack_features_parallel(feature_list, num_thread=num_thread)
        n_frames = features.shape[0]
        mu = features.mean(0)
        cov = (features**2).mean(0)

        #n_frames, mu, cov = mean_std_many(features_server, feature_list, in_context=False, num_thread=num_thread)
        self.mu = mu[None]
        self.invcov = 1./cov[None]
        self.w = numpy.asarray([1.0])
        self.cst = numpy.zeros(self.w.shape)
        self.det = numpy.zeros(self.w.shape)
        self.cov_var_ctl = 1.0 / copy.deepcopy(self.invcov)
        self._compute_all()

    def EM_split(self,
                 features_server,
                 feature_list,
                 distrib_nb,
                 iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
                 num_thread=1,
                 llk_gain=0.01,
                 save_partial=False,
                 output_file_name="ubm",
                 ceil_cov=10,
                 floor_cov=1e-2):
        """Expectation-Maximization estimation of the Mixture parameters.
        
        :param features_server: sidekit.FeaturesServer used to load data
        :param feature_list: list of feature files to train the GMM
        :param distrib_nb: final number of distributions
        :param iterations: list of iteration number for each step of the learning process
        :param num_thread: number of thread to launch for parallel computing
        :param llk_gain: limit of the training gain. Stop the training when gain between
                two iterations is less than this value
        :param save_partial: name of the file to save intermediate mixtures,
               if True, save before each split of the distributions
        :param ceil_cov:
        :param floor_cov:
        
        :return llk: a list of log-likelihoods obtained after each iteration
        """
        llk = []

        self._init(features_server, feature_list, num_thread)

        # for N iterations:
        for it in iterations[:int(numpy.log2(distrib_nb))]:
            # print("Iteration:", it)
            # Save current model before spliting
            if save_partial:
                self.write('{}_{}g.h5'.format(output_file_name, self.get_distrib_nb()), prefix='')

            self._split_ditribution()

            # initialize the accumulator
            accum = copy.deepcopy(self)

            for i in tqdm(range(it), desc="Iteration #{}".format(it)):
                accum._reset()

                # serialize the accum
                accum._serialize()
                llk_acc = numpy.zeros(1)
                sh = llk_acc.shape
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                    llk_acc = numpy.ctypeslib.as_array(tmp.get_obj())
                    llk_acc = llk_acc.reshape(sh)

                logging.debug('Expectation')
                # E step
                self._expectation_list(stat_acc=accum,
                                       feature_list=feature_list,
                                       feature_server=features_server,
                                       llk_acc=llk_acc,
                                       num_thread=num_thread)
                llk.append(llk_acc[0] / numpy.sum(accum.w))

                # M step
                logging.debug('Maximisation')
                self._maximization(accum, ceil_cov=ceil_cov, floor_cov=floor_cov)
                if i > 0:
                    # gain = llk[-1] - llk[-2]
                    # if gain < llk_gain:
                        # logging.debug(
                        #    'EM (break) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        #    self.mu.shape[0], i + 1, it, gain, self.name,
                        #    len(cep))
                    #    break
                    # else:
                        # logging.debug(
                        #    'EM (continu) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        #    self.mu.shape[0], i + 1, it, gain, self.name,
                        #    len(cep))
                    #    break
                    pass
                else:
                    # logging.debug(
                    #    'EM (start) distrib_nb: %d %i/%i llk: %f -- %s, %d',
                    #    self.mu.shape[0], i + 1, it, llk[-1],
                    #    self.name, len(cep))
                    pass

        return llk

    def EM_uniform(self, cep, distrib_nb, iteration_min=3, iteration_max=10,
                   llk_gain=0.01, do_init=True):

        """Expectation-Maximization estimation of the Mixture parameters.

        :param cep: set of feature frames to consider
        :param cep: set of feature frames to consider
        :param distrib_nb: number of distributions
        :param iteration_min: minimum number of iterations to perform
        :param iteration_max: maximum number of iterations to perform
        :param llk_gain: gain in term of likelihood, stop the training when the gain is less than this value
        :param do_init: boolean, if True initialize the GMM from the training data

        :return llk: a list of log-likelihoods obtained after each iteration

        """
        llk = []

        if do_init:
            self._init_uniform(cep, distrib_nb)
        accum = copy.deepcopy(self)

        for i in range(0, iteration_max):
            accum._reset()
            # serialize the accum
            accum._serialize()
            llk_acc = numpy.zeros(1)
            sh = llk_acc.shape
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                llk_acc = numpy.ctypeslib.as_array(tmp.get_obj())
                llk_acc = llk_acc.reshape(sh)

            # E step
            # llk.append(self._expectation_parallel(accum, cep, num_thread) / cep.shape[0])
            # self._expectation(accum,cep)
            llk.append(self._expectation(accum, cep) / cep.shape[0])

            # M step
            self._maximization(accum)
            if i > 0:
                gain = llk[-1] - llk[-2]
                if gain < llk_gain and i >= iteration_min:
                    logging.debug(
                        'EM (break) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        self.mu.shape[0], i + 1, iteration_max, gain, self.name,
                        len(cep))
                    break
                else:
                    logging.debug(
                        'EM (continu) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        self.mu.shape[0], i + 1, iteration_max, gain, self.name,
                        len(cep))
            else:
                logging.debug(
                    'EM (start) distrib_nb: %d %i/%i llk: %f -- %s, %d',
                    self.mu.shape[0], i + 1, iteration_max, llk[-1],
                    self.name, len(cep))
        return llk

    def _init_uniform(self, cep, distrib_nb):
        """

        :param cep: matrix of acoustic frames
        :param distrib_nb: number of distributions
        """

        # Load data to initialize the mixture
        # self._init(fs, cep)
        cov_tmp = copy.deepcopy(self.invcov)
        nb = cep.shape[0]
        self.w = numpy.full(distrib_nb, 1.0 / distrib_nb, "d")
        self.cst = numpy.zeros(distrib_nb, "d")
        self.det = numpy.zeros(distrib_nb, "d")

        for i in range(0, distrib_nb):
            start = nb // distrib_nb * i
            end = max(start + 10, nb)
            mean = numpy.mean(cep[start:end, :], axis=0)
            cov = (cep[start:end, :]**2).mean(0)
            if i == 0:
                self.mu = mean
                self.invcov = 1./cov[None]
            else:
                self.mu = numpy.vstack((self.mu, mean))
                self.invcov = numpy.vstack((self.invcov, cov))
        self.cov_var_ctl = 1.0 / copy.deepcopy(self.invcov)

        self._compute_all()

    def EM_diag2full(self, diagonal_mixture, features_server, featureList, iterations=2, num_thread=1):
        """Expectation-Maximization estimation of the Mixture parameters.

        :param features_server: sidekit.FeaturesServer used to load data
        :param featureList: list of feature files to train the GMM
        :param iterations: list of iteration number for each step of the learning process
        :param num_thread: number of thread to launch for parallel computing

        :return llk: a list of log-likelihoods obtained after each iteration
        """
        llk = []

        # Convert the covariance matrices into full ones
        distrib_nb = diagonal_mixture.w.shape[0]
        dim = diagonal_mixture.mu.shape[1]

        self.w = diagonal_mixture.w
        self.cst = diagonal_mixture.cst
        self.det = diagonal_mixture.det
        self.mu = diagonal_mixture.mu

        self.invcov = numpy.empty((distrib_nb, dim, dim))
        self.invchol = numpy.empty((distrib_nb, dim, dim))
        for gg in range(distrib_nb):
            self.invcov[gg] = numpy.diag(diagonal_mixture.invcov[gg, :])
            self.invchol[gg] = numpy.linalg.cholesky(self.invcov[gg])
            self.cov_var_ctl = numpy.diag(diagonal_mixture.cov_var_ctl)
        self.name = diagonal_mixture.name
        self.A = numpy.zeros(self.cst.shape)  # we keep zero here as it is not used for full covariance distributions

        # Create Accumulator
        accum = copy.deepcopy(self)

        # Run iterations of EM
        for it in range(iterations):
            logging.debug('EM convert full it: %d', it)

            accum._reset()

            # serialize the accum
            accum._serialize()
            llk_acc = numpy.zeros(1)
            sh = llk_acc.shape
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                llk_acc = numpy.ctypeslib.as_array(tmp.get_obj())
                llk_acc = llk_acc.reshape(sh)

            logging.debug('Expectation')
            # E step
            self._expectation_list(stat_acc=accum,
                                   feature_list=featureList,
                                   feature_server=features_server,
                                   llk_acc=llk_acc,
                                   num_thread=num_thread)
            llk.append(llk_acc[0] / numpy.sum(accum.w))

            # M step
            logging.debug('Maximisation')
            self._maximization(accum)
            if it > 0:
                # gain = llk[-1] - llk[-2]
                # if gain < llk_gain:
                    # logging.debug(
                    #    'EM (break) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                    #    self.mu.shape[0], i + 1, it, gain, self.name,
                    #    len(cep))
                #    break
                # else:
                    # logging.debug(
                    #    'EM (continu) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                    #    self.mu.shape[0], i + 1, it, gain, self.name,
                    #    len(cep))
                #    break
                pass
            else:
                # logging.debug(
                #    'EM (start) distrib_nb: %d %i/%i llk: %f -- %s, %d',
                #    self.mu.shape[0], i + 1, it, llk[-1],
                #    self.name, len(cep))
                pass

        return llk

    def merge(self, model_list):
        """
        Merge a list of Mixtures into a new one. Weights are normalized uniformly
        :param model_list: a list of Mixture objects to merge
        """
        self.w = numpy.hstack(([mod.w for mod in model_list]))
        self.w /= self.w.sum()

        self.mu = numpy.vstack(([mod.mu for mod in model_list]))
        self.invcov = numpy.vstack(([mod.invcov for mod in model_list]))
        self.invchol = numpy.vstack(([mod.invchol for mod in model_list]))
        self.cov_var_ctl = numpy.vstack(([mod.cov_var_ctl for mod in model_list]))
        self.cst = numpy.hstack(([mod.cst for mod in model_list]))
        self.det = numpy.hstack(([mod.det for mod in model_list]))
        self.name = "_".join([mod.name for mod in model_list])
        self.A = numpy.hstack(([mod.A for mod in model_list]))

        self._compute_all()
        assert self.validate(), "Error while merging models"
