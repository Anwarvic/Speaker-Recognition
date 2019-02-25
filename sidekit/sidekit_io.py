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

:mod:`sidekit_io` provides methods to read and write from and to different 
formats.
"""

import h5py
import array
import numpy
import os
import pickle
import struct
import gzip
import logging
from sidekit.sidekit_wrappers import check_path_existance


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def read_vect(filename):
    """Read vector in ALIZE binary format and return an array
    
    :param filename: name of the file to read from
    
    :return: a numpy.ndarray object
    """
    with open(filename, 'rb') as f:
        struct.unpack("<2l", f.read(8))
        data = array.array("d")
        data.fromstring(f.read())
    return numpy.array(data)


def read_matrix(filename):
    """Read matrix in ALIZE binary format and return a ndarray
    
    :param filename: name of the file to read from
    
    :return: a numpy.ndarray object
    """
    with open(filename, 'rb') as f:
        m_dim = struct.unpack("<2l", f.read(8))
        data = array.array("d")
        data.fromstring(f.read())
        T = numpy.array(data)
        T.resize(m_dim[0], m_dim[1])
    return T


@check_path_existance
def write_matrix(m, filename):
    """Write a  matrix in ALIZE binary format

    :param m: a 2-dimensional ndarray
    :param filename: name of the file to write in

    :exception: TypeError if m is not a 2-dimensional ndarray
    """
    if not m.ndim == 2:
        raise TypeError("To write vector, use write_vect")
    else:
        with open(filename, 'wb') as mf:
            data = numpy.array(m.flatten())
            mf.write(struct.pack("<l", m.shape[0]))
            mf.write(struct.pack("<l", m.shape[1]))
            mf.write(struct.pack("<" + "d" * m.shape[0] * m.shape[1], *data))


@check_path_existance
def write_vect(v, filename):
    """Write a  vector in ALIZE binary format

    :param v: a 1-dimensional ndarray
    :param filename: name of the file to write in
    
    :exception: TypeError if v is not a 1-dimensional ndarray
    """
    if not v.ndim == 1:
        raise TypeError("To write matrix, use write_matrix")
    else:
        with open(filename, 'wb') as mf:
            mf.write(struct.pack("<l", 1))
            mf.write(struct.pack("<l", v.shape[0]))
            mf.write(struct.pack("<" + "d" * v.shape[0], *v))


@check_path_existance
def write_matrix_int(m, filename):
    """Write matrix of int in ALIZE binary format
    
    :param m: a 2-dimensional ndarray of int
    :param filename: name of the file to write in
    """
    if not m.ndim == 2:
        raise TypeError("To write vector, use write_vect")
    if not m.dtype == 'int64':
        raise TypeError("m must be a ndarray of int64")
    with open(filename, 'wb') as mf:
        data = numpy.array(m.flatten())
        mf.write(struct.pack("<l", m.shape[0]))
        mf.write(struct.pack("<l", m.shape[1]))
        mf.write(struct.pack("<" + "l" * m.shape[0] * m.shape[1], *data))


def read_pickle(filename):
    """
    Read a generic pickle file and return the content

    :param filename: name of the pickle file to read

    :return: the content of the file
    """
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


@check_path_existance
def write_pickle(obj, filename):
    """
    Dump an object in a picke file.

    :param obj: object to serialize and write
    :param filename: name of the file to write
    """
    if not (os.path.exists(os.path.dirname(filename)) or os.path.dirname(filename) == ''):
        os.makedirs(os.path.dirname(filename))
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)


@check_path_existance
def write_tv_hdf5(data, output_filename):
    """
    Write the TotalVariability matrix, the mean and the residual covariance in HDF5 format.

    :param data: a tuple of three elements: the matrix, the mean vector and the inverse covariance vector
    :param output_filename: name fo the file to create
    """
    tv = data[0]
    tv_mean = data[1]
    tv_sigma = data[2]
    d = dict()
    d['tv/tv'] = tv
    d['tv/tv_mean'] = tv_mean
    d['tv/tv_sigma'] = tv_sigma
    write_dict_hdf5(d, output_filename)


def read_tv_hdf5(input_filename):
    """
    Read the TotalVariability matrix, the mean and the residual covariance from a HDF5 file.

    :param input_filename: name of the file to read from

    :return: a tuple of three elements: the matrix, the mean vector and the inverse covariance vector
    """
    with h5py.File(input_filename, "r") as f:
        tv = f.get("tv/tv").value
        tv_mean = f.get("tv/tv_mean").value
        tv_sigma = f.get("tv/tv_sigma").value
    return tv, tv_mean, tv_sigma


@check_path_existance
def write_dict_hdf5(data, output_filename):
    """
    Write a dictionary into a HDF5 file

    :param data: the dictionary to write
    :param output_filename: the name of the file to create
    """
    with h5py.File(output_filename, "w") as f:
        for key in data:
            value = data[key]
            if isinstance(value, numpy.ndarray) or isinstance(value, list):
                f.create_dataset(key,
                                 data=value,
                                 compression="gzip",
                                 fletcher32=True)
            else:
                f.create_dataset(key, data=value)


def read_key_hdf5(input_filename, key):
    """
    Read key value from a HDF5 file.

    :param input_filename: the name of the file to read from
    :param key: the name of the key

    :return: a value
    """
    with h5py.File(input_filename, "r") as f:
        return f.get(key).value


def read_dict_hdf5(input_filename):
    """
    Read a dictionary from an HDF5 file.

    :param input_filename: name of the file to read from

    :return: the dictionary
    """
    data = dict()
    with h5py.File(input_filename, "r") as f:
        for key in f.keys():
            logging.debug('key: '+key)
            for key2 in f.get(key).keys():
                data[key+'/'+key2] = f.get(key).get(key2).value
    return data


@check_path_existance
def write_norm_hdf5(data, output_filename):
    """
    Write the normalization parameters into a HDF5 file.

    :param data: a tuple of two lists. The first list contains mean vectors for each iteration,
    the second list contains covariance matrices for each iteration
    :param output_filename: name of the file to write in
    """
    with h5py.File(output_filename, "w") as f:
        means = data[0]
        covs = data[1]
        f.create_dataset("norm/means", data=means,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset("norm/covs", data=covs,
                         compression="gzip",
                         fletcher32=True)


def read_norm_hdf5(input_filename):
    """
    Read normalization parameters from a HDF5 file.

    :param input_filename: the name of the file to read from

    :return: a tuple of two lists. The first list contains mean vectors for each iteration,
        the second list contains covariance matrices for each iteration
    """
    with h5py.File(input_filename, "r") as f:
        means = f.get("norm/means").value
        covs = f.get("norm/covs").value
    return means, covs


@check_path_existance
def write_plda_hdf5(data, output_filename):
    """
    Write a PLDA model in a HDF5 file.

    :param data: a tuple of 4 elements: the mean vector, the between class covariance matrix,
        the within class covariance matrix and the residual matrix
    :param output_filename: the name of the file to read from
    """
    mean = data[0]
    mat_f = data[1]
    mat_g = data[2]
    sigma = data[3]
    with h5py.File(output_filename, "w") as f:
        f.create_dataset("plda/mean", data=mean,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset("plda/f", data=mat_f,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset("plda/g", data=mat_g,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset("plda/sigma", data=sigma,
                         compression="gzip",
                         fletcher32=True)


def read_plda_hdf5(input_filename):
    """
    Read a PLDA model from a HDF5 file.

    :param input_filename: the name of the file to read from

    :return: a tuple of 4 elements: the mean vector, the between class covariance matrix,
        the within class covariance matrix and the residual matrix
    """
    with h5py.File(input_filename, "r") as f:
        mean = f.get("plda/mean").value
        mat_f = f.get("plda/f").value
        mat_g = f.get("plda/g").value
        sigma = f.get("plda/sigma").value
    return mean, mat_f, mat_g, sigma


@check_path_existance
def write_fa_hdf5(data, output_filename):
    """
    Write a generic factor analysis model into a HDF5 file. (Used for instance for JFA storing)

    :param data: a tuple of 5 elements: the mean vector, the between class covariance matrix,
        the within class covariance matrix, the MAP matrix and the residual covariancematrix
    :param output_filename: the name of the file to write to
    :return:
    """
    mean = data[0]
    f = data[1]
    g = data[2]
    h = data[3]
    sigma = data[4]
    with h5py.File(output_filename, "w") as fh:
        kind = numpy.zeros(5, dtype="int16")  # FA with 5 matrix
        if mean is not None:
            kind[0] = 1
            fh.create_dataset("fa/mean", data=mean,
                              compression="gzip",
                              fletcher32=True)
        if f is not None:
            kind[1] = 1
            fh.create_dataset("fa/f", data=f,
                              compression="gzip",
                              fletcher32=True)
        if g is not None:
            kind[2] = 1
            fh.create_dataset("fa/g", data=g,
                              compression="gzip",
                              fletcher32=True)
        if h is not None:
            kind[3] = 1
            fh.create_dataset("fa/h", data=h,
                              compression="gzip",
                              fletcher32=True)
        if sigma is not None:
            kind[4] = 1
            fh.create_dataset("fa/sigma", data=sigma,
                              compression="gzip",
                              fletcher32=True)
        fh.create_dataset("fa/kind", data=kind,
                          compression="gzip",
                          fletcher32=True)


def read_fa_hdf5(input_filename):
    """
    Read a generic FA model from a HDF5 file

    :param input_filename: the name of the file to read from

    :return: a tuple of 5 elements: the mean vector, the between class covariance matrix,
        the within class covariance matrix, the MAP matrix and the residual covariancematrix
    """
    with h5py.File(input_filename, "r") as fh:
        kind = fh.get("fa/kind").value
        mean = f = g = h = sigma = None
        if kind[0] != 0:
            mean = fh.get("fa/mean").value
        if kind[1] != 0:
            f = fh.get("fa/f").value
        if kind[2] != 0:
            g = fh.get("fa/g").value
        if kind[3] != 0:
            h = fh.get("fa/h").value
        if kind[4] != 0:
            sigma = fh.get("fa/sigma").value
    return mean, f, g, h, sigma


def h5merge(output_filename, input_filename_list):
    """
    Merge a list of HDF5 files into a new one.

    :param output_filename: the name of the new file resulting from the merge.
    :param input_filename_list: list of thge input files
    """
    with h5py.File(output_filename, "w") as fo:
        for ifn in input_filename_list:
            logging.debug('read '+ifn)
            data = read_dict_hdf5(ifn)
            for key in data:
                value = data[key]
                if isinstance(value, numpy.ndarray) or isinstance(value, list):
                    fo.create_dataset(key,
                                      data=value,
                                      compression="gzip",
                                      fletcher32=True)
                else:
                    fo.create_dataset(key, data=value)


def init_logging(level=logging.INFO, filename=None):
    """
    Initialize a logger

    :param level: level of messages to catch
    :param filename: name of the output file
    """
    numpy.set_printoptions(linewidth=250, precision=4)
    frm = '%(asctime)s - %(levelname)s - %(message)s'
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(format=frm, level=level)
    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setFormatter(logging.Formatter(frm))
        fh.setLevel(level)
        root.addHandler(fh)


def write_matrix_hdf5(M, filename):
    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("matrix", data=M,
                           compression="gzip",
                           fletcher32=True)


def read_matrix_hdf5(filename):
    with h5py.File(filename, "r") as h5f:
        M = h5f.get("matrix").value
    return M
