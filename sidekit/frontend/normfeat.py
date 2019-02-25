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

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""
import numpy
import pandas
import scipy.stats as stats
from scipy.signal import lfilter


__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2019 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def rasta_filt(x):
    """Apply RASTA filtering to the input signal.
    
    :param x: the input audio signal to filter.
        cols of x = critical bands, rows of x = frame
        same for y but after filtering
        default filter is single pole at 0.94
    """
    x = x.T
    numerator = numpy.arange(.2, -.3, -.1)
    denominator = numpy.array([1, -0.94])

    # Initialize the state.  This avoids a big spike at the beginning
    # resulting from the dc offset level in each band.
    # (this is effectively what rasta/rasta_filt.c does).
    # Because Matlab uses a DF2Trans implementation, we have to
    # specify the FIR part to get the state right (but not the IIR part)
    y = numpy.zeros(x.shape)
    zf = numpy.zeros((x.shape[0], 4))
    for i in range(y.shape[0]):
        y[i, :4], zf[i, :4] = lfilter(numerator, 1, x[i, :4], axis=-1, zi=[0, 0, 0, 0])

    # .. but don't keep any of these values, just output zero at the beginning
    y = numpy.zeros(x.shape)

    # Apply the full filter to the rest of the signal, append it
    for i in range(y.shape[0]):
        y[i, 4:] = lfilter(numerator, denominator, x[i, 4:], axis=-1, zi=zf[i, :])[0]
    
    return y.T


def cms(features, label=None, global_mean=None):
    """Performs cepstral mean subtraction
    
    :param features: a feature stream of dimension dim x nframes 
            where dim is the dimension of the acoustic features and nframes the 
            number of frames in the stream
    :param label: a logical vector
    :param global_mean: pre-computed mean to use for feature normalization if given

    :return: a feature stream
    """
    # If no label file as input: all speech are speech
    if label is None:
        label = numpy.ones(features.shape[0]).astype(bool)
    if label.sum() == 0:
        mu = numpy.zeros((features.shape[1]))
    if global_mean is not None:
        mu = global_mean
    else:
        mu = numpy.mean(features[label, :], axis=0)
    features -= mu


def cmvn(features, label=None, global_mean=None, global_std=None):
    """Performs mean and variance normalization
    
    :param features: a feature stream of dimension dim x nframes 
        where dim is the dimension of the acoustic features and nframes the 
        number of frames in the stream
    :param global_mean: pre-computed mean to use for feature normalization if given
    :param global_std: pre-computed standard deviation to use for feature normalization if given
    :param label: a logical verctor

    :return: a sequence of features
    """
    # If no label file as input: all speech are speech
    if label is None:
        label = numpy.ones(features.shape[0]).astype(bool)

    if global_mean is not None and global_std is not None:
        mu = global_mean
        stdev = global_std
        features -= mu
        features /= stdev

    elif not label.sum() == 0:
        mu = numpy.mean(features[label, :], axis=0)
        stdev = numpy.std(features[label, :], axis=0)
        features -= mu
        features /= stdev


def stg(features, label=None, win=301):
    """Performs feature warping on a sliding window
    
    :param features: a feature stream of dimension dim x nframes 
        where dim is the dimension of the acoustic features and nframes the
        number of frames in the stream
    :param label: label of selected frames to compute the Short Term Gaussianization, by default, al frames are used
    :param win: size of the frame window to consider, must be an odd number to get a symetric context on left and right
    :return: a sequence of features
    """

    # If no label file as input: all speech are speech
    if label is None:
        label = numpy.ones(features.shape[0]).astype(bool)
    speech_features = features[label, :]

    add_a_feature = False
    if win % 2 == 1:
        # one feature per line
        nframes, dim = numpy.shape(speech_features)

        # If the number of frames is not enough for one window
        if nframes < win:
            # if the number of frames is not odd, duplicate the last frame
            # if nframes % 2 == 1:
            if not nframes % 2 == 1:
                nframes += 1
                add_a_feature = True
                speech_features = numpy.concatenate((speech_features, [speech_features[-1, ]]))
            win = nframes

        # create the output feature stream
        stg_features = numpy.zeros(numpy.shape(speech_features))

        # Process first window
        r = numpy.argsort(speech_features[:win, ], axis=0)
        r = numpy.argsort(r, axis=0)
        arg = (r[: (win - 1) / 2] + 0.5) / win
        stg_features[: (win - 1) / 2, :] = stats.norm.ppf(arg, 0, 1)

        # process all following windows except the last one
        for m in range(int((win - 1) / 2), int(nframes - (win - 1) / 2)):
            idx = list(range(int(m - (win - 1) / 2), int(m + (win - 1) / 2 + 1)))
            foo = speech_features[idx, :]
            r = numpy.sum(foo < foo[(win - 1) / 2], axis=0) + 1
            arg = (r - 0.5) / win
            stg_features[m, :] = stats.norm.ppf(arg, 0, 1)

        # Process the last window
        r = numpy.argsort(speech_features[list(range(nframes - win, nframes)), ], axis=0)
        r = numpy.argsort(r, axis=0)
        arg = (r[(win + 1) / 2: win, :] + 0.5) / win
        
        stg_features[list(range(int(nframes - (win - 1) / 2), nframes)), ] = stats.norm.ppf(arg, 0, 1)
    else:
        # Raise an exception
        raise Exception('Sliding window should have an odd length')

    # wrapFeatures = np.copy(features)
    if add_a_feature:
        stg_features = stg_features[:-1]
    features[label, :] = stg_features


def cep_sliding_norm(features, win=301, label=None, center=True, reduce=False):
    """
    Performs a cepstal mean substitution and standard deviation normalization
    in a sliding windows. MFCC is modified.

    :param features: the MFCC, a numpy array
    :param win: the size of the sliding windows
    :param label: vad label if available
    :param center: performs mean subtraction
    :param reduce: performs standard deviation division

    """
    if label is None:
        label = numpy.ones(features.shape[0]).astype(bool)

    if numpy.sum(label) <= win:
        if reduce:
            cmvn(features, label)
        else:
            cms(features, label)
    else:
        d_win = win // 2

        df = pandas.DataFrame(features[label, :])
        r = df.rolling(window=win, center=True)
        mean = r.mean().values
        std = r.std().values

        mean[0:d_win, :] = mean[d_win, :]
        mean[-d_win:, :] = mean[-d_win-1, :]

        std[0:d_win, :] = std[d_win, :]
        std[-d_win:, :] = std[-d_win-1, :]

        if center:
            features[label, :] -= mean
            if reduce:
                features[label, :] /= std
