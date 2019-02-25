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
import numpy.matlib
import scipy
from scipy.fftpack.realtransforms import dct
from sidekit.frontend.vad import pre_emphasis
from sidekit.frontend.io import *
from sidekit.frontend.normfeat import *
from sidekit.frontend.features import *
import numpy.matlib

PARAM_TYPE = numpy.float32

__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2019 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def hz2mel(f, htk=True):
    """Convert an array of frequency in Hz into mel.
    
    :param f: frequency to convert
    
    :return: the equivalence on the mel scale.
    """
    if htk:
        return 2595 * numpy.log10(1 + f / 700.)
    else:
        f = numpy.array(f)

        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        f_0 = 0.
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = f < brkfrq

        z = numpy.zeros_like(f)
        # fill in parts separately
        z[linpts] = (f[linpts] - f_0) / f_sp
        z[~linpts] = brkpt + (numpy.log(f[~linpts] / brkfrq)) / numpy.log(logstep)

        if z.shape == (1,):
            return z[0]
        else:
            return z


def mel2hz(z, htk=True):
    """Convert an array of mel values in Hz.
    
    :param m: ndarray of frequencies to convert in Hz.
    
    :return: the equivalent values in Hertz.
    """
    if htk:
        return 700. * (10**(z / 2595.) - 1)
    else:
        z = numpy.array(z, dtype=float)
        f_0 = 0
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = (z < brkpt)

        f = numpy.zeros_like(z)

        # fill in parts separately
        f[linpts] = f_0 + f_sp * z[linpts]
        f[~linpts] = brkfrq * numpy.exp(numpy.log(logstep) * (z[~linpts] - brkpt))

        if f.shape == (1,):
            return f[0]
        else:
            return f


def hz2bark(f):
    """
    Convert frequencies (Hertz) to Bark frequencies

    :param f: the input frequency
    :return:
    """
    return 6. * numpy.arcsinh(f / 600.)


def bark2hz(z):
    """
    Converts frequencies Bark to Hertz (Hz)

    :param z:
    :return:
    """
    return 600. * numpy.sinh(z / 6.)


def compute_delta(features,
                  win=3,
                  method='filter',
                  filt=numpy.array([.25, .5, .25, 0, -.25, -.5, -.25])):
    """features is a 2D-ndarray  each row of features is a a frame
    
    :param features: the feature frames to compute the delta coefficients
    :param win: parameter that set the length of the computation window.
            The size of the window is (win x 2) + 1
    :param method: method used to compute the delta coefficients
        can be diff or filter
    :param filt: definition of the filter to use in "filter" mode, default one
        is similar to SPRO4:  filt=numpy.array([.2, .1, 0, -.1, -.2])
        
    :return: the delta coefficients computed on the original features.
    """
    # First and last features are appended to the begining and the end of the 
    # stream to avoid border effect
    x = numpy.zeros((features.shape[0] + 2 * win, features.shape[1]), dtype=PARAM_TYPE)
    x[:win, :] = features[0, :]
    x[win:-win, :] = features
    x[-win:, :] = features[-1, :]

    delta = numpy.zeros(x.shape, dtype=PARAM_TYPE)

    if method == 'diff':
        filt = numpy.zeros(2 * win + 1, dtype=PARAM_TYPE)
        filt[0] = -1
        filt[-1] = 1

    for i in range(features.shape[1]):
        delta[:, i] = numpy.convolve(features[:, i], filt)

    return delta[win:-win, :]


def pca_dct(cep, left_ctx=12, right_ctx=12, p=None):
    """Apply DCT PCA as in [McLaren 2015] paper:
    Mitchell McLaren and Yun Lei, 'Improved Speaker Recognition 
    Using DCT coefficients as features' in ICASSP, 2015
    
    A 1D-dct is applied to the cepstral coefficients on a temporal
    sliding window.
    The resulting matrix is then flatten and reduced by using a Principal
    Component Analysis.
    
    :param cep: a matrix of cepstral cefficients, 1 line per feature vector
    :param left_ctx: number of frames to consider for left context
    :param right_ctx: number of frames to consider for right context
    :param p: a PCA matrix trained on a developpment set to reduce the
       dimension of the features. P is a portait matrix
    """
    y = numpy.r_[numpy.resize(cep[0, :], (left_ctx, cep.shape[1])),
                 cep,
                 numpy.resize(cep[-1, :], (right_ctx, cep.shape[1]))]

    ceps = framing(y, win_size=left_ctx + 1 + right_ctx).transpose(0, 2, 1)
    dct_temp = (dct_basis(left_ctx + 1 + right_ctx, left_ctx + 1 + right_ctx)).T
    if p is None:
        p = numpy.eye(dct_temp.shape[0] * cep.shape[1], dtype=PARAM_TYPE)
    return (numpy.dot(ceps.reshape(-1, dct_temp.shape[0]),
                      dct_temp).reshape(ceps.shape[0], -1)).dot(p)


def shifted_delta_cepstral(cep, d=1, p=3, k=7):
    """
    Compute the Shifted-Delta-Cepstral features for language identification
    
    :param cep: matrix of feature, 1 vector per line
    :param d: represents the time advance and delay for the delta computation
    :param k: number of delta-cepstral blocks whose delta-cepstral 
       coefficients are stacked to form the final feature vector
    :param p: time shift between consecutive blocks.
    
    return: cepstral coefficient concatenated with shifted deltas
    """

    y = numpy.r_[numpy.resize(cep[0, :], (d, cep.shape[1])),
                 cep,
                 numpy.resize(cep[-1, :], (k * 3 + d, cep.shape[1]))]

    delta = compute_delta(y, win=d, method='diff')
    sdc = numpy.empty((cep.shape[0], cep.shape[1] * k))

    idx = numpy.zeros(delta.shape[0], dtype='bool')
    for ii in range(k):
        idx[d + ii * p] = True
    for ff in range(len(cep)):
        sdc[ff, :] = delta[idx, :].reshape(1, -1)
        idx = numpy.roll(idx, 1)
    return numpy.hstack((cep, sdc))


def trfbank(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt, midfreq=1000):
    """Compute triangular filterbank for cepstral coefficient computation.

    :param fs: sampling frequency of the original signal.
    :param nfft: number of points for the Fourier Transform
    :param lowfreq: lower limit of the frequency band filtered
    :param maxfreq: higher limit of the frequency band filtered
    :param nlinfilt: number of linear filters to use in low frequencies
    :param  nlogfilt: number of log-linear filters to use in high frequencies
    :param midfreq: frequency boundary between linear and log-linear filters

    :return: the filter bank and the central frequencies of each filter
    """
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    # ------------------------
    # Compute the filter bank
    # ------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    frequences = numpy.zeros(nfilt + 2, dtype=PARAM_TYPE)
    if nlogfilt == 0:
        linsc = (maxfreq - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt + 2] = lowfreq + numpy.arange(nlinfilt + 2) * linsc
    elif nlinfilt == 0:
        low_mel = hz2mel(lowfreq)
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2)
        # mels[nlinfilt:]
        melsc = (max_mel - low_mel) / (nfilt + 1)
        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences = mel2hz(mels)
    else:
        # Compute linear filters on [0;1000Hz]
        linsc = (min([midfreq, maxfreq]) - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
        # Compute log-linear filters on [1000;maxfreq]
        low_mel = hz2mel(min([1000, maxfreq]))
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2, dtype=PARAM_TYPE)
        melsc = (max_mel - low_mel) / (nlogfilt + 1)

        # Verify that mel2hz(melsc)>linsc
        while mel2hz(melsc) < linsc:
            # in this case, we add a linear filter
            nlinfilt += 1
            nlogfilt -= 1
            frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
            low_mel = hz2mel(frequences[nlinfilt - 1] + 2 * linsc)
            max_mel = hz2mel(maxfreq)
            mels = numpy.zeros(nlogfilt + 2, dtype=PARAM_TYPE)
            melsc = (max_mel - low_mel) / (nlogfilt + 1)

        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences[nlinfilt:] = mel2hz(mels)

    heights = 2. / (frequences[2:] - frequences[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, int(numpy.floor(nfft / 2)) + 1), dtype=PARAM_TYPE)
    # FFT bins (in Hz)
    n_frequences = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = frequences[i]
        cen = frequences[i + 1]
        hi = frequences[i + 2]

        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1, numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        left_slope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1,
                           min(numpy.floor(hi * nfft / fs) + 1, nfft), dtype=numpy.int)
        right_slope = heights[i] / (hi - cen)
        fbank[i][lid] = left_slope * (n_frequences[lid] - low)
        fbank[i][rid[:-1]] = right_slope * (hi - n_frequences[rid[:-1]])

    return fbank, frequences


def mel_filter_bank(fs, nfft, lowfreq, maxfreq, widest_nlogfilt, widest_lowfreq, widest_maxfreq,):
    """Compute triangular filterbank for cepstral coefficient computation.

    :param fs: sampling frequency of the original signal.
    :param nfft: number of points for the Fourier Transform
    :param lowfreq: lower limit of the frequency band filtered
    :param maxfreq: higher limit of the frequency band filtered
    :param widest_nlogfilt: number of log filters
    :param widest_lowfreq: lower frequency of the filter bank
    :param widest_maxfreq: higher frequency of the filter bank
    :param widest_maxfreq: higher frequency of the filter bank

    :return: the filter bank and the central frequencies of each filter
    """

    # ------------------------
    # Compute the filter bank
    # ------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    widest_freqs = numpy.zeros(widest_nlogfilt + 2, dtype=PARAM_TYPE)
    low_mel = hz2mel(widest_lowfreq)
    max_mel = hz2mel(widest_maxfreq)
    mels = numpy.zeros(widest_nlogfilt+2)
    melsc = (max_mel - low_mel) / (widest_nlogfilt + 1)
    mels[:widest_nlogfilt + 2] = low_mel + numpy.arange(widest_nlogfilt + 2) * melsc
    # Back to the frequency domain
    widest_freqs = mel2hz(mels)

    # Select filters in the narrow band
    sub_band_freqs = numpy.array([fr for fr in widest_freqs if lowfreq <= fr <= maxfreq], dtype=PARAM_TYPE)

    heights = 2./(sub_band_freqs[2:] - sub_band_freqs[0:-2])
    nfilt = sub_band_freqs.shape[0] - 2

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, numpy.floor(nfft/2)+1), dtype=PARAM_TYPE)
    # FFT bins (in Hz)
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = sub_band_freqs[i]
        cen = sub_band_freqs[i+1]
        hi = sub_band_freqs[i+2]
        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1, numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        left_slope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1, min(numpy.floor(hi * nfft / fs) + 1,
                                                                 nfft), dtype=numpy.int)
        right_slope = heights[i] / (hi - cen)
        fbank[i][lid] = left_slope * (nfreqs[lid] - low)
        fbank[i][rid[:-1]] = right_slope * (hi - nfreqs[rid[:-1]])

    return fbank, sub_band_freqs


def power_spectrum(input_sig,
                   fs=8000,
                   win_time=0.025,
                   shift=0.01,
                   prefac=0.97):
    """
    Compute the power spectrum of the signal.
    :param input_sig:
    :param fs:
    :param win_time:
    :param shift:
    :param prefac:
    :return:
    """
    window_length = int(round(win_time * fs))
    overlap = window_length - int(shift * fs)
    framed = framing(input_sig, window_length, win_shift=window_length-overlap).copy()

    # Pre-emphasis filtering is applied after framing to be consistent with stream processing
    framed = pre_emphasis(framed, prefac)

    l = framed.shape[0]
    n_fft = 2 ** int(numpy.ceil(numpy.log2(window_length)))
    # Windowing has been changed to hanning which is supposed to have less noisy sidelobes
    # ham = numpy.hamming(window_length)
    window = numpy.hanning(window_length)

    spec = numpy.ones((l, int(n_fft / 2) + 1), dtype=PARAM_TYPE)
    log_energy = numpy.log((framed**2).sum(axis=1))
    dec = 500000
    start = 0
    stop = min(dec, l)
    while start < l:
        ahan = framed[start:stop, :] * window
        mag = numpy.fft.rfft(ahan, n_fft, axis=-1)
        spec[start:stop, :] = mag.real**2 + mag.imag**2
        start = stop
        stop = min(stop + dec, l)

    return spec, log_energy


def mfcc(input_sig,
         lowfreq=100, maxfreq=8000,
         nlinfilt=0, nlogfilt=24,
         nwin=0.025,
         fs=16000,
         nceps=13,
         shift=0.01,
         get_spec=False,
         get_mspec=False,
         prefac=0.97):
    """Compute Mel Frequency Cepstral Coefficients.

    :param input_sig: input signal from which the coefficients are computed.
            Input audio is supposed to be RAW PCM 16bits
    :param lowfreq: lower limit of the frequency band filtered. 
            Default is 100Hz.
    :param maxfreq: higher limit of the frequency band filtered.
            Default is 8000Hz.
    :param nlinfilt: number of linear filters to use in low frequencies.
            Default is 0.
    :param nlogfilt: number of log-linear filters to use in high frequencies.
            Default is 24.
    :param nwin: length of the sliding window in seconds
            Default is 0.025.
    :param fs: sampling frequency of the original signal. Default is 16000Hz.
    :param nceps: number of cepstral coefficients to extract. 
            Default is 13.
    :param shift: shift between two analyses. Default is 0.01 (10ms).
    :param get_spec: boolean, if true returns the spectrogram
    :param get_mspec:  boolean, if true returns the output of the filter banks
    :param prefac: pre-emphasis filter value

    :return: the cepstral coefficients in a ndaray as well as 
            the Log-spectrum in the mel-domain in a ndarray.

    .. note:: MFCC are computed as follows:
        
            - Pre-processing in time-domain (pre-emphasizing)
            - Compute the spectrum amplitude by windowing with a Hamming window
            - Filter the signal in the spectral domain with a triangular filter-bank, whose filters are approximatively
               linearly spaced on the mel scale, and have equal bandwith in the mel scale
            - Compute the DCT of the log-spectrom
            - Log-energy is returned as first coefficient of the feature vector.
    
    For more details, refer to [Davis80]_.
    """
    # Compute power spectrum
    spec, log_energy = power_spectrum(input_sig,
                                      fs,
                                      win_time=nwin,
                                      shift=shift,
                                      prefac=prefac)

    # Filter the spectrum through the triangle filter-bank
    n_fft = 2 ** int(numpy.ceil(numpy.log2(int(round(nwin * fs)))))
    fbank = trfbank(fs, n_fft, lowfreq, maxfreq, nlinfilt, nlogfilt)[0]

    mspec = numpy.log(numpy.dot(spec, fbank.T))   # A tester avec log10 et log

    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    # The C0 term is removed as it is the constant term
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, 1:nceps + 1]
    lst = list()
    lst.append(ceps)
    lst.append(log_energy)
    if get_spec:
        lst.append(spec)
    else:
        lst.append(None)
        del spec
    if get_mspec:
        lst.append(mspec)
    else:
        lst.append(None)
        del mspec

    return lst


def fft2barkmx(n_fft, fs, nfilts=0, width=1., minfreq=0., maxfreq=8000):
    """
    Generate a matrix of weights to combine FFT bins into Bark
    bins.  n_fft defines the source FFT size at sampling rate fs.
    Optional nfilts specifies the number of output bands required
    (else one per bark), and width is the constant width of each
    band in Bark (default 1).
    While wts has n_fft columns, the second half are all zero.
    Hence, Bark spectrum is fft2barkmx(n_fft,fs) * abs(fft(xincols, n_fft));
    2004-09-05  dpwe@ee.columbia.edu  based on rastamat/audspec.m

    :param n_fft: the source FFT size at sampling rate fs
    :param fs: sampling rate
    :param nfilts: number of output bands required
    :param width: constant width of each band in Bark (default 1)
    :param minfreq:
    :param maxfreq:
    :return: a matrix of weights to combine FFT bins into Bark bins
    """
    maxfreq = min(maxfreq, fs / 2.)

    min_bark = hz2bark(minfreq)
    nyqbark = hz2bark(maxfreq) - min_bark

    if nfilts == 0:
        nfilts = numpy.ceil(nyqbark) + 1

    wts = numpy.zeros((nfilts, n_fft))

    # bark per filt
    step_barks = nyqbark / (nfilts - 1)

    # Frequency of each FFT bin in Bark
    binbarks = hz2bark(numpy.arange(n_fft / 2 + 1) * fs / n_fft)

    for i in range(nfilts):
        f_bark_mid = min_bark + i * step_barks
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        lof = (binbarks - f_bark_mid - 0.5)
        hif = (binbarks - f_bark_mid + 0.5)
        wts[i, :n_fft // 2 + 1] = 10 ** (numpy.minimum(numpy.zeros_like(hif), numpy.minimum(hif, -2.5 * lof) / width))

    return wts


def fft2melmx(n_fft,
              fs=8000,
              nfilts=0,
              width=1.,
              minfreq=0,
              maxfreq=4000,
              htkmel=False,
              constamp=False):
    """
    Generate a matrix of weights to combine FFT bins into Mel
    bins.  n_fft defines the source FFT size at sampling rate fs.
    Optional nfilts specifies the number of output bands required
    (else one per "mel/width"), and width is the constant width of each
    band relative to standard Mel (default 1).
    While wts has n_fft columns, the second half are all zero.
    Hence, Mel spectrum is fft2melmx(n_fft,fs)*abs(fft(xincols,n_fft));
    minfreq is the frequency (in Hz) of the lowest band edge;
    default is 0, but 133.33 is a common standard (to skip LF).
    maxfreq is frequency in Hz of upper edge; default fs/2.
    You can exactly duplicate the mel matrix in Slaney's mfcc.m
    as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
    htkmel=1 means use HTK's version of the mel curve, not Slaney's.
    constamp=1 means make integration windows peak at 1, not sum to 1.
    frqs returns bin center frqs.

    % 2004-09-05  dpwe@ee.columbia.edu  based on fft2barkmx

    :param n_fft:
    :param fs:
    :param nfilts:
    :param width:
    :param minfreq:
    :param maxfreq:
    :param htkmel:
    :param constamp:
    :return:
    """
    maxfreq = min(maxfreq, fs / 2.)

    if nfilts == 0:
        nfilts = numpy.ceil(hz2mel(maxfreq, htkmel) / 2.)

    wts = numpy.zeros((nfilts, n_fft))

    # Center freqs of each FFT bin
    fftfrqs = numpy.arange(n_fft / 2 + 1) / n_fft * fs

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfreq, htkmel)
    maxmel = hz2mel(maxfreq, htkmel)
    binfrqs = mel2hz(minmel +  numpy.arange(nfilts + 2) / (nfilts + 1) * (maxmel - minmel), htkmel)

    for i in range(nfilts):
        _fs = binfrqs[i + numpy.arange(3, dtype=int)]
        # scale by width
        _fs = _fs[1] + width * (_fs - _fs[1])
        # lower and upper slopes for all bins
        loslope = (fftfrqs - _fs[0]) / (_fs[1] - __fs[0])
        hislope = (_fs[2] - fftfrqs)/(_fs[2] - _fs[1])

        wts[i, 1 + numpy.arange(n_fft//2 + 1)] =numpy.maximum(numpy.zeros_like(loslope),numpy.minimum(loslope, hislope))

    if not constamp:
        # Slaney-style mel is scaled to be approx constant E per channel
        wts = numpy.dot(numpy.diag(2. / (binfrqs[2 + numpy.arange(nfilts)] - binfrqs[numpy.arange(nfilts)])) , wts)

    # Make sure 2nd half of FFT is zero
    wts[:, n_fft // 2 + 1: n_fft] = 0

    return wts, binfrqs


def audspec(power_spectrum,
            fs=16000,
            nfilts=None,
            fbtype='bark',
            minfreq=0,
            maxfreq=8000,
            sumpower=True,
            bwidth=1.):
    """

    :param power_spectrum:
    :param fs:
    :param nfilts:
    :param fbtype:
    :param minfreq:
    :param maxfreq:
    :param sumpower:
    :param bwidth:
    :return:
    """
    if nfilts is None:
        nfilts = int(numpy.ceil(hz2bark(fs / 2)) + 1)

    if not fs == 16000:
        maxfreq = min(fs / 2, maxfreq)

    nframes, nfreqs = power_spectrum.shape
    n_fft = (nfreqs -1 ) * 2

    if fbtype == 'bark':
        wts = fft2barkmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'mel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'htkmel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq, True, True)
    elif fbtype == 'fcmel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq, True, False)
    else:
        print('fbtype {} not recognized'.format(fbtype))

    wts = wts[:, :nfreqs]

    if sumpower:
        audio_spectrum = power_spectrum.dot(wts.T)
    else:
        audio_spectrum = numpy.dot(numpy.sqrt(power_spectrum), wts.T)**2

    return audio_spectrum, wts


def postaud(x, fmax, fbtype='bark', broaden=0):
    """
    do loudness equalization and cube root compression

    :param x:
    :param fmax:
    :param fbtype:
    :param broaden:
    :return:
    """
    nframes, nbands = x.shape

    # Include frequency points at extremes, discard later
    nfpts = nbands + 2 * broaden

    if fbtype == 'bark':
        bandcfhz = bark2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'mel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'htkmel' or fbtype == 'fcmel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2mel(fmax,1), num=nfpts),1)
    else:
        print('unknown fbtype {}'.format(fbtype))

    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[broaden:(nfpts - broaden)]

    # Hynek's magic equal-loudness-curve formula
    fsq = bandcfhz ** 2
    ftmp = fsq + 1.6e5
    eql = ((fsq / ftmp) ** 2) * ((fsq + 1.44e6) / (fsq + 9.61e6))

    # weight the critical bands
    z = numpy.matlib.repmat(eql.T,nframes,1) * x

    # cube root compress
    z = z ** .33

    # replicate first and last band (because they are unreliable as calculated)
    if broaden == 1:
      y = z[:, numpy.hstack((0,numpy.arange(nbands), nbands - 1))]
    else:
      y = z[:, numpy.hstack((1,numpy.arange(1, nbands - 1), nbands - 2))]

    return y, eql


def dolpc(x, model_order=8):
    """
    compute autoregressive model from spectral magnitude samples

    :param x:
    :param model_order:
    :return:
    """
    nframes, nbands = x.shape

    r = numpy.real(numpy.fft.ifft(numpy.hstack((x,x[:,numpy.arange(nbands-2,0,-1)]))))

    # First half only
    r = r[:, :nbands]

    # Find LPC coeffs by Levinson-Durbin recursion
    y_lpc = numpy.ones((r.shape[0], model_order + 1))

    for ff in range(r.shape[0]):
        y_lpc[ff, 1:], e, _ = levinson(r[ff, :-1].T, order=model_order, allow_singularity=True)
        # Normalize each poly by gain
        y_lpc[ff, :] /= e

    return y_lpc


def lpc2cep(a, nout):
    """
    Convert the LPC 'a' coefficients in each column of lpcas
    into frames of cepstra.
    nout is number of cepstra to produce, defaults to size(lpcas,1)
    2003-04-11 dpwe@ee.columbia.edu

    :param a:
    :param nout:
    :return:
    """
    ncol , nin = a.shape

    order = nin - 1

    if nout is None:
        nout = order + 1

    c = numpy.zeros((ncol, nout))

    # First cep is log(Error) from Durbin
    c[:, 0] = -numpy.log(a[:, 0])

    # Renormalize lpc A coeffs
    a /= numpy.tile(a[:, 0][:, None], (1, nin))

    for n in range(1, nout):
        sum = 0
        for m in range(1, n):
            sum += (n - m)  * a[:, m] * c[:, n - m]
        c[:, n] = -(a[:, n] + sum / n)

    return c


def lpc2spec(lpcas, nout=17):
    """
    Convert LPC coeffs back into spectra
    nout is number of freq channels, default 17 (i.e. for 8 kHz)

    :param lpcas:
    :param nout:
    :return:
    """
    [cols, rows] = lpcas.shape
    order = rows - 1

    gg = lpcas[:, 0]
    aa = lpcas / numpy.tile(gg, (rows,1)).T

    # Calculate the actual z-plane polyvals: nout points around unit circle
    zz = numpy.exp((-1j * numpy.pi / (nout - 1)) * numpy.outer(numpy.arange(nout).T,  numpy.arange(order + 1)))

    # Actual polyvals, in power (mag^2)
    features = ( 1./numpy.abs(aa.dot(zz.T))**2) / numpy.tile(gg, (nout, 1)).T

    F = numpy.zeros((cols, rows-1))
    M = numpy.zeros((cols, rows-1))

    for c in range(cols):
        aaa = aa[c, :]
        rr = numpy.roots(aaa)
        ff = numpy.angle(rr.T)
        zz = numpy.exp(1j * numpy.outer(ff, numpy.arange(len(aaa))))
        mags = numpy.sqrt(((1./numpy.abs(zz.dot(aaa)))**2)/gg[c])
        ix = numpy.argsort(ff)
        keep = ff[ix] > 0
        ix = ix[keep]
        F[c, numpy.arange(len(ix))] = ff[ix]
        M[c, numpy.arange(len(ix))] = mags[ix]

    F = F[:, F.sum(axis=0) != 0]
    M = M[:, M.sum(axis=0) != 0]

    return features, F, M


def spec2cep(spec, ncep=13, type=2):
    """
    Calculate cepstra from spectral samples (in columns of spec)
    Return ncep cepstral rows (defaults to 9)
    This one does type II dct, or type I if type is specified as 1
    dctm returns the DCT matrix that spec was multiplied by to give cep.

    :param spec:
    :param ncep:
    :param type:
    :return:
    """
    nrow, ncol = spec.shape

    # Make the DCT matrix
    dctm = numpy.zeros(ncep, nrow);
    #if type == 2 || type == 3
    #    # this is the orthogonal one, the one you want
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[1:2:(2*nrow-1)]/(2*nrow)*pi) * sqrt(2/nrow);

    #    if type == 2
    #        # make it unitary! (but not for HTK type 3)
    #        dctm(1,:) = dctm(1,:)/sqrt(2);

    #elif type == 4:  # type 1 with implicit repeating of first, last bins
    #    """
    #    Deep in the heart of the rasta/feacalc code, there is the logic
    #    that the first and last auditory bands extend beyond the edge of
    #    the actual spectra, and they are thus copied from their neighbors.
    #    Normally, we just ignore those bands and take the 19 in the middle,
    #    but when feacalc calculates mfccs, it actually takes the cepstrum
    #    over the spectrum *including* the repeated bins at each end.
    #    Here, we simulate 'repeating' the bins and an nrow+2-length
    #    spectrum by adding in extra DCT weight to the first and last
    #    bins.
    #    """
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[1:nrow]/(nrow+1)*pi) * 2;
    #        # Add in edge points at ends (includes fixup scale)
    #        dctm(i,1) = dctm(i,1) + 1;
    #        dctm(i,nrow) = dctm(i,nrow) + ((-1)^(i-1));

    #   dctm = dctm / (2*(nrow+1));
    #else % dpwe type 1 - same as old spec2cep that expanded & used fft
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[0:(nrow-1)]/(nrow-1)*pi) * 2 / (2*(nrow-1));
    #    dctm(:,[1 nrow]) = dctm(:, [1 nrow])/2;

    #cep = dctm*log(spec);
    return None, None, None


def lifter(x, lift=0.6, invs=False):
    """
    Apply lifter to matrix of cepstra (one per column)
    lift = exponent of x i^n liftering
    or, as a negative integer, the length of HTK-style sin-curve liftering.
    If inverse == 1 (default 0), undo the liftering.

    :param x:
    :param lift:
    :param invs:
    :return:
    """
    nfrm , ncep = x.shape

    if lift == 0:
        y = x
    else:
        if lift > 0:
            if lift > 10:
                print('Unlikely lift exponent of {} did you mean -ve?'.format(lift))
            liftwts = numpy.hstack((1, numpy.arange(1, ncep)**lift))

        elif lift < 0:
        # Hack to support HTK liftering
            L = float(-lift)
            if (L != numpy.round(L)):
                print('HTK liftering value {} must be integer'.format(L))

            liftwts = numpy.hstack((1, 1 + L/2*numpy.sin(numpy.arange(1, ncep) * numpy.pi / L)))

        if invs:
            liftwts = 1 / liftwts

        y = x.dot(numpy.diag(liftwts))

    return y


def plp(input_sig,
         nwin=0.025,
         fs=16000,
         plp_order=13,
         shift=0.01,
         get_spec=False,
         get_mspec=False,
         prefac=0.97,
         rasta=True):
    """
    output is matrix of features, row = feature, col = frame

    % fs is sampling rate of samples, defaults to 8000
    % dorasta defaults to 1; if 0, just calculate PLP
    % modelorder is order of PLP model, defaults to 8.  0 -> no PLP

    :param input_sig:
    :param fs: sampling rate of samples default is 8000
    :param rasta: default is True, if False, juste compute PLP
    :param model_order: order of the PLP model, default is 8, 0 means no PLP

    :return: matrix of features, row = features, column are frames
    """
    plp_order -= 1

    # first compute power spectrum
    powspec, log_energy = power_spectrum(input_sig, fs, nwin, shift, prefac)

    # next group to critical bands
    audio_spectrum = audspec(powspec, fs)[0]
    nbands = audio_spectrum.shape[0]

    if rasta:
        # put in log domain
        nl_aspectrum = numpy.log(audio_spectrum)

        #  next do rasta filtering
        ras_nl_aspectrum = rasta_filt(nl_aspectrum)

        # do inverse log
        audio_spectrum = numpy.exp(ras_nl_aspectrum)

    # do final auditory compressions
    post_spectrum = postaud(audio_spectrum, fs / 2.)[0]

    if plp_order > 0:

        # LPC analysis
        lpcas = dolpc(post_spectrum, plp_order)

        # convert lpc to cepstra
        cepstra = lpc2cep(lpcas, plp_order + 1)

        # .. or to spectra
        spectra, F, M = lpc2spec(lpcas, nbands)

    else:

        # No LPC smoothing of spectrum
        spectra = post_spectrum
        cepstra = spec2cep(spectra)

    cepstra = lifter(cepstra, 0.6)

    lst = list()
    lst.append(cepstra)
    lst.append(log_energy)
    if get_spec:
        lst.append(powspec)
    else:
        lst.append(None)
        del powspec
    if get_mspec:
        lst.append(post_spectrum)
    else:
        lst.append(None)
        del post_spectrum

    return lst


def framing(sig, win_size, win_shift=1, context=(0, 0), pad='zeros'):
    """
    :param sig: input signal, can be mono or multi dimensional
    :param win_size: size of the window in term of samples
    :param win_shift: shift of the sliding window in terme of samples
    :param context: tuple of left and right context
    :param pad: can be zeros or edge
    """
    dsize = sig.dtype.itemsize
    if sig.ndim == 1:
        sig = sig[:, numpy.newaxis]
    # Manage padding
    c = (context, ) + (sig.ndim - 1) * ((0, 0), )
    _win_size = win_size + sum(context)
    shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
    strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    if pad == 'zeros':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                                  shape=shape,
                                                  strides=strides).squeeze()
    elif pad == 'edge':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'edge'),
                                                  shape=shape,
                                                  strides=strides).squeeze()


def dct_basis(nbasis, length):
    """
    :param nbasis: number of CT coefficients to keep
    :param length: length of the matrix to process
    :return: a basis of DCT coefficients
    """
    return scipy.fftpack.idct(numpy.eye(nbasis, length), norm='ortho')


def levinson(r, order=None, allow_singularity=False):
    r"""Levinson-Durbin recursion.

    Find the coefficients of a length(r)-1 order autoregressive linear process

    :param r: autocorrelation sequence of length N + 1 (first element being the zero-lag autocorrelation)
    :param order: requested order of the autoregressive coefficients. default is N.
    :param allow_singularity: false by default. Other implementations may be True (e.g., octave)

    :return:
        * the `N+1` autoregressive coefficients :math:`A=(1, a_1...a_N)`
        * the prediction errors
        * the `N` reflections coefficients values

    This algorithm solves the set of complex linear simultaneous equations
    using Levinson algorithm.

    .. math::

        \bold{T}_M \left( \begin{array}{c} 1 \\ \bold{a}_M \end{array} \right) =
        \left( \begin{array}{c} \rho_M \\ \bold{0}_M  \end{array} \right)

    where :math:`\bold{T}_M` is a Hermitian Toeplitz matrix with elements
    :math:`T_0, T_1, \dots ,T_M`.

    .. note:: Solving this equations by Gaussian elimination would
        require :math:`M^3` operations whereas the levinson algorithm
        requires :math:`M^2+M` additions and :math:`M^2+M` multiplications.

    This is equivalent to solve the following symmetric Toeplitz system of
    linear equations

    .. math::

        \left( \begin{array}{cccc}
        r_1 & r_2^* & \dots & r_{n}^*\\
        r_2 & r_1^* & \dots & r_{n-1}^*\\
        \dots & \dots & \dots & \dots\\
        r_n & \dots & r_2 & r_1 \end{array} \right)
        \left( \begin{array}{cccc}
        a_2\\
        a_3 \\
        \dots \\
        a_{N+1}  \end{array} \right)
        =
        \left( \begin{array}{cccc}
        -r_2\\
        -r_3 \\
        \dots \\
        -r_{N+1}  \end{array} \right)

    where :math:`r = (r_1  ... r_{N+1})` is the input autocorrelation vector, and
    :math:`r_i^*` denotes the complex conjugate of :math:`r_i`. The input r is typically
    a vector of autocorrelation coefficients where lag 0 is the first
    element :math:`r_1`.


    .. doctest::

        >>> import numpy; from spectrum import LEVINSON
        >>> T = numpy.array([3., -2+0.5j, .7-1j])
        >>> a, e, k = LEVINSON(T)

    """
    #from numpy import isrealobj
    T0  = numpy.real(r[0])
    T = r[1:]
    M = len(T)

    if order is None:
        M = len(T)
    else:
        assert order <= M, 'order must be less than size of the input data'
        M = order

    realdata = numpy.isrealobj(r)
    if realdata is True:
        A = numpy.zeros(M, dtype=float)
        ref = numpy.zeros(M, dtype=float)
    else:
        A = numpy.zeros(M, dtype=complex)
        ref = numpy.zeros(M, dtype=complex)

    P = T0

    for k in range(M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            #save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + A[j] * T[k-j-1]
            temp = -save / P
        if realdata:
            P = P * (1. - temp**2.)
        else:
            P = P * (1. - (temp.real**2+temp.imag**2))

        if (P <= 0).any() and allow_singularity==False:
            raise ValueError("singular matrix")
        A[k] = temp
        ref[k] = temp # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k+1)//2
        if realdata is True:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj]
                if j != kj:
                    A[kj] += temp*save
        else:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj].conjugate()
                if j != kj:
                    A[kj] = A[kj] + temp * save.conjugate()

    return A, P, ref
