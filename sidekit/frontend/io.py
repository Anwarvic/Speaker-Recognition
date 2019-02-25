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

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""
import audioop
import decimal
import h5py
import logging
import math
import numpy
import os
import struct
import warnings
import wave
import scipy.signal

from scipy.signal import decimate
from sidekit.sidekit_wrappers import check_path_existance


__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


# HTK parameters
WAVEFORM = 0
LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11
ANON = 12

_E = 0o000100  # has energy
_N = 0o000200  # absolute energy supressed
_D = 0o000400  # has delta coefficients
_A = 0o001000  # has acceleration coefficients
_C = 0o002000  # is compressed
_Z = 0o004000  # has zero mean static coef.
_K = 0o010000  # has CRC checksum
_0 = 0o020000  # has 0th cepstral coef.
_V = 0o040000  # has VQ data
_T = 0o100000  # has third differential coef.

parms16bit = [WAVEFORM, IREFC, DISCRETE]


@check_path_existance
def write_pcm(data, output_file_name):
    """Write signal to single channel PCM 16 bits
    
    :param data: audio signal to write in a RAW PCM file.
    :param output_file_name: name of the file to write
    """
    with open(output_file_name, 'wb') as of:
        if numpy.abs(data).max() < 1.:
            data = numpy.around(numpy.array(data) * 16384, decimals=0).astype('int16')
        of.write(struct.pack('<' + 'h' * data.shape[0], *data))


def read_pcm(input_file_name):
    """Read signal from single channel PCM 16 bits

    :param input_file_name: name of the PCM file to read.
    
    :return: the audio signal read from the file in a ndarray encoded  on 16 bits, None and 2 (depth of the encoding in bytes)
    """
    with open(input_file_name, 'rb') as f:
        f.seek(0, 2)  # Go to te end of the file
        # get the sample count
        sample_count = int(f.tell() / 2)
        f.seek(0, 0)  # got to the begining of the file
        data = numpy.asarray(struct.unpack('<' + 'h' * sample_count, f.read()))
    return data.astype(numpy.float32), None, 2


def read_wav(input_file_name):
    """
    :param input_file_name:
    :return:
    """
    with wave.open(input_file_name, "r") as wfh:
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = wfh.getparams()
        raw = wfh.readframes(nframes * nchannels)
        out = struct.unpack_from("%dh" % nframes * nchannels, raw)
        sig = numpy.reshape(numpy.array(out), (-1, nchannels)).squeeze()
        return sig.astype(numpy.float32), framerate, sampwidth
    

def pcmu2lin(p, s=4004.189931):
    """Convert Mu-law PCM to linear X=(P,S)
    lin = pcmu2lin(pcmu) where pcmu contains a vector
    of mu-law values in the range 0 to 255.
    No checking is performed to see that numbers are in this range.

    Output values are divided by the scale factor s:

        s		Output Range
        1		+-8031	(integer values)
        4004.2	+-2.005649 (default)
        8031		+-1
        8159		+-0.9843118 (+-1 nominal full scale)

    The default scaling factor 4004.189931 is equal to
    sqrt((2207^2 + 5215^2)/2) this follows ITU standard G.711.
    The sine wave with PCM-Mu values [158 139 139 158 30 11 11 30]
    has a mean square value of unity corresponding to 0 dBm0.
    :param p: input signal encoded in PCM mu-law to convert
    :param s: conversion value from mu-scale oto linear scale
    """
    t = 4 / s
    m = 15 - (p % 16)
    q = numpy.floor(p // 128)
    e = (127 - p - m + 128 * q) / 16
    x = (m + 16.5) * numpy.power(2, e) - 16.5
    z = (q - 0.5) * x * t
    return z


def read_sph(input_file_name, mode='p'):
    """
    Read a SPHERE audio file

    :param input_file_name: name of the file to read
    :param mode: specifies the following (\* =default)
    
    .. note::
    
        - Scaling:
        
            - 's'    Auto scale to make data peak = +-1 (use with caution if reading in chunks)
            - 'r'    Raw unscaled data (integer values)
            - 'p'    Scaled to make +-1 equal full scale
            - 'o'    Scale to bin centre rather than bin edge (e.g. 127 rather than 127.5 for 8 bit values,
                     can be combined with n+p,r,s modes)
            - 'n'    Scale to negative peak rather than positive peak (e.g. 128.5 rather than 127.5 for 8 bit values,
                     can be combined with o+p,r,s modes)

        - Format
       
           - 'l'    Little endian data (Intel,DEC) (overrides indication in file)
           - 'b'    Big endian data (non Intel/DEC) (overrides indication in file)

       - File I/O
       
           - 'f'    Do not close file on exit
           - 'd'    Look in data directory: voicebox('dir_data')
           - 'w'    Also read the annotation file \*.wrd if present (as in TIMIT)
           - 't'    Also read the phonetic transcription file \*.phn if present (as in TIMIT)

        - NMAX     maximum number of samples to read (or -1 for unlimited [default])
        - NSKIP    number of samples to skip from start of file (or -1 to continue from previous read when FFX
                   is given instead of FILENAME [default])

    :return: a tupple such that (Y, FS)
    
    .. note::
    
        - Y data matrix of dimension (samples,channels)
        - FS         sample frequency in Hz
        - WRD{\*,2}  cell array with word annotations: WRD{\*,:)={[t_start t_end],'text'} where times are in seconds
                     only present if 'w' option is given
        - PHN{\*,2}  cell array with phoneme annotations: PHN{\*,:)={[t_start	t_end],'phoneme'} where times
                     are in seconds only present if 't' option is present
        - FFX        Cell array containing

            1. filename
            2. header information
        
            1. first header field name
            2. first header field value
            3. format string (e.g. NIST_1A)
            4. 
                1. file id
                2. current position in file
                3. dataoff    byte offset in file to start of data
                4. order  byte order (l or b)
                5. nsamp    number of samples
                6. number of channels
                7. nbytes    bytes per data value
                8. bits    number of bits of precision
                9. fs	sample frequency
                10. min value
                11. max value
                12. coding 0=PCM,1=uLAW + 0=no compression, 0=shorten,20=wavpack,30=shortpack
                13. file not yet decompressed
                
            5. temporary filename

    If no output parameters are specified,
    header information will be printed.
    The code to decode shorten-encoded files, is 
    not yet released with this toolkit.
    """
    codings = dict([('pcm', 1), ('ulaw', 2)])
    compressions = dict([(',embedded-shorten-', 1),
                         (',embedded-wavpack-', 2),
                         (',embedded-shortpack-', 3)])
    byteorder = 'l'
    endianess = dict([('l', '<'), ('b', '>')])

    if not mode == 'p':
        mode = [mode, 'p']
    k = list((m >= 'p') & (m <= 's') for m in mode)
    # scale to input limits not output limits
    mno = all([m != 'o' for m in mode])
    sc = ''
    if k[0]:
        sc = mode[0]
    # Get byte order (little/big endian)
    if any([m == 'l' for m in mode]):
        byteorder = 'l'
    elif any([m == 'b' for m in mode]):
        byteorder = 'b'
    ffx = ['', '', '', '', '']

    if isinstance(input_file_name, str):
        if os.path.exists(input_file_name):
            fid = open(input_file_name, 'rb')
        elif os.path.exists("".join((input_file_name, '.sph'))):
            input_file_name = "".join((input_file_name, '.sph'))
            fid = open(input_file_name, 'rb')
        else:
            raise Exception('Cannot find file {}'.format(input_file_name))
        ffx[0] = input_file_name
    elif not isinstance(input_file_name, str):
        ffx = input_file_name
    else:
        fid = input_file_name

    # Read the header
    if ffx[3] == '':
        fid.seek(0, 0)  # go to the begining of the file
        l1 = fid.readline().decode("utf-8")
        l2 = fid.readline().decode("utf-8")
        if not (l1 == 'NIST_1A\n') & (l2 == '   1024\n'):
            logging.warning('File does not begin with a SPHERE header')
        ffx[2] = l1.rstrip()
        hlen = int(l2[3:7])
        hdr = {}
        while True:  # Read the header and fill a dictionary
            st = fid.readline().decode("utf-8").rstrip()
            if st[0] != ';':
                elt = st.split(' ')
                if elt[0] == 'end_head':
                    break
                if elt[1][0] != '-':
                    logging.warning('Missing ''-'' in SPHERE header')
                    break
                if elt[1][1] == 's':
                    hdr[elt[0]] = elt[2]
                elif elt[1][1] == 'i':
                    hdr[elt[0]] = int(elt[2])
                else:
                    hdr[elt[0]] = float(elt[2])

        if 'sample_byte_format' in list(hdr.keys()):
            if hdr['sample_byte_format'][0] == '0':
                bord = 'l'
            else:
                bord = 'b'
            if (bord != byteorder) & all([m != 'b' for m in mode]) \
                    & all([m != 'l' for m in mode]):
                byteorder = bord

        icode = 0  # Get encoding, default is PCM
        if 'sample_coding' in list(hdr.keys()):
            icode = -1  # unknown code
            for coding in list(codings.keys()):
                if hdr['sample_coding'].startswith(coding):
                    # is the signal compressed
                    # if len(hdr['sample_coding']) > codings[coding]:
                    if len(hdr['sample_coding']) > len(coding):
                        for compression in list(compressions.keys()):
                            if hdr['sample_coding'].endswith(compression):
                                icode = 10 * compressions[compression] \
                                        + codings[coding] - 1
                                break
                    else:  # if the signal is not compressed
                        icode = codings[coding] - 1
                        break
        # initialize info of the files with default values
        info = [fid, 0, hlen, ord(byteorder), 0, 1, 2, 16, 1, 1, -1, icode]
        # Get existing info from the header
        if 'sample_count' in list(hdr.keys()):
            info[4] = hdr['sample_count']
        if not info[4]:  # if no info sample_count or zero
            # go to the end of the file
            fid.seek(0, 2)  # Go to te end of the file
            # get the sample count
            info[4] = int(math.floor((fid.tell() - info[2]) / (info[5] * info[6])))  # get the sample_count
        if 'channel_count' in list(hdr.keys()):
            info[5] = hdr['channel_count']
        if 'sample_n_bytes' in list(hdr.keys()):
            info[6] = hdr['sample_n_bytes']
        if 'sample_sig_bits' in list(hdr.keys()):
            info[7] = hdr['sample_sig_bits']
        if 'sample_rate' in list(hdr.keys()):
            info[8] = hdr['sample_rate']
        if 'sample_min' in list(hdr.keys()):
            info[9] = hdr['sample_min']
        if 'sample_max' in list(hdr.keys()):
            info[10] = hdr['sample_max']

        ffx[1] = hdr
        ffx[3] = info
    info = ffx[3]
    ksamples = info[4]
    if ksamples > 0:
        fid = info[0]
        if (icode >= 10) & (ffx[4] == ''):  # read compressed signal
            # need to use a script with SHORTEN
            raise Exception('compressed signal, need to unpack in a script with SHORTEN')
        info[1] = ksamples
        # use modes o and n to determine effective peak
        pk = 2 ** (8 * info[6] - 1) * (1 + (float(mno) / 2 - int(all([m != 'b'
                                                                      for m in
                                                                      mode]))) / 2 **
                                       info[7])
        fid.seek(1024)  # jump after the header
        nsamples = info[5] * ksamples
        if info[6] < 3:
            if info[6] < 2:
                logging.debug('Sphere i1 PCM')
                y = numpy.fromfile(fid, endianess[byteorder]+"i1", -1)
                if info[11] % 10 == 1:
                    if y.shape[0] % 2:
                        y = numpy.frombuffer(audioop.ulaw2lin(
                                numpy.concatenate((y, numpy.zeros(1, 'int8'))), 2),
                                numpy.int16)[:-1]/32768.
                    else:
                        y = numpy.frombuffer(audioop.ulaw2lin(y, 2), numpy.int16)/32768.
                    pk = 1.
                else:
                    y = y - 128
            else:
                logging.debug('Sphere i2')
                y = numpy.fromfile(fid, endianess[byteorder]+"i2", -1)
        else:  # non verifie
            if info[6] < 4:
                y = numpy.fromfile(fid, endianess[byteorder]+"i1", -1)
                y = y.reshape(nsamples, 3).transpose()
                y = (numpy.dot(numpy.array([1, 256, 65536]), y) - (numpy.dot(y[2, :], 2 ** (-7)).astype(int) * 2 ** 24))
            else:
                y = numpy.fromfile(fid, endianess[byteorder]+"i4", -1)

        if sc != 'r':
            if sc == 's':
                if info[9] > info[10]:
                    info[9] = numpy.min(y)
                    info[10] = numpy.max(y)
                sf = 1 / numpy.max(list(list(map(abs, info[9:11]))), axis=0)
            else:
                sf = 1 / pk
            y = sf * y

        if info[5] > 1:
            y = y.reshape(ksamples, info[5])
    else:
        y = numpy.array([])
    if mode != 'f':
        fid.close()
        info[0] = -1
        if not ffx[4] == '':
            pass  # VERIFY SCRIPT, WHICH CASE IS HANDLED HERE
    return y.astype(numpy.float32), int(info[8]), int(info[6])


def read_audio(input_file_name, framerate=None):
    """ Read a 1 or 2-channel audio file in SPHERE, WAVE or RAW PCM format.
    The format is determined from the file extension.
    If the sample rate read from the file is a multiple of the one given
    as parameter, we apply a decimation function to subsample the signal.
    
    :param input_file_name: name of the file to read from
    :param framerate: frame rate, optional, if lower than the one read from the file, subsampling is applied
    :return: the signal as a numpy array and the sampling frequency
    """
    if framerate is None:
        raise TypeError("Expected sampling frequency required in sidekit.frontend.io.read_audio")
    ext = os.path.splitext(input_file_name)[-1]
    if ext.lower() == '.sph':
        sig, read_framerate, sampwidth = read_sph(input_file_name, 'p')
    elif ext.lower() == '.wav' or ext.lower() == '.wave':
        sig, read_framerate, sampwidth = read_wav(input_file_name)
    elif ext.lower() == '.pcm' or ext.lower() == '.raw':
        sig, read_framerate, sampwidth = read_pcm(input_file_name)
        read_framerate = framerate
    else:
        raise TypeError("Unknown extension of audio file")

    # Convert to 16 bit encoding if needed
    sig *= (2**(15-sampwidth))

    if framerate > read_framerate:
        logging.warning("Warning in read_audio, up-sampling function is not implemented yet!")
    elif read_framerate % float(framerate) == 0 and not framerate == read_framerate:
        logging.info("downsample")
        sig = scipy.signal.decimate(sig, int(read_framerate / float(framerate)), n=None, ftype='iir', axis=0)
    return sig.astype(numpy.float32), framerate


@check_path_existance
def write_label(label,
                output_file_name,
                selected_label='speech',
                frame_per_second=100):
    """Save labels in ALIZE format

    :param output_file_name: name of the file to write to
    :param label: label to write in the file given as a ndarray of boolean
    :param selected_label: label to write to the file. Default is 'speech'.
    :param frame_per_second: number of frame per seconds. Used to convert
            the frame number into time. Default is 100.
    """
    if label.shape[0] > 0:
        bits = label[:-1] ^ label[1:]
        # convert true value into a list of feature indexes
        # append 0 at the beginning of the list, append the last index to the list
        idx = [0] + (numpy.arange(len(bits))[bits] + 1).tolist() + [len(label)]
        framerate = decimal.Decimal(1) / decimal.Decimal(frame_per_second)
        # for each pair of indexes (idx[i] and idx[i+1]), create a segment
        with open(output_file_name, 'w') as fid:
            for i in range(~label[0], len(idx) - 1, 2):
                fid.write('{} {} {}\n'.format(str(idx[i]*framerate),
                                              str(idx[i + 1]*framerate), selected_label))


def read_label(input_file_name, selected_label='speech', frame_per_second=100):
    """Read label file in ALIZE format

    :param input_file_name: the label file name
    :param selected_label: the label to return. Default is 'speech'.
    :param frame_per_second: number of frame per seconds. Used to convert
            the frame number into time. Default is 100.

    :return: a logical array
    """
    with open(input_file_name) as f:
        segments = f.readlines()

    if len(segments) == 0:
        lbl = numpy.zeros(0).astype(bool)
    else:
        # initialize the length from the last segment's end
        foo1, stop, foo2 = segments[-1].rstrip().split()
        lbl = numpy.zeros(int(float(stop) * 100)).astype(bool)
    
        begin = numpy.zeros(len(segments))
        end = numpy.zeros(len(segments))
    
        for s in range(len(segments)):
            start, stop, label = segments[s].rstrip().split()
            if label == selected_label:
                begin[s] = int(round(float(start) * frame_per_second))
                end[s] = int(round(float(stop) * frame_per_second))
                lbl[begin[s]:end[s]] = True
    return lbl


def read_spro4(input_file_name,
               label_file_name="",
               selected_label="",
               frame_per_second=100):
    """Read a feature stream in SPRO4 format 
    
    :param input_file_name: name of the feature file to read from
    :param label_file_name: name of the label file to read if required.
        By Default, the method assumes no label to read from.    
    :param selected_label: label to select in the label file. Default is none.
    :param frame_per_second: number of frame per seconds. Used to convert
            the frame number into time. Default is 0.
    
    :return: a sequence of features in a numpy array
    """
    with open(input_file_name, 'rb') as f:

        tmp_s = struct.unpack("8c", f.read(8))
        s = ()
        for i in range(len(tmp_s)):
            s += (tmp_s[i].decode("utf-8"),)
        f.seek(0, 2)  # Go to te end of the file
        size = f.tell()  # get the position
        f.seek(0, 0)  # go back to the begining of the file
        head_size = 0

        if "".join(s) == '<header>':
            # swap empty header for general header the code need changing
            struct.unpack("19b", f.read(19))
            head_size = 19

        dim = struct.unpack("H", f.read(2))[0]
        struct.unpack("4b", f.read(4))
        struct.unpack("f", f.read(4))
        n_frames = int(math.floor((size - 10 - head_size) / (4 * dim)))

        features = numpy.asarray(struct.unpack('f' * n_frames * dim,
                                               f.read(4 * n_frames * dim)))
        features.resize((n_frames, dim))

    lbl = numpy.ones(numpy.shape(features)[0]).astype(bool)
    if not label_file_name == "":
        lbl = read_label(label_file_name, selected_label, frame_per_second)

    features = features[lbl, :]
    return features.astype(numpy.float32)


def read_hdf5_segment(file_handler,
                      show,
                      dataset_list,
                      label,
                      start=None, stop=None,
                      global_cmvn=False):
    """Read a segment from a stream in HDF5 format. Return the features in the
    range start:end
    In case the start and end cannot be reached, the first or last feature are copied
    so that the length of the returned segment is always end-start

    :param file_name: name of the file to open
    :param dataset: identifier of the dataset in the HDF5 file
    :param mask:
    :param start:
    :param end:

    :return:read_hdf5_segment
    """
    h5f = file_handler

    compression_type = {0: 'none', 1: 'htk', 2: 'percentile'}
    if "compression" not in h5f:
        compression = 'none'
        logging.warning("Warning, default feature storage mode is now using compression")
    else:
        compression = compression_type[h5f["compression"].value]

    if show not in h5f:
        raise Exception('show {} is not in the HDF5 file'.format(show))

    # Get the selected segment
    dataset_length = h5f[show + "/" + next(h5f[show].__iter__())].shape[0]

    # Deal with the case where start < 0 or stop > feat.shape[0]
    if start is None:
        start = 0
    pad_begining = -start if start < 0 else 0
    start = max(start, 0)

    if stop is None:
        stop = dataset_length
    pad_end = stop - dataset_length if stop > dataset_length else 0
    stop = min(stop, dataset_length)
    global_cmvn = global_cmvn and not (start is None or stop is None)

    # Get the data between start and stop
    # Concatenate all required datasets
    feat = []
    global_mean = []
    global_std = []

    feat = []
    for data_id in ['energy', 'cep', 'fb', 'bnf']:
        if data_id in dataset_list:
            if "/".join((show, data_id)) in h5f:
                dataset_id = show + '/{}'.format(data_id)
                if compression == 'none':
                    data = _read_segment(h5f, dataset_id, start, stop)
                    if data.ndim ==1:
                        data = data[:, numpy.newaxis]
                    feat.append(data)
                elif compression == 'htk':
                    feat.append(_read_segment_htk(h5f, dataset_id, start, stop))
                else:
                    feat.append(_read_segment_percentile(h5f, dataset_id, start, stop))
                global_mean.append(h5f["/".join((show, "{}_mean".format(data_id)))].value)
                global_std.append(h5f["/".join((show, "{}_std".format(data_id)))].value)

            else:
                raise Exception('{} is not in the HDF5 file'.format(data_id))

    feat = numpy.hstack(feat)
    global_mean = numpy.hstack(global_mean)
    global_std = numpy.hstack(global_std)

    if label is None:
        if "/".join((show, "vad")) in h5f:
            label = h5f.get("/".join((show, "vad"))).value.astype('bool').squeeze()[start:stop]
        else:
            label = numpy.ones(feat.shape[0], dtype='bool')
    # Pad the segment if needed
    feat = numpy.pad(feat, ((pad_begining, pad_end), (0, 0)), mode='edge')
    label = numpy.pad(label, (pad_begining, pad_end), mode='edge')
    #stop += pad_begining + pad_end

    return  feat, label, global_mean, global_std, global_cmvn


def read_spro4_segment(input_file_name, start=0, end=None):
    """Read a segment from a stream in SPRO4 format. Return the features in the
    range start:end
    In case the start and end cannot be reached, the first or last feature are copied
    so that the length of the returned segment is always end-start
    
    :param input_file_name: name of the feature file to read from
    :param start: index of the first frame to read (start at zero)
    :param end: index of the last frame following the segment to read.
       end < 0 means that end is the value of the right_context to add 
       at the end of the file

    :return: a sequence of features in a ndarray of length end-start
    """
    with open(input_file_name, 'rb') as f:

        tmpS = struct.unpack("8c", f.read(8))
        s = ()
        for i in range(len(tmpS)):
            s += (tmpS[i].decode("utf-8"),)
        f.seek(0, 2)  # Go to te end of the file
        size = f.tell()  # get the position
        f.seek(0, 0)  # go back to the begining of the file
        head_size = 0

        if "".join(s) == '<header>':
            # swap empty header for general header the code need changing
            struct.unpack("19b", f.read(19))
            head_size = 19

        dim = struct.unpack("H", f.read(2))[0]
        struct.unpack("4b", f.read(4))
        struct.unpack("f", f.read(4))
        n_frames = int(math.floor((size - 10 - head_size) / (4 * dim)))
        if end is None:
            end = n_frames
        elif end < 0:
            end = n_frames - end
            
        s, e = max(0, start), min(n_frames, end)
        f.seek(2 + 4 + 4 + dim * 4 * s, 0)
        features = numpy.fromfile(f, '<f', (e-s) * dim)
        features.resize(e-s, dim)
        
    if start != s or end != e:  # repeat first or/and last frame as required
        features = numpy.r_[numpy.repeat(features[[0]], s-start, axis=0),
                            features, numpy.repeat(features[[-1]], end-e, axis=0)]
        
    return features.astype(numpy.float32)


@check_path_existance
def write_spro4(features, output_file_name):
    """Write a feature stream in SPRO4 format.
    
    :param features: sequence of features to write
    :param output_file_name: name of the file to write to
    """
    _, dim = numpy.shape(features)  # get feature stream's dimensions
    f = open(output_file_name, 'wb')  # open outputFile
    f.write(struct.pack("H", dim))  # write feature dimension
    f.write(struct.pack("4b", 25, 0, 0, 0))  # write flag (not important)
    f.write(struct.pack("f", 100.0))  # write frequency of feature extraciton
    data = features.flatten()  # Write the data
    f.write(struct.pack('f' * len(data), *data))
    f.close()


@check_path_existance
def write_htk(features,
              output_file_name,
              framerate=100,
              dt=9):
    """ Write htk feature file

            0. WAVEFORM Acoustic waveform
            1.  LPC Linear prediction coefficients
            2.  LPREFC LPC Reflection coefficients: -lpcar2rf([1 LPC]);LPREFC(1)=[];
            3.  LPCEPSTRA    LPC Cepstral coefficients
            4. LPDELCEP     LPC cepstral+delta coefficients (obsolete)
            5.  IREFC        LPC Reflection coefficients (16 bit fixed point)
            6.  MFCC         Mel frequency cepstral coefficients
            7.  FBANK        Log Fliter bank energies
            8.  MELSPEC      linear Mel-scaled spectrum
            9.  USER         User defined features
            10.  DISCRETE     Vector quantised codebook
            11.  PLP          Perceptual Linear prediction    
    
    :param features: vector for waveforms, one row per frame for other types
    :param output_file_name: name of the file to write to
    :param framerate: feature sample in Hz
    :param dt: data type (also includes Voicebox code for generating data)
        
            0. WAVEFORM Acoustic waveform
            1.  LPC Linear prediction coefficients
            2.  LPREFC LPC Reflection coefficients: -lpcar2rf([1 LPC]);LPREFC(1)=[];
            3.  LPCEPSTRA    LPC Cepstral coefficients
            4. LPDELCEP     LPC cepstral+delta coefficients (obsolete)
            5.  IREFC        LPC Reflection coefficients (16 bit fixed point)
            6.  MFCC         Mel frequency cepstral coefficients
            7.  FBANK        Log Fliter bank energies
            8.  MELSPEC      linear Mel-scaled spectrum
            9.  USER         User defined features
            10.  DISCRETE     Vector quantised codebook
            11.  PLP          Perceptual Linear prediction
            12.  ANON
    """
    sampling_period = 1./framerate
    
    pk = dt & 0x3f
    dt &= ~_K  # clear unsupported CRC bit
    features = numpy.atleast_2d(features)
    if pk == 0:
        features = features.reshape(-1, 1)
    with open(output_file_name, 'wb') as fh:
        fh.write(struct.pack(">IIHH", len(features)+(4 if dt & _C else 0), sampling_period*1e7,
                             features.shape[1] * (2 if (pk in parms16bit or dt & _C) else 4), dt))
        if pk == 5:
            features *= 32767.0
        if pk in parms16bit:
            features = features.astype('>h')
        elif dt & _C:
            mmax, mmin = features.max(axis=0), features.min(axis=0)
            mmax[mmax == mmin] += 32767
            mmin[mmax == mmin] -= 32767  # to avoid division by zero for constant coefficients
            scale = 2 * 32767. / (mmax - mmin)
            bias = 0.5 * scale * (mmax + mmin)
            features = features * scale - bias
            numpy.array([scale]).astype('>f').tofile(fh)
            numpy.array([bias]).astype('>f').tofile(fh)
            features = features.astype('>h')
        else:
            features = features.astype('>f')
        features.tofile(fh)

def read_htk(input_file_name,
             label_file_name="",
             selected_label="",
             frame_per_second=100):
    """Read a sequence of features in HTK format

    :param input_file_name: name of the file to read from
    :param label_file_name: name of the label file to read from
    :param selected_label: label to select
    :param frame_per_second: number of frames per second
    
    :return: a tupple (d, fp, dt, tc, t) described below
    
    .. note::
    
        - d = data: column vector for waveforms, 1 row per frame for other types
        - fp = frame period in seconds
        - dt = data type (also includes Voicebox code for generating data)
        
            0. WAVEFORM Acoustic waveform
            1.  LPC Linear prediction coefficients
            2.  LPREFC LPC Reflection coefficients: -lpcar2rf([1 LPC]);LPREFC(1)=[];
            3.  LPCEPSTRA    LPC Cepstral coefficients
            4. LPDELCEP     LPC cepstral+delta coefficients (obsolete)
            5.  IREFC        LPC Reflection coefficients (16 bit fixed point)
            6.  MFCC         Mel frequency cepstral coefficients
            7.  FBANK        Log Fliter bank energies
            8.  MELSPEC      linear Mel-scaled spectrum
            9.  USER         User defined features
            10.  DISCRETE     Vector quantised codebook
            11.  PLP          Perceptual Linear prediction
            12.  ANON
            
        - tc = full type code = dt plus (optionally) 
                one or more of the following modifiers
                
            - 64  _E  Includes energy terms
            - 128  _N  Suppress absolute energy
            - 256  _D  Include delta coefs
            - 512  _A  Include acceleration coefs
            - 1024  _C  Compressed
            - 2048  _Z  Zero mean static coefs
            - 4096  _K  CRC checksum (not implemented yet)
            - 8192  _0  Include 0'th cepstral coef
            - 16384  _V  Attach VQ index
            - 32768  _T  Attach delta-delta-delta index
            
        - t = text version of type code e.g. LPC_C_K

    This function is a translation of the Matlab code from
    VOICEBOX is a MATLAB toolbox for speech processing.
    by  Mike Brookes
    Home page: `VOICEBOX <http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html>`
    """
    kinds = ['WAVEFORM', 'LPC', 'LPREFC', 'LPCEPSTRA', 'LPDELCEP', 'IREFC',
             'MFCC', 'FBANK', 'MELSPEC', 'USER', 'DISCRETE', 'PLP', 'ANON',
             '???']
    with open(input_file_name, 'rb') as fid:
        nf = struct.unpack(">l", fid.read(4))[0]  # number of frames
        # frame interval (in seconds)
        fp = struct.unpack(">l", fid.read(4))[0] * 1.e-7
        by = struct.unpack(">h", fid.read(2))[0]  # bytes per frame
        tc = struct.unpack(">h", fid.read(2))[0]  # type code
        tc += 65536 * (tc < 0)
        cc = 'ENDACZK0VT'  # list of suffix codes
        nhb = len(cc)  # number of suffix codes
        ndt = 6  # number of bits for base type
        hb = list(int(math.floor(tc * 2 ** x))
                  for x in range(- (ndt + nhb), -ndt + 1))
        # extract bits from type code
        hd = list(hb[x] - 2 * hb[x - 1] for x in range(nhb, 0, -1))
        # low six bits of tc represent data type
        dt = tc - hb[-1] * 2 ** ndt

        # hd(7)=1 CRC check
        # hd(5)=1 compressed data
        if dt == 5:
            fid.seek(0, 2)  # Go to te end of the file
            flen = fid.tell()  # get the position
            fid.seek(0, 0)  # go back to the begining of the file
            if flen > 14 + by * nf:  # if file too long
                dt = 2  # change type to LPRFEC
                hd[4] = 1  # set compressed flag
                nf += 4  # frame count doesn't include
                # compression constants in this case

        # 16 bit data for waveforms, IREFC and DISCRETE
        if any([dt == x for x in [0, 5, 10]]):
            n_dim = int(by * nf / 2)
            data = numpy.asarray(struct.unpack(">" + "h" * n_dim, fid.read(2 * n_dim)))
            d = data.reshape(nf, by / 2)
            if dt == 5:
                d /= 32767  # scale IREFC
        else:
            if hd[4]:  # compressed data - first read scales
                nf -= 4  # frame count includes compression constants
                n_col = int(by / 2)
                scales = numpy.asarray(struct.unpack(">" + "f" * n_col, fid.read(4 * n_col)))
                biases = numpy.asarray(struct.unpack(">" + "f" * n_col, fid.read(4 * n_col)))
                data = numpy.asarray(struct.unpack(">" + "h" * n_col * nf, fid.read(2 * n_col * nf)))
                d = data.reshape(nf, n_col)
                d = d + biases
                d = d / scales
            else:
                data = numpy.asarray(struct.unpack(">" + "f" * int(by / 4) * nf, fid.read(by * nf)))
                d = data.reshape(nf, by / 4)

    t = kinds[min(dt, len(kinds) - 1)]

    lbl = numpy.ones(numpy.shape(d)[0]).astype(bool)
    if not label_file_name == "":
        lbl = read_label(label_file_name, selected_label, frame_per_second)

    d = d[lbl, :]

    return d.astype(numpy.float32), fp, dt, tc, t


def read_htk_segment(input_file_name,
                     start=0,
                     stop=None):
    """Read a segment from a stream in SPRO4 format. Return the features in the
    range start:end
    In case the start and end cannot be reached, the first or last feature are copied
    so that the length of the returned segment is always end-start
    
    :param input_file_name: name of the feature file to read from or file-like
        object alowing to seek in the file
    :param start: index of the first frame to read (start at zero)
    :param stop: index of the last frame following the segment to read.
       end < 0 means that end is the value of the right_context to add 
       at the end of the file
       
    :return: a sequence of features in a ndarray of length end-start
    """
    try:
        fh = open(input_file_name, 'rb')
    except TypeError:
        fh = input_file_name
    try:
        fh.seek(0)
        n_samples, _, sample_size, parm_kind = struct.unpack(">IIHH", fh.read(12))
        pk = parm_kind & 0x3f
        if parm_kind & _C:
            scale, bias = numpy.fromfile(fh, '>f', sample_size).reshape(2, sample_size/2)
            n_samples -= 4
        s, e = max(0, start), min(n_samples, stop)
        fh.seek(s*sample_size, 1)
        dtype, _bytes = ('>h', 2) if parm_kind & _C or pk in parms16bit else ('>f', 4)
        m = numpy.fromfile(fh, dtype, (e - s) * sample_size / _bytes).reshape(e - s, sample_size / _bytes)
        if parm_kind & _C:
            m = (m + bias) / scale
        if pk == IREFC:
            m /= 32767.0
        if pk == WAVEFORM:
            m = m.ravel()
    finally:
        if fh is not input_file_name:
            fh.close()
    if start != s or stop != e:  # repeat first or/and last frame as required
        m = numpy.r_[numpy.repeat(m[[0]], s-start, axis=0), m, numpy.repeat(m[[-1]], stop-e, axis=0)]
    return m.astype(numpy.float32)

def _add_dataset_header(fh,
                        dataset_id,
                        _min_val,
                        _range,
                        _header):
    """
    Create a dataset in the HDF5 file and write the data
    after compressing float to int
    """
    _c_header = (_header - _min_val) / _range
    numpy.clip(_c_header, 0., 1.)
    _c_header = (_c_header * 65535 + 0.499).astype(int)

    fh.create_dataset(dataset_id + '_header',
                      data=_c_header,
                      maxshape=(None, None),
                      compression="gzip",
                      fletcher32=True)
    fh.create_dataset(dataset_id + '_min_range',
                      data=numpy.array([_min_val, _range]).astype('float32'),
                      maxshape=(2,),
                      compression="gzip",
                      fletcher32=True)

def _add_percentile_dataset(fh,
                            dataset_id,
                            data):
    """
    Create the dataset in the HDF5 file, write the data
    compressed in int8 format and the header compressed in
    int format
    """
    _min_val = data.min()
    _range = data.ptp()

    logging.debug("dataset_id = {}\ndata.shape = {}".format(dataset_id, data.shape))
    logging.debug("data.min, max = {}, {}".format(data.min(), data.max()))

    if data.ndim == 1:
        data = data[:, numpy.newaxis]

    # First write the compression information in the dataset header
    _header = numpy.zeros((data.shape[1], 4))

    logging.debug("data.mean()= {}, data.std() = {}".format(data.mean(), data.std()))

    for j, p in enumerate([0, 25, 75, 100]):
        _header[:, j] = numpy.percentile(data, p, axis=0, interpolation='lower')
    _add_dataset_header(fh, dataset_id, _min_val, _range, _header)

    # now write the compressed data
    c_data = numpy.zeros(data.shape, dtype=numpy.uint8)
    for i in range(data.shape[1]):
        p0, p25, p75, p100 = _header[i]
        mat1 = numpy.uint8((((data[:, i] - p0) / (p25 - p0)) * 64 + 0.5))
        mat1 = numpy.clip(mat1, 0, 64) * (data[:, i] < p25)
        mat2 = (numpy.uint8(((data[:, i] - p25) / (p75 - p25)) * 128 + 0.5) + 64)
        mat2 = numpy.clip(mat2, 64, 192) * ((data[:, i] >= p25) & (data[:, i] < p75))
        mat3 = (numpy.uint8(((data[:, i] - p75) / (p100 - p75)) * 63 + 0.5) + 192)
        mat3 = numpy.clip(mat3, 192, 255) * (data[:, i] >= p75)
        c_data[:, i] = mat1 + mat2 + mat3

    logging.debug("p0, p25, p75, p100 = {}, {}, {}, {}".format(p0, p25, p75, p100))
    logging.debug("dans _add_percentile_dataset\n {}".format(c_data[:5,:5]))

    fh.create_dataset(dataset_id,
                      data=c_data,
                      maxshape=(None, None),
                      compression="gzip",
                      fletcher32=True)

def _read_dataset(h5f, dataset_id):
    data = h5f[dataset_id].value
    if data.ndim == 1:
        data = data[:, numpy.newaxis]
    return data

def _read_segment(h5f, dataset_id, s, e):
    data = h5f[dataset_id][s:e]
    return data

def _read_dataset_htk(h5f, dataset_id):
    (A, B) = h5f[dataset_id + "comp"].value
    data = (h5f[dataset_id].value + B) / A
    if data.ndim == 1:
        data = data[:, numpy.newaxis]
    return data

def _read_segment_htk(h5f, dataset_id, e, s):
    (A, B) = h5f[dataset_id + "comp"].value
    data = (h5f[dataset_id][s:e, :] + B) / A
    return data

def _read_dataset_percentile(h5f, dataset_id):
    # read the header
    (_min_val, _range) = h5f[dataset_id + "_min_range"].value
    c_header = h5f[dataset_id + "_header"].value
    _header = numpy.full(c_header.shape, _min_val)
    _header += c_header * _range * 1.52590218966964e-05

    # decompress the data
    c_data = h5f[dataset_id].value
    mat1 = (_header[:,[0]] + (_header[:,[1]] - _header[:,[0]]) * c_data.T * (1/64)) * (c_data.T <= 64)
    mat2 = (_header[:,[1]] + (_header[:,[2]] - _header[:,[1]]) * (c_data.T - 64) * (1/128)) * ((c_data.T > 64) & (c_data.T<=192))
    mat3 = (_header[:,[2]] + (_header[:,[3]] - _header[:,[2]]) * (c_data.T - 192) * (1/63)) * (c_data.T > 192)
    return (mat1+mat2+mat3).T

def _read_segment_percentile(h5f, dataset_id, s, e):
    # read the header
    (_min_val, _range) = h5f[dataset_id + "_min_range"].value
    c_header = h5f[dataset_id + "_header"].value
    _header = numpy.full(c_header.shape, _min_val)
    _header += c_header * _range * 1.52590218966964e-05

    c_data = h5f[dataset_id].value[s:e, :]
    mat1 = (_header[:,[0]] + (_header[:,[1]] - _header[:,[0]]) * c_data.T * (1/64)) * (c_data.T <= 64)
    mat2 = (_header[:,[1]] + (_header[:,[2]] - _header[:,[1]]) * (c_data.T - 64) * (1/128)) * ((c_data.T > 64) & (c_data.T<=192))
    mat3 = (_header[:,[2]] + (_header[:,[3]] - _header[:,[2]]) * (c_data.T - 192) * (1/63)) * (c_data.T > 192)
    return (mat1+mat2+mat3).T


def _write_show(show,
                fh,
                cep, cep_mean, cep_std,
                energy, energy_mean, energy_std,
                fb, fb_mean, fb_std,
                bnf, bnf_mean, bnf_std,
                label):
    if cep is not None:
        fh.create_dataset(show + '/cep', data=cep.astype('float32'),
                          maxshape=(None, None),
                          compression="gzip",
                          fletcher32=True)
    if cep_mean is not None:
        fh.create_dataset(show + '/cep_mean', data=cep_mean.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if cep_std is not None:
        fh.create_dataset(show + '/cep_std', data=cep_std.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if energy is not None:
        fh.create_dataset(show + '/energy', data=energy.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if energy_mean is not None:
        fh.create_dataset(show + '/energy_mean', data=energy_mean)
    if energy_std is not None:
        fh.create_dataset(show + '/energy_std', data=energy_std)
    if fb is not None:
        fh.create_dataset(show + '/fb', data=fb.astype('float32'),
                          maxshape=(None, None),
                          compression="gzip",
                          fletcher32=True)
    if fb_mean is not None:
        fh.create_dataset(show + '/fb_mean', data=fb_mean.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if fb_std is not None:
        fh.create_dataset(show + '/fb_std', data=fb_std.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if bnf is not None:
        fh.create_dataset(show + '/bnf', data=bnf.astype('float32'),
                          maxshape=(None, None),
                          compression="gzip",
                          fletcher32=True)
    if bnf_mean is not None:
        fh.create_dataset(show + '/bnf_mean', data=bnf_mean.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if bnf_std is not None:
        fh.create_dataset(show + '/bnf_std', data=bnf_std.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if label is not None and not show + "/vad" in fh:
        fh.create_dataset(show + '/' + "vad", data=label.astype('int8'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)

def _write_show_htk(show,
                    fh,
                    cep, cep_mean, cep_std,
                    energy, energy_mean, energy_std,
                    fb, fb_mean, fb_std,
                    bnf, bnf_mean, bnf_std,
                    label):
    if cep is not None:
        A_cep = 2 * 32767. / (cep.max() - cep.min())
        B_cep = (cep.max() + cep.min()) * 32767. / (cep.max() - cep.min())
        fh.create_dataset(show + '/cep_comp', data=numpy.array([A_cep, B_cep]).astype('float32'),
                          maxshape=(2,),
                          compression="gzip",
                          fletcher32=True)
        fh.create_dataset(show + '/cep', data=(A_cep*cep - B_cep).astype("short"),
                          maxshape=(None, None),
                          compression="gzip",
                          fletcher32=True)
    if energy is not None:
        A_energy = 2 * 32767. / (energy.max() - energy.min())
        B_energy = (energy.max() + energy.min()) * 32767. / (energy.max() - energy.min())
        fh.create_dataset(show + '/energy_comp', data=numpy.array([A_energy, B_energy]).astype('float32'),
                          maxshape=(2,),
                          compression="gzip",
                          fletcher32=True)
        fh.create_dataset(show + '/energy', data=(A_energy * energy - B_energy).astype("short"),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if fb is not None:
        A_fb = 2 * 32767. / (fb.max() - fb.min())
        B_fb = (fb.max() + fb.min()) * 32767. / (fb.max() - fb.min())
        fh.create_dataset(show + '/fb_comp', data=numpy.array([A_fb, B_fb]).astype('float32'),
                          maxshape=(2,),
                          compression="gzip",
                          fletcher32=True)
        fh.create_dataset(show + '/fb', data=(A_fb * fb - B_fb).astype("short"),
                          maxshape=(None, None),
                          compression="gzip",
                          fletcher32=True)
    if bnf is not None:
        A_bnf = 2 * 32767. / (bnf.max() - bnf.min())
        B_bnf = (bnf.max() + bnf.min()) * 32767. / (bnf.max() - bnf.min())
        fh.create_dataset(show + '/bnf_comp', data=numpy.array([A_bnf, B_bnf]).astype('float32'),
                          maxshape=(2,),
                          compression="gzip",
                          fletcher32=True)
        fh.create_dataset(show + '/bnf', data=(A_bnf * bnf - B_bnf).astype("short"),
                          maxshape=(None, None),
                          compression="gzip",
                          fletcher32=True)
    if energy_mean is not None:
        fh.create_dataset(show + '/energy_mean', data=energy_mean)
    if energy_std is not None:
        fh.create_dataset(show + '/energy_std', data=energy_std)
    if cep_mean is not None:
        fh.create_dataset(show + '/cep_mean', data=cep_mean.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if cep_std is not None:
        fh.create_dataset(show + '/cep_std', data=cep_std.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if fb_mean is not None:
        fh.create_dataset(show + '/fb_mean', data=fb_mean.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if fb_std is not None:
        fh.create_dataset(show + '/fb_std', data=fb_std.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if bnf_mean is not None:
        fh.create_dataset(show + '/bnf_mean', data=bnf_mean.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if bnf_std is not None:
        fh.create_dataset(show + '/bnf_std', data=bnf_std.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)

    if label is not None and not show + "/vad" in fh:
        fh.create_dataset(show + '/' + "vad", data=label.astype('int8'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)

def _write_show_percentile(show,
                           fh,
                           cep, cep_mean, cep_std,
                           energy, energy_mean, energy_std,
                           fb, fb_mean, fb_std,
                           bnf, bnf_mean, bnf_std,
                           label):
    if cep is not None:
        logging.debug("dans add_show_per_centil, cep = {}".format(cep[:5, :5]))
        _add_percentile_dataset(fh, show + '/cep', cep)

    if energy is not None:
        _add_percentile_dataset(fh, show + '/energy', energy)

    if fb is not None:
        _add_percentile_dataset(fh, show + '/fb', fb)

    if bnf is not None:
        _add_percentile_dataset(fh, show + '/bnf', bnf)

    if cep_mean is not None:
        fh.create_dataset(show + '/cep_mean', data=cep_mean.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)

    if cep_std is not None:
        fh.create_dataset(show + '/cep_std', data=cep_std.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)

    if energy_mean is not None:
        fh.create_dataset(show + '/energy_mean', data=energy_mean)

    if energy_std is not None:
        fh.create_dataset(show + '/energy_std', data=energy_std)

    if fb_mean is not None:
        fh.create_dataset(show + '/fb_mean', data=fb_mean.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if fb_std is not None:
        fh.create_dataset(show + '/fb_std', data=fb_std.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if bnf_mean is not None:
        fh.create_dataset(show + '/bnf_mean', data=bnf_mean.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)
    if bnf_std is not None:
        fh.create_dataset(show + '/bnf_std', data=bnf_std.astype('float32'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)

    if label is not None and not show + "/vad" in fh:
        fh.create_dataset(show + '/' + "vad", data=label.astype('int8'),
                          maxshape=(None,),
                          compression="gzip",
                          fletcher32=True)



def write_hdf5(show,
               fh,
               cep, cep_mean, cep_std,
               energy, energy_mean, energy_std,
               fb, fb_mean, fb_std,
               bnf, bnf_mean, bnf_std,
               label,
               compression='percentile'):
    """
    :param show: identifier of the show to write
    :param fh: HDF5 file handler
    :param cep: cepstral coefficients to store
    :param cep_mean: pre-computed mean of the cepstral coefficient
    :param cep_std: pre-computed standard deviation of the cepstral coefficient
    :param energy: energy coefficients to store
    :param energy_mean: pre-computed mean of the energy
    :param energy_std: pre-computed standard deviation of the energy
    :param fb: filter-banks coefficients to store
    :param fb_mean: pre-computed mean of the filter bank coefficient
    :param fb_std: pre-computed standard deviation of the filter bank coefficient
    :param bnf: bottle-neck features to store
    :param bnf_mean: pre-computed mean of the bottleneck features
    :param bnf_std: pre-computed standard deviation of the bottleneck features
    :param label: vad labels to store
    :param compressed: boolean, default is False
    :return:
    """
    #write the the type of compression: could be:
    # 0 = no compression
    # 1 HTK'style compression
    # 2 compression percentile
    compression_type = {'none':0, 'htk':1, 'percentile':2}
    if "compression" not in fh:
        fh.create_dataset('compression', data=compression_type[compression])
    else:
        assert(fh['compression'].value == compression_type[compression])

    if compression == 'none':
        _write_show(show,
                    fh,
                    cep, cep_mean, cep_std,
                    energy, energy_mean, energy_std,
                    fb, fb_mean, fb_std,
                    bnf, bnf_mean, bnf_std,
                    label)
    elif compression == 'htk':
        _write_show_htk(show,
                        fh,
                        cep, cep_mean, cep_std,
                        energy, energy_mean, energy_std,
                        fb, fb_mean, fb_std,
                        bnf, bnf_mean, bnf_std,
                        label)
    else:
        # Default: use percentile compression
        _write_show_percentile(show,
                               fh,
                               cep, cep_mean, cep_std,
                               energy, energy_mean, energy_std,
                               fb, fb_mean, fb_std,
                               bnf, bnf_mean, bnf_std,
                               label)

def read_hdf5(h5f, show, dataset_list=("cep", "fb", "energy", "vad", "bnf")):
    """

    :param h5f: HDF5 file handler to read from
    :param show: identifier of the show to read
    :param dataset_list: list of datasets to read and concatenate
    :return:
    """
    compression_type = {0:'none', 1:'htk', 2:'percentile'}
    if "compression" not in h5f:
        compression = 'none'
        logging.warning("Warning, default feature storage mode is now using compression")
    else:
        compression = compression_type[h5f["compression"].value]

    if show not in h5f:
        raise Exception('show {} is not in the HDF5 file'.format(show))

    # initialize the list of features to concatenate
    feat = []

    if "energy" in dataset_list:
        if "/".join((show, "energy")) in h5f:
            dataset_id = show + '/energy'
            if compression == 'none':
                feat.append(_read_dataset(h5f, dataset_id))
            elif compression == 'htk':
                feat.append(_read_dataset_htk(h5f, dataset_id))
            else:
                feat.append(_read_dataset_percentile(h5f, dataset_id))
        else:
            raise Exception('energy is not in the HDF5 file')

    if "cep" in dataset_list:
        if "/".join((show, "cep")) in h5f:
            dataset_id = show + '/cep'
            if compression == 'none':
                feat.append(_read_dataset(h5f, dataset_id))
            elif compression == 'htk':
                feat.append(_read_dataset_htk(h5f, dataset_id))
            else:
                feat.append(_read_dataset_percentile(h5f, dataset_id))
        else:
            raise Exception('cep) is not in the HDF5 file')

    if "fb" in dataset_list:
        if "/".join((show, "fb")) in h5f:
            dataset_id = show + '/fb'
            if compression == 'none':
                feat.append(_read_dataset(h5f, dataset_id))
            elif compression == 'htk':
                feat.append(_read_dataset_htk(h5f, dataset_id))
            else:
                feat.append(_read_dataset_percentile(h5f, dataset_id))
        else:
            raise Exception('cep) is not in the HDF5 file')

    if "bnf" in dataset_list:
        if "/".join((show, "bnf")) in h5f:
            dataset_id = show + '/bnf'
            if compression == 'none':
                feat.append(_read_dataset(h5f, dataset_id))
            elif compression == 'htk':
                feat.append(_read_dataset_htk(h5f, dataset_id))
            else:
                feat.append(_read_dataset_percentile(h5f, dataset_id))
        else:
            raise Exception('cep) is not in the HDF5 file')

    feat = numpy.hstack(feat)

    label = None
    if "vad" in dataset_list:
        if "/".join((show, "vad")) in h5f:
            label = h5f.get("/".join((show, "vad"))).value.astype('bool').squeeze()
        else:
            warnings.warn("Warning...........no VAD in this HDF5 file")
            label = numpy.ones(feat.shape[0], dtype='bool')

    return feat.astype(numpy.float32), label








