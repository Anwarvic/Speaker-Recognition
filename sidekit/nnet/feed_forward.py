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


The authors would like to thank the BUT Speech@FIT group (http://speech.fit.vutbr.cz) and Lukas BURGET
for sharing the source code that strongly inspired this module. Thank you for your valuable contribution.
"""
import copy
import ctypes
import h5py
import logging
import multiprocessing
import numpy
import os
import time
import torch
import warnings

import sidekit.frontend
from sidekit.sidekit_io import init_logging
from sidekit.sidekit_wrappers import check_path_existance

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def kaldi_to_hdf5(input_file_name, output_file_name):
    """
    Convert a text file containing frame alignment from Kaldi into an
    HDF5 file with the following structure:

        show/start/labels

    :param input_file_name:
    :param output_file_name:
    :return:
    """
    with open(input_file_name, "r") as fh:
        lines = [line.rstrip() for line in fh]

    with h5py.File(output_file_name, "w") as h5f:
        for line in lines[1:-1]:
            show = line.split('_')[0] + '_' + line.split('_')[1]
            start = int(line.split('_')[2].split('-')[0])
            label = numpy.array([int(x) for x in line.split()[1:]], dtype="int16")
            h5f.create_dataset(show + "/{}".format(start), data=label,
                               maxshape=(None,),
                               compression="gzip",
                               fletcher32=True)


def segment_mean_std_hdf5(input_segment):
    """
    Compute the sum and square sum of all features for a list of segments.
    Input files are in HDF5 format

    :param input_segment: list of segments to read from, each element of the list is a tuple of 5 values,
        the filename, the index of thefirst frame, index of the last frame, the number of frames for the
        left context and the number of frames for the right context

    :return: a tuple of three values, the number of frames, the sum of frames and the sum of squares
    """
    features_server, show, start, stop, traps = input_segment
    # Load the segment of frames plus left and right context
    feat, _ = features_server.load(show,
                                   start=start-features_server.context[0],
                                   stop=stop+features_server.context[1])
    if traps:
        # Get traps
        feat, _ = features_server.get_traps(feat=feat,
                                            label=None,
                                            start=features_server.context[0],
                                            stop=feat.shape[0] - features_server.context[1])
    else:
        # Get features in context
        feat, _ = features_server.get_context(feat=feat,
                                              label=None,
                                              start=features_server.context[0],
                                              stop=feat.shape[0] - features_server.context[1])
    return feat.shape[0], feat.sum(axis=0), numpy.sum(feat ** 2, axis=0)


def mean_std_many(features_server, feature_size, seg_list, traps=False, num_thread=1):
    """
    Compute the mean and standard deviation from a list of segments.

    :param features_server: FeaturesServer used to load data
    :param feature_size: dimension o the features to accumulate
    :param seg_list: list of file names with start and stop indices
    :param traps: apply traps processing on the features in context
    :param traps: apply traps processing on the features in context
    :param num_thread: number of parallel processing to run
    :return: a tuple of three values, the number of frames, the mean and the standard deviation
    """
    inputs = [(copy.deepcopy(features_server), seg[0], seg[1], seg[2], traps) for seg in seg_list]
    pool = multiprocessing.Pool(processes=num_thread)
    res = pool.map(segment_mean_std_hdf5, inputs)

    total_n = 0
    total_f = numpy.zeros(feature_size)
    total_s = numpy.zeros(feature_size)
    for N, F, S in res:
        total_n += N
        total_f += F
        total_s += S
    return total_n, total_f / total_n, total_s / total_n



def init_weights(module):
    if type(module) == torch.nn.Linear:
        module.weight.data.normal_(0.0, 0.1)
        if module.bias is not None:
            module.bias.data.uniform_(-4.1, -3.9)


class FForwardNetwork():
    def __init__(self,
                 model,
                 filename=None,
                 input_mean=None,
                 input_std=None,
                 output_file_name=None,
                 optimizer='adam'
                 ):
        """

        """
        self.model = model
        self.input_mean = input_mean
        self.input_std = input_std
        self.optimizer = optimizer
        if output_file_name is None:
            self.output_file_name = "MyModel.mdl"
        else:
            self.output_file_name = output_file_name

    def random_init(self):
        """
        Randomly initialize the model parameters (weights and bias)
        """
        self.model.apply(init_weights)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.model.forward(x)

    def training(self,
                 training_seg_list,
                 cross_validation_seg_list,
                 feature_size,
                 segment_buffer_size=200,
                 batch_size=512,
                 nb_epoch=20,
                 features_server_params=None,
                 output_file_name="",
                 traps=False,
                 logger=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 num_thread=2):

        # shuffle the training list
        shuffle_idx = numpy.random.permutation(numpy.arange(len(training_seg_list)))
        training_seg_list = [training_seg_list[idx] for idx in shuffle_idx]
        # split the list of files to process
        training_segment_sets = [training_seg_list[i:i + segment_buffer_size]
                                 for i in range(0, len(training_seg_list), segment_buffer_size)]

        # If not done yet, compute mean and standard deviation on all training data
        if self.input_mean is None or self.input_std is None:
            logger.critical("Compute mean and std")
            if False:
                fs = sidekit.FeaturesServer(**features_server_params)
                #self.log.info("Compute mean and standard deviation from the training features")
                feature_nb, self.input_mean, self.input_std = mean_std_many(fs,
                                                                            feature_size,
                                                                            training_seg_list,
                                                                            traps=traps,
                                                                            num_thread=num_thread)
                logger.critical("Done")
            else:
                data = numpy.load("mean_std.npz")
                self.input_mean = data["mean"]
                self.input_std = data["std"]

        # Initialized cross validation error
        last_cv_error = -1 * numpy.inf

        for ep in range(nb_epoch):

            logger.critical("Start epoch {} / {}".format(ep + 1, nb_epoch))
            features_server = sidekit.FeaturesServer(**features_server_params)
            running_loss = accuracy = n = nbatch = 0.0

            # Move model to requested device (GPU)
            self.model.to(device)

            # Set training parameters
            self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
            if self.optimizer.lower() == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters())
            elif self.optimizer.lower() == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.0001, momentum=0.9)
            elif self.optimizer.lower() == 'adadelta':
                optimizer = torch.optim.Adadelta(self.model.parameters())
            else:
                logger.critical("unknown optimizer, using default Adam")
                optimizer = torch.optim.Adam(self.model.parameters())

            for idx_mb, file_list in enumerate(training_segment_sets):
                traps = False
                l = []
                f = []
                for idx, val in enumerate(file_list):
                    show, s, _, label = val
                    e = s + len(label)
                    l.append(label)
                    # Load the segment of frames plus left and right context
                    feat, _ = features_server.load(show,
                                                   start=s - features_server.context[0],
                                                   stop=e + features_server.context[1])
                    if traps:
                        # Get features in context
                        f.append(features_server.get_traps(feat=feat,
                                                           label=None,
                                                           start=features_server.context[0],
                                                           stop=feat.shape[0]-features_server.context[1])[0])
                    else:
                        # Get features in context
                        f.append(features_server.get_context(feat=feat,
                                                             label=None,
                                                             start=features_server.context[0],
                                                             stop=feat.shape[0]-features_server.context[1])[0])
                lab = numpy.hstack(l)
                fea = numpy.vstack(f).astype(numpy.float32)
                assert numpy.all(lab != -1) and len(lab) == len(fea)  # make sure that all frames have defined label
                shuffle = numpy.random.permutation(len(lab))
                label = lab.take(shuffle, axis=0)
                data = fea.take(shuffle, axis=0)

                # normalize the input
                data = (data - self.input_mean) / self.input_std

                # Send data and label to the GPU
                data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
                label = torch.from_numpy(label).to(device)

                for jj, (X, t) in enumerate(zip(torch.split(data, batch_size), torch.split(label, batch_size))):

                    optimizer.zero_grad()
                    lab_pred = self.forward(X)
                    loss = self.criterion(lab_pred, t)
                    loss.backward()
                    optimizer.step()

                    accuracy += (torch.argmax(lab_pred.data, 1) == t).sum().item()
                    nbatch += 1
                    n += len(X)
                    running_loss += loss.item() / (batch_size * nbatch)
                    if nbatch % 200 == 199:
                        logger.critical("loss = {} | accuracy = {} ".format(running_loss,  accuracy / n) )

            logger.critical("Start Cross-Validation")
            optimizer.zero_grad()
            running_loss = accuracy = n = nbatch = 0.0

            for ii, cv_segment in enumerate(cross_validation_seg_list):
                show, s, e, label = cv_segment
                e = s + len(label)
                t = torch.from_numpy(label.astype('long')).to(device)
                # Load the segment of frames plus left and right context
                feat, _ = features_server.load(show,
                                               start=s - features_server.context[0],
                                               stop=e + features_server.context[1])
                if traps:
                    # Get features in context
                    X = features_server.get_traps(feat=feat,
                                                  label=None,
                                                  start=features_server.context[0],
                                                  stop=feat.shape[0] - features_server.context[1])[0].astype(numpy.float32)
                else:
                    X = features_server.get_context(feat=feat,
                                                    label=None,
                                                    start=features_server.context[0],
                                                    stop=feat.shape[0] - features_server.context[1])[0].astype(numpy.float32)

                X = (X - self.input_mean) / self.input_std
                lab_pred = self.forward(torch.from_numpy(X).type(torch.FloatTensor).to(device))
                loss = self.criterion(lab_pred, t)
                accuracy += (torch.argmax(lab_pred.data, 1) == t).sum().item()
                n += len(X)
                running_loss += loss.item() / len(X)

            logger.critical("Cross Validation loss = {} | accuracy = {} ".format(running_loss, accuracy / n))

            # Save the current version of the network
            torch.save(self.model.to('cpu').state_dict(), output_file_name.format(ep))

            # Early stopping with very basic loss criteria
            #if last_cv_error >= accuracy / n:
            #    break
            last_cv_error = accuracy / n

    def extract_bnf(self,
                    feature_file_list,
                    features_server,
                    output_file_structure,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    logger=None
                    ):
        """

        :param feature_file_list:
        :param features_server:
        :param output_file_structure:
        :return:
        """
        # Send the model on the device
        self.model.eval()
        self.model.to(device)

        for show in feature_file_list:
            logger.info("Process file %s", show)

            # Load the segment of frames plus left and right context
            feat, label = features_server.load(show)
            feat = (feat - self.input_mean) / self.input_std
            # Get bottle neck features from features in context
            bnf = self.forward(torch.from_numpy(
                (features_server.get_context(feat=feat)[0] -self.input_mean) / self.input_std).type(torch.FloatTensor).to(device)).cpu().detach().numpy()

            # Create the directory if it doesn't exist
            dir_name = os.path.dirname(output_file_structure.format(show))  # get the path
            if not os.path.exists(dir_name) and (dir_name is not ''):
                os.makedirs(dir_name)

            # Save in HDF5 format, labels are saved if they don't exist in the output file
            with h5py.File(output_file_structure.format(show), "a") as h5f:
                vad = None if show + "vad" in h5f else label
                bnf_mean = bnf[vad, :].mean(axis=0)
                bnf_std = bnf[vad, :].std(axis=0)
                sidekit.frontend.io.write_hdf5(show, h5f,
                                               None, None, None,
                                               None, None, None,
                                               None, None, None,
                                               bnf, bnf_mean, bnf_std,
                                               vad,
                                               compressed='percentile')

    def compute_ubm_dnn(self,
                        ndim,
                        training_list,
                        dnn_features_server,
                        features_server,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        viterbi=False):
        """

        :param ndim: number of pseudo-distributions of the UBM to train
        :param training_list: list of files to process to train the model
        :param dnn_features_server: FeaturesServer to feed the network
        :param features_server: FeaturesServer providing features to compute the first and second order statistics
        :param viterbi: boolean, if True, keep only one coefficient to one and the others at zeros
        :return: a Mixture object
        """

        # Accumulate statistics using the DNN (equivalent to E step)
        print("Train a UBM with {} Gaussian distributions".format(ndim))

        # Initialize the accumulator given the size of the first feature file
        feature_size = features_server.load(training_list[0])[0].shape[1]

        # Initialize one Mixture for UBM storage and one Mixture to accumulate the
        # statistics
        ubm = sidekit.Mixture()
        ubm.cov_var_ctl = numpy.ones((ndim, feature_size))

        accum = sidekit.Mixture()
        accum.mu = numpy.zeros((ndim, feature_size), dtype=numpy.float32)
        accum.invcov = numpy.zeros((ndim, feature_size), dtype=numpy.float32)
        accum.w = numpy.zeros(ndim, dtype=numpy.float32)

        self.model.eval()
        self.model.to(device)

        # Compute the zero, first and second order statistics
        for idx, seg in enumerate(training_list):

            print("accumulate stats: {}".format(seg))
            # Process the current segment and get the stat0 per frame
            features, _ = dnn_features_server.load(seg)
            stat_features, labels = features_server.load(seg)

            s0 = self.forward(torch.from_numpy(
                dnn_features_server.get_context(feat=features)[0][labels]).type(torch.FloatTensor).to(device))
            stat_features = stat_features[labels, :]

            s0 = s0.cpu().data.numpy()

            if viterbi:
                max_idx = s0.argmax(axis=1)
                z = numpy.zeros(s0.shape).flatten()
                z[numpy.ravel_multi_index(numpy.vstack((numpy.arange(30), max_idx)), s0.shape)] = 1.
                s0 = z.reshape(s0.shape)

            # zero order statistics
            accum.w += s0.sum(0)

            # first order statistics
            accum.mu += numpy.dot(stat_features.T, s0).T

            # second order statistics
            accum.invcov += numpy.dot(numpy.square(stat_features.T), s0).T

        # M step
        ubm._maximization(accum)

        return ubm

    def compute_stat_dnn(model,
                         segset,
                         stat0,
                         stat1,
                         dnn_features_server,
                         features_server,
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         seg_indices=None):
        """
        Single thread version of the statistic computation using a DNN.

        :param model: neural network as a torch.nn.Module object
        :param segset: list of segments to process
        :param stat0: local matrix of zero-order statistics
        :param stat1: local matrix of first-order statistics
        :param dnn_features_server: FeaturesServer that provides input data for the DNN
        :param features_server: FeaturesServer that provide additional features to compute first order statistics
        :param seg_indices: indices of the
        :return: a StatServer with all computed statistics
        """
        model.cpu()
        for idx in seg_indices:
            logging.debug('Compute statistics for {}'.format(segset[idx]))

            show = segset[idx]
            channel = 0
            if features_server.features_extractor is not None \
                    and show.endswith(features_server.double_channel_extension[1]):
                channel = 1
            stat_features, labels = features_server.load(show, channel=channel)
            features, _ = dnn_features_server.load(show, channel=channel)
            stat_features = stat_features[labels, :]

            s0 = model(torch.from_numpy(dnn_features_server.get_context(feat=features)[0]).type(torch.FloatTensor).cpu())[labels]
            s0.cpu().data.numpy()
            s1 = numpy.dot(stat_features.T, s0).T

            stat0[idx, :] = s0.sum(axis=0)
            stat1[idx, :] = s1.flatten()


    def compute_stat(self,
                     idmap,
                     ndim,
                     dnn_features_server,
                     features_server,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Single thread version of the statistic computation using a DNN.

        :param model: neural network as a torch.nn.Module object
        :param segset: list of segments to process
        :param stat0: local matrix of zero-order statistics
        :param stat1: local matrix of first-order statistics
        :param dnn_features_server: FeaturesServer that provides input data for the DNN
        :param features_server: FeaturesServer that provide additional features to compute first order statistics
        :param seg_indices: indices of the
        :return: a StatServer with all computed statistics
        """
        # get dimension of the features
        feature_size = features_server.load(idmap.rightids[0])[0].shape[1]

        # Create and initialize a StatServer
        ss = sidekit.StatServer(idmap)
        ss.stat0 = numpy.zeros((idmap.leftids.shape[0], ndim), dtype=numpy.float32)
        ss.stat1 = numpy.zeros((idmap.leftids.shape[0], ndim * feature_size), dtype=numpy.float32)

        self.model.cpu()
        for idx in numpy.arange(len(idmap.rightids)):
            logging.debug('Compute statistics for {}'.format(idmap.rightids[idx]))

            show = idmap.rightids[idx]
            channel = 0
            if features_server.features_extractor is not None \
                    and show.endswith(features_server.double_channel_extension[1]):
                channel = 1
            stat_features, labels = features_server.load(show, channel=channel)
            features, _ = dnn_features_server.load(show, channel=channel)
            stat_features = stat_features[labels, :]

            s0 = self.model(torch.from_numpy(
                dnn_features_server.get_context(feat=features)[0][labels]).type(torch.FloatTensor).cpu())

            s0 = s0.cpu().data.numpy()
            s1 = numpy.dot(stat_features.T, s0).T

            ss.stat0[idx, :] = s0.sum(axis=0)
            ss.stat1[idx, :] = s1.flatten()

        # Return StatServer
        return ss

    def compute_stat_dnn_parallel(self,
                                  idmap,
                                  ndim,
                                  dnn_features_server,
                                  features_server,
                                  num_thread=1):
        """

        :param idmap: IdMap that describes segment to process
        :param model: neural netork as a torch.nn.Module object
        :param ndim: number of distributions in the neural network
        :param dnn_features_server: FeaturesServer to feed the Neural Network
        :param features_server: FeaturesServer that provide additional features to compute first order statistics
        :param num_thread: number of parallel process to run
        :return:
        """

        # get dimension of the features
        feature_size = features_server.load(idmap.rightids[0])[0].shape[1]

        # Create and initialize a StatServer
        ss = sidekit.StatServer(idmap)
        ss.stat0 = numpy.zeros((idmap.leftids.shape[0], ndim), dtype=numpy.float32)
        ss.stat1 = numpy.zeros((idmap.leftids.shape[0], ndim * feature_size), dtype=numpy.float32)

        with warnings.catch_warnings():
            ct = ctypes.c_float
            warnings.simplefilter('ignore', RuntimeWarning)
            tmp_stat0 = multiprocessing.Array(ct, ss.stat0.size)
            ss.stat0 = numpy.ctypeslib.as_array(tmp_stat0.get_obj())
            ss.stat0 = ss.stat0.reshape(ss.segset.shape[0], ndim)

            tmp_stat1 = multiprocessing.Array(ct, ss.stat1.size)
            ss.stat1 = numpy.ctypeslib.as_array(tmp_stat1.get_obj())
            ss.stat1 = ss.stat1.reshape(ss.segset.shape[0], ndim * feature_size)

        # Split indices
        sub_lists = numpy.array_split(numpy.arange(idmap.leftids.shape[0]), num_thread)

        # Start parallel processing (make sure THEANO uses CPUs)
        jobs = []
        multiprocessing.freeze_support()
        for idx in range(num_thread):
            p = multiprocessing.Process(target=FForwardNetwork.compute_stat_dnn,
                                        args=(self.model,
                                              ss.segset,
                                              ss.stat0,
                                              ss.stat1,
                                              copy.deepcopy(dnn_features_server),
                                              copy.deepcopy(features_server),
                                              torch.device("cpu"),
                                              sub_lists[idx]))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

        # Return StatServer
        return ss

    def segmental_training(self,
                           training_seg_list,
                           cross_validation_seg_list,
                           feature_size,
                           segment_buffer_size=200,
                           batch_size=512,
                           nb_epoch=20,
                           features_server_params=None,
                           output_file_name="",
                           traps=False,
                           logger=None,
                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                           num_thread=2):

        # shuffle the training list
        shuffle_idx = numpy.random.permutation(numpy.arange(len(training_seg_list)))
        training_seg_list = [training_seg_list[idx] for idx in shuffle_idx]

        # If not done yet, compute mean and standard deviation on all training data
        if self.input_mean is None or self.input_std is None:
            logger.critical("Compute mean and std")
            if False:
                fs = sidekit.FeaturesServer(**features_server_params)
                feature_nb, self.input_mean, self.input_std = mean_std_many(fs,
                                                                            feature_size,
                                                                            training_seg_list,
                                                                            traps=traps,
                                                                            num_thread=num_thread)
                logger.critical("Done")
            else:
                data = numpy.load("mean_std.npz")
                self.input_mean = data["mean"][:24]
                self.input_std = data["std"][:24]

        # Initialized cross validation error
        last_cv_error = -1 * numpy.inf

        for ep in range(nb_epoch):

            logger.critical("Start epoch {} / {}".format(ep + 1, nb_epoch))
            features_server = sidekit.FeaturesServer(**features_server_params)
            running_loss = accuracy = n = nbatch = 0.0

            # Move model to requested device (GPU)
            self.model.to(device)

            # Set training parameters
            self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
            print("optimizer = {}".format(self.optimizer.lower()))
            # Set optimizer, default is Adam
            if self.optimizer.lower() == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters())
            elif self.optimizer.lower() == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01, momentum=0.9)
            elif self.optimizer.lower() == 'adadelta':
                optimizer = torch.optim.Adadelta(self.model.parameters())
            else:
                logger.critical("unknown optimizer, using default Adam")
                optimizer = torch.optim.Adam(self.model.parameters())

            for seg_idx, seg in enumerate(training_seg_list):
                show, s, _, label = seg
                e = s + len(label)
                # Load the segment of frames plus left and right context
                feat, _ = features_server.load(show,
                                               start=s - features_server.context[0],
                                               stop=e + features_server.context[1])

                # Cut the segment in batches of "batch_size" frames if possible
                for ii in range((feat.shape[0] - sum(features_server.context)) // batch_size):
                    data = ((feat[ii * batch_size:(ii + 1) * batch_size + sum(features_server.context), :] - self.input_mean) / self.input_std).T
                    data = data[None, ...]
                    lab = label[ii * batch_size:(ii + 1) * batch_size]
                    # Send data and label to the GPU
                    X = torch.from_numpy(data).type(torch.FloatTensor).to(device)
                    t = torch.from_numpy(lab).to(device)
                    optimizer.zero_grad()
                    lab_pred = self.forward(X)
                    #lab_pred = torch.t(self.forward(X)[0])
                    loss = self.criterion(lab_pred, t)
                    loss.backward()
                    optimizer.step()

                    accuracy += (torch.argmax(lab_pred.data, 1) == t).sum().item()
                    nbatch += 1
                    n += batch_size
                    running_loss += loss.item() / (batch_size * nbatch)
                    if nbatch % 200 == 199:
                        logger.critical("loss = {} | accuracy = {} ".format(running_loss,  accuracy / n) )

            logger.critical("Start Cross-Validation")
            optimizer.zero_grad()
            running_loss = accuracy = n = 0.0

            for ii, cv_segment in enumerate(cross_validation_seg_list):
                show, s, e, label = cv_segment
                e = s + len(label)
                t = torch.from_numpy(label.astype('long')).to(device)
                # Load the segment of frames plus left and right context
                feat, _ = features_server.load(show,
                                               start=s - features_server.context[0],
                                               stop=e + features_server.context[1])
                feat = (feat - self.input_mean) / self.input_std
                nfeat = feat.shape[0]
                feat = (feat.T)[None, ...]
                lab_pred = self.forward(torch.from_numpy(feat).type(torch.FloatTensor).to(device))
                #lab_pred = torch.t(self.forward(torch.from_numpy(feat).type(torch.FloatTensor).to(device))[0])
                loss = self.criterion(lab_pred, t)
                accuracy += (torch.argmax(lab_pred.data, 1) == t).sum().item()
                running_loss += loss.item()
                n += nfeat

            logger.critical("Cross Validation loss = {} | accuracy = {} ".format(running_loss / n, accuracy / n))

            # Save the current version of the network
            torch.save(self.model.to('cpu').state_dict(), output_file_name.format(ep))

            # Early stopping with very basic loss criteria
            #if last_cv_error >= accuracy / n:
            #    break








