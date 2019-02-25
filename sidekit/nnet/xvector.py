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
Copyright 2014-2019 Yevhenii Prokopalo, Anthony Larcher


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
import random
import torch
from torch.autograd import Variable
import json
import subprocess
import resource
import scipy.linalg as la

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reS'


def GetListOfFiles(MainFolder):
    ListOfFiles = []
    for file in os.listdir(MainFolder):
        path = os.path.join(MainFolder, file)
        if not os.path.isdir(path):
            ListOfFiles.append(path)
        else:
            ListOfFiles += GetListOfFiles(path)
    return ListOfFiles


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


class Xtractor(torch.nn.Module):
    """

    """
    def __init__(self):
        """

        """
        super(Xtractor, self).__init__()
        self.conv0 = torch.nn.Conv1d(20, 512, 5)
        self.conv1 = torch.nn.Conv1d(512, 512, 3, dilation=2)
        self.conv2 = torch.nn.Conv1d(512, 512, 3, dilation=3)
        self.conv3 = torch.nn.Conv1d(512, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1500, 1)
        self.lin1 = torch.nn.Linear(3000, 512)
        self.lin2 = torch.nn.Linear(512, 512)
        self.lin3 = torch.nn.Linear(512, 1951)

        self.norm1 = torch.nn.BatchNorm1d(512)
        self.norm2 = torch.nn.BatchNorm1d(512)
        self.norm3 = torch.nn.BatchNorm1d(512)
        self.norm4 = torch.nn.BatchNorm1d(512)
        self.norm5 = torch.nn.BatchNorm1d(1500)
        self.norm7 = torch.nn.BatchNorm1d(512)

        self.pooling = torch.nn.AvgPool1d(186)
        self.rel = torch.nn.Softplus()


    def init_weights(self, SpeakersCount):
        """

        :param weights:
        :param bias:
        :return:
        """
        self.conv0.weight.data = torch.FloatTensor(numpy.random.rand(512, 20, 5) - 0.5)
        self.conv1.weight.data = torch.FloatTensor(numpy.random.rand(512, 512, 3) - 0.5)
        self.conv2.weight.data = torch.FloatTensor(numpy.random.rand(512, 512, 3) - 0.5)
        self.conv3.weight.data = torch.FloatTensor(numpy.random.rand(512, 512, 1) - 0.5)
        self.conv4.weight.data = torch.FloatTensor(numpy.random.rand(1500, 512, 1) - 0.5)
        self.lin1.weight.data = torch.FloatTensor(numpy.random.rand(512, 3000) - 0.5)
        self.lin2.weight.data = torch.FloatTensor(numpy.random.rand(512, 512) - 0.5)
        self.lin3.weight.data = torch.FloatTensor(numpy.random.rand(SpeakersCount, 512) - 0.5)

        self.conv0.bias.data = torch.FloatTensor(numpy.random.rand(512) - 0.5)
        self.conv1.bias.data = torch.FloatTensor(numpy.random.rand(512) - 0.5)
        self.conv2.bias.data = torch.FloatTensor(numpy.random.rand(512) - 0.5)
        self.conv3.bias.data = torch.FloatTensor(numpy.random.rand(512) - 0.5)
        self.conv4.bias.data = torch.FloatTensor(numpy.random.rand(1500) - 0.5)
        self.lin1.bias.data = torch.FloatTensor(numpy.random.rand(512) - 0.5)
        self.lin2.bias.data = torch.FloatTensor(numpy.random.rand(512) - 0.5)
        self.lin3.bias.data = torch.FloatTensor(numpy.random.rand(SpeakersCount) - 0.5)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        l1out = self.rel(self.conv0(x))
        l1norm = self.norm1(l1out)

        l1param = {"mean": self.norm1.running_mean.detach().cpu().numpy(),
                   "var": self.norm1.running_var.detach().cpu().numpy(),
                   "gamma": self.norm1.weight.detach().cpu().numpy(),
                   "beta": self.norm1.bias.detach().cpu().numpy(), "eps": self.norm1.eps}

        l2out = self.rel(self.conv1(l1norm))
        l2norm = self.norm2(l2out)

        l2param = {"mean": self.norm2.running_mean.detach().cpu().numpy(),
                   "var": self.norm2.running_var.detach().cpu().numpy(),
                   "gamma": self.norm2.weight.detach().cpu().numpy(),
                   "beta": self.norm2.bias.detach().cpu().numpy(), "eps": self.norm2.eps}

        l3out = self.rel(self.conv2(l2norm))
        l3norm = self.norm3(l3out)

        l3param = {"mean": self.norm3.running_mean.detach().cpu().numpy(),
                   "var": self.norm3.running_var.detach().cpu().numpy(),
                   "gamma": self.norm3.weight.detach().cpu().numpy(),
                   "beta": self.norm3.bias.detach().cpu().numpy(), "eps": self.norm3.eps}

        l4out = self.rel(self.conv3(l3norm))
        l4norm = self.norm4(l4out)

        l4param = {"mean": self.norm4.running_mean.detach().cpu().numpy(),
                   "var": self.norm4.running_var.detach().cpu().numpy(),
                   "gamma": self.norm4.weight.detach().cpu().numpy(),
                   "beta": self.norm4.bias.detach().cpu().numpy(), "eps": self.norm4.eps}

        l5out = self.rel(self.conv4(l4norm))
        l5norm = self.norm5(l5out)

        l5param = {"mean": self.norm5.running_mean.detach().cpu().numpy(),
                   "var": self.norm5.running_var.detach().cpu().numpy(),
                   "gamma": self.norm5.weight.detach().cpu().numpy(),
                   "beta": self.norm5.bias.detach().cpu().numpy(), "eps": self.norm5.eps}

        mean = torch.mean(l5norm, dim=2)
        std = torch.std(l5norm, dim=2)
        l6inp = torch.cat([mean, std], dim=1)

        l6out = self.rel(self.lin1(l6inp))
        l7out = self.rel(self.lin2(l6out))
        l7norm = self.norm7(l7out)

        l7param = {"mean": self.norm7.running_mean.detach().cpu().numpy(),
                   "var": self.norm7.running_var.detach().cpu().numpy(),
                   "gamma": self.norm7.weight.detach().cpu().numpy(),
                   "beta": self.norm7.bias.detach().cpu().numpy(), "eps": self.norm7.eps}

        l8out = self.rel(self.lin3(l7norm))
        result = torch.nn.functional.softmax(l8out, dim=1)

        params = {"l1": l1param, "l2": l2param, "l3": l3param, "l4": l4param, "l5": l5param, "l7": l7param}
        return result, params

    def LossFN(self, x, label):
        """

        :param x:
        :param lable:
        :return:
        """
        loss = - torch.trace(torch.mm(torch.log10(x), torch.t(label)))
        return loss

    def getWeights(self):
        """

        :return:
        """
        weights = []
        weights.append(self.conv0.weight.data.detach().cpu().numpy())
        weights.append(self.conv1.weight.data.detach().cpu().numpy())
        weights.append(self.conv2.weight.data.detach().cpu().numpy())
        weights.append(self.conv3.weight.data.detach().cpu().numpy())
        weights.append(self.conv4.weight.data.detach().cpu().numpy())
        weights.append(self.lin1.weight.data.detach().cpu().numpy())
        weights.append(self.lin2.weight.data.detach().cpu().numpy())
        weights.append(self.lin3.weight.data.detach().cpu().numpy())

        bias = []
        bias.append(self.conv0.bias.data.detach().cpu().numpy())
        bias.append(self.conv1.bias.data.detach().cpu().numpy())
        bias.append(self.conv2.bias.data.detach().cpu().numpy())
        bias.append(self.conv3.bias.data.detach().cpu().numpy())
        bias.append(self.conv4.bias.data.detach().cpu().numpy())
        bias.append(self.lin1.bias.data.detach().cpu().numpy())
        bias.append(self.lin2.bias.data.detach().cpu().numpy())
        bias.append(self.lin3.bias.data.detach().cpu().numpy())
        return weights, bias


def WorkerOpt(weights, bias, ListOfFiles, alpha, return_dict, WorkerID, NormParam):
    """

    :param weights:
    :param bias:
    :param ListOfFiles:
    :param alpha:
    :param return_dict:
    :param WorkerID:
    :param NormParam:
    :return:
    """
    manager = multiprocessing.Manager()
    LoaderDict = manager.dict()

    losslist = []

    Model = Xtractor()
    Model.init_weights(weights, bias)

    Model.cuda(WorkerID)
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)

    loading = multiprocessing.Process(target=Loader, args=(ListOfFiles[0], LoaderDict, WorkerID))
    loading.start()
    loading.join()
    batch = LoaderDict[WorkerID]

    for i in range(len(ListOfFiles)):

        if i != (len(ListOfFiles) - 1):
            loading = multiprocessing.Process(target=Loader, args=(ListOfFiles[i + 1], LoaderDict, WorkerID))
            loading.start()

        optimizer.zero_grad()
        output, params = Model.forward(torch.FloatTensor(batch["data"]).cuda(WorkerID))
        NormParam = MoveAvgParam(NormParam, params)
        # print(NormParam)
        loss = Model.LossFN(output, torch.FloatTensor(batch["lable"]).cuda(WorkerID))
        loss.backward()
        optimizer.step()
        losslist.append(loss.detach().cpu().numpy().tolist())

        if i != (len(ListOfFiles) - 1):
            loading.join()
            batch = LoaderDict[WorkerID]

    weights, bias = Model.getWeights()
    return_dict[WorkerID] = {"weights": weights, "bias": bias, "loss": losslist, "NormParam": NormParam}


def MoveAvgParam(NormParam, params):
    """

    :param NormParam:
    :param params:
    :return:
    """
    listofkeys = list(NormParam.keys())
    if len(listofkeys) < 6:
        return params
    else:
        for i in range(len(listofkeys)):
            listofkeys2 = list(NormParam[listofkeys[i]].keys())
            for j in range(len(listofkeys2)):
                NormParam[listofkeys[i]][listofkeys2[j]] = (NormParam[listofkeys[i]][listofkeys2[j]] +
                                                            params[listofkeys[i]][listofkeys2[j]]) / 2
        return NormParam


def GetAvgNormParam(ListOfParams):
    """

    :param ListOfParams:
    :return:
    """
    listofkeys = list(ListOfParams[0].keys())
    listofkeys2 = list(ListOfParams[0][listofkeys[0]].keys())
    for k in range(1, len(ListOfParams)):
        for i in range(len(listofkeys)):
            for j in range(len(listofkeys2)):
                ListOfParams[0][listofkeys[i]][listofkeys2[j]] = ListOfParams[0][listofkeys[i]][listofkeys2[j]] + \
                                                                 ListOfParams[k][listofkeys[i]][listofkeys2[j]]
    for i in range(len(listofkeys)):
        for j in range(len(listofkeys2)):
            ListOfParams[0][listofkeys[i]][listofkeys2[j]] = ListOfParams[0][listofkeys[i]][listofkeys2[j]] / len(
                ListOfParams)
    return ListOfParams[0]


def SaveNormParam(NormParam, epoque):
    """

    :param NormParam:
    :param epoque:
    :return:
    """
    listofkeys = list(NormParam.keys())
    listofkeys2 = list(NormParam[listofkeys[0]].keys())
    for i in range(len(listofkeys)):
        for j in range(len(listofkeys2) - 1):
            NormParam[listofkeys[i]][listofkeys2[j]] = NormParam[listofkeys[i]][listofkeys2[j]].tolist()
    with open('/lium/raid01_b/prokopalo/ME3/NormParams_epoque_' + str(epoque) + '.json', 'w') as fp:
        json.dump(NormParam, fp)


def Loader(File, LoaderDict, WorkerID):
    """

    :param File:
    :param LoaderDict:
    :param WorkerID:
    :return:
    """
    with open(File, 'r') as fp:
        batch = (json.load(fp))
    LoaderDict[WorkerID] = batch



def WCWorker(deltas, biasdeltas, wc_dict, WorkerID):
    """

    :param deltas:
    :param biasdeltas:
    :param wc_dict:
    :param WorkerID:
    :return:
    """
    weights = deltas[0]
    for i in range(1, len(deltas)):
        weights = weights + deltas[i]
    bias = biasdeltas[0]
    for i in range(1, len(biasdeltas)):
        bias = bias + biasdeltas[i]

    weights = weights / len(deltas)
    bias = bias / len(biasdeltas)
    wc_dict[WorkerID] = {"weights": weights, "bias": bias}


def SaveModel(path, weights, bias, index, epoque):
    """

    :param weights:
    :param bias:
    :param index:
    :param epoque:
    :return:
    """
    for i in range(len(weights)):
        weights[i] = weights[i].tolist()
        bias[i] = bias[i].tolist()
    result = {"weights": weights, "bias": bias}
    with open(path + '/Vopt_epoque_' + str(epoque) + '_set_' + str(index) + '.json',
              'w') as fp:
        json.dump(result, fp)

def train_xtractor(spk_count, file_list, gpu_nb=4, alpha=0.001, epoch_nb=3):
    """

    :param spk_count:
    :return:
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    wc_dict = manager.dict()
    losslist = [[], [], [], []]

    for epoque in range(epoch_nb):

        random.shuffle(file_list)
        random.shuffle(file_list)
        StructList = []
        NormParam = {}
        for i in range(150):
            tmp = []
            for j in range(gpu_nb):
                tmp.append(file_list[(100 * (gpu_nb * i + j)):(100 * (gpu_nb * i + j + 1))])
            StructList.append(tmp)
        finalSet = file_list[60000:]

        savecounter = 0
        for i in range(150):
            savecounter += 1

            proc = []
            for j in range(gpu_nb):
                proc.append(multiprocessing.Process(target=WorkerOpt, args=(
                weights, bias, StructList[i][j], alpha, return_dict, j, NormParam)))
                proc[j].start()
            for j in range(gpu_nb):
                proc[j].join()

            proc = []

            for j in range(2 * gpu_nb):
                proc.append(multiprocessing.Process(target=WCWorker,
                                                    args=([return_dict[0]["weights"][j],
                                                           return_dict[1]["weights"][j],
                                                           return_dict[2]["weights"][j],
                                                           return_dict[3]["weights"][j]],
                                                          [return_dict[0]["bias"][j],
                                                           return_dict[1]["bias"][j],
                                                           return_dict[2]["bias"][j],
                                                           return_dict[3]["bias"][j]], wc_dict, j)))
                proc[j].start()
            for j in range(2 * gpu_nb):
                proc[j].join()

            for j in range(2 * gpu_nb):
                weights[j] = wc_dict[j]["weights"]
                bias[j] = wc_dict[j]["bias"]

            NormParam = GetAvgNormParam([return_dict[0]["NormParam"],
                                         return_dict[1]["NormParam"],
                                         return_dict[2]["NormParam"],
                                         return_dict[3]["NormParam"]])

            for j in range(gpu_nb):
                losslist[j] = losslist[j] + return_dict[j]["loss"]

            if savecounter == 10:
                savecounter = 0
                SaveModel(weights, bias, i, epoque)
                for j in range(gpu_nb):
                    with open('/lium/raid01_b/prokopalo/ME3/Loss/loss_epoque_' + str(epoque) + '_set_' + str(
                            i) + '_gpu_' + str(j) + '.json', 'w') as fp:
                        json.dump(losslist[j], fp)

        proc = multiprocessing.Process(target=WorkerOpt,
                                       args=(weights, bias, finalSet, alpha, return_dict, 0, NormParam))
        proc.start()
        proc.join()

        weights = return_dict[0]["weights"]
        bias = return_dict[0]["bias"]
        SaveModel(weights, bias, "final", epoque)
        with open('/lium/raid01_b/prokopalo/ME3/Loss/final_loss_epoque_' + str(epoque) + '.json', 'w') as fp:
            json.dump(return_dict[0]["loss"], fp)

        SaveNormParam(return_dict[0]["NormParam"], epoque)


