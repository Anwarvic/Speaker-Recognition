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

:mod:`svm_training` provides utilities to train Support Vector Machines
to perform speaker verification.
"""
import numpy
import os
import logging
from sidekit.libsvm.svmutil import svm_problem, svm_parameter, svm_train
import multiprocessing
import sidekit.sv_utils


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def svm_training_singleThread(K, msn, bsn, svm_dir, background_sv, models, enroll_sv):
    """Train Suport Vector Machine classifiers for two classes task 
    (as implemented for nowbut miht change in the future to include multi-class
    classification)
    
    :param K: pre-computed part of the Gram matrix
    :param msn: maximum number of sessions to train a SVM
    :param bsn: number of session used as background impostors
    :param svm_dir: directory where to store the SVM models
    :param background_sv: StatServer of super-vectors for background impostors. All
          super-vectors are used without selection
    :param models: list of models to train. The models must be included in the 
          enroll_sv StatServer
    :param enroll_sv: StatServer of super-vectors used for the target models
    """
    gram = numpy.zeros((bsn + msn, bsn + msn))
    gram[:bsn, :bsn] = K
    # labels of the target examples are set to 1
    # labels of the impostor vectors are set to 2
    K_label = (2 * numpy.ones(bsn, 'int')).tolist() + numpy.ones(msn, 'int').tolist()

    for model in models:
        logging.info('Train SVM model for %s', model)    
        # Compute the part of the Kernel which depends on the enrollment data
        csn = enroll_sv.get_model_segments(model).shape[0]
        X = numpy.vstack((background_sv.stat1, enroll_sv.get_model_stat1(model)))
        gram[:bsn + csn, bsn:bsn + csn] = numpy.dot(X, enroll_sv.get_model_stat1(model).transpose())
        gram[bsn:bsn + csn, :bsn] = gram[:bsn, bsn:bsn + csn].transpose()

        # train the SVM for the current model (where libsvm is used)
        Kernel = numpy.zeros((gram.shape[0], gram.shape[1] + 1)).tolist()
        for i in range(gram.shape[0]):
            Kernel[i][0] = int(i + 1)
            Kernel[i][1:] = gram[i, ]

        # isKernel=True must be set for precomputer kernel
        # Precomputed kernel data (-t 4)
        prob = svm_problem(K_label, Kernel, isKernel=True)
        c = 1 / numpy.mean(numpy.diag(gram))
        param = svm_parameter('-t 4 -c {}'.format(c))
        svm = svm_train(prob, param)
        # Compute the weights
        w = -numpy.dot(X[numpy.array(svm.get_sv_indices()) - 1, ].transpose(), numpy.array(svm.get_sv_coef()))
        bsvm = svm.rho[0]
        svmFileName = os.path.join(svm_dir, model + '.svm')
        sidekit.sv_utils.save_svm(svmFileName, w, bsvm)


def svm_training(svmDir, background_sv, enroll_sv, num_thread=1):
    """Train Suport Vector Machine classifiers for two classes task 
    (as implemented for nowbut miht change in the future to include multi-class
    classification)
    Training is parallelized on multiple threads.
    
    :param svmDir: directory where to store the SVM models
    :param background_sv: StatServer of super-vectors for background impostors. All
          super-vectors are used without selection
    :param enroll_sv: StatServer of super-vectors used for the target models
    :param num_thread: number of thread to launch in parallel
    """
    assert isinstance(background_sv, sidekit.StatServer), 'Second parameter has to be a StatServer'
    assert isinstance(enroll_sv, sidekit.StatServer), 'Third parameter has to be a StatServer'

    # The effective Kernel is initialize for the case of multi-session
    # by considering the maximum number of sessions per speaker.
    # For the SVM training, only a subpart of the kernel is used, accordingly
    # to the number of sessions of the current speaker
    K = background_sv.precompute_svm_kernel_stat1()
    msn = max([enroll_sv.modelset.tolist().count(a) for a in enroll_sv.modelset.tolist()])
    bsn = K.shape[0]

    # Split the list of unique model names
    listOfModels = numpy.array_split(numpy.unique(enroll_sv.modelset), num_thread)
    
    # Process each sub-list of models in a separate thread
    jobs = []
    for idx, models in enumerate(listOfModels):
        p = multiprocessing.Process(target=svm_training_singleThread,
                                    args=(K, msn, bsn, svmDir, background_sv, models, enroll_sv))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()


