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
"""

from ctypes import *
from ctypes.util import find_library
import logging
import numpy
import os
import sys
import importlib


# Read environment variable if it exists
SIDEKIT_CONFIG={"libsvm":True,
                "mpi":False
                }

if 'SIDEKIT' in os.environ:
    for cfg in os.environ['SIDEKIT'].split(","):
        k, val = cfg.split("=")
        if k == "libsvm":
           if val == "false":
                SIDEKIT_CONFIG["libsvm"] = False 
        elif k == "mpi":
            if val == "true":
               SIDEKIT_CONFIG["mpi"] = True 

PARALLEL_MODULE = 'multiprocessing'  # can be , threading, multiprocessing MPI is planned in the future
PARAM_TYPE = numpy.float32
STAT_TYPE = numpy.float64  # can be numpy.float32 to speed up the computation but can lead to numerical issuess

# Import bosaris-like classes
from sidekit.bosaris import IdMap
from sidekit.bosaris import Ndx
from sidekit.bosaris import Key
from sidekit.bosaris import Scores
from sidekit.bosaris import DetPlot
from sidekit.bosaris import effective_prior
from sidekit.bosaris import logit_effective_prior
from sidekit.bosaris import fast_minDCF

# Import classes
from sidekit.features_extractor import FeaturesExtractor
from sidekit.features_server import FeaturesServer
from sidekit.mixture import Mixture
from sidekit.statserver import StatServer
from sidekit.factor_analyser import FactorAnalyser

from sidekit.frontend.io import write_pcm
from sidekit.frontend.io import read_pcm
from sidekit.frontend.io import pcmu2lin
from sidekit.frontend.io import read_sph
from sidekit.frontend.io import write_label
from sidekit.frontend.io import read_label
from sidekit.frontend.io import read_spro4
from sidekit.frontend.io import read_audio
from sidekit.frontend.io import write_spro4
from sidekit.frontend.io import read_htk
from sidekit.frontend.io import write_htk

from sidekit.frontend.vad import vad_energy
from sidekit.frontend.vad import vad_snr
from sidekit.frontend.vad import label_fusion
from sidekit.frontend.vad import speech_enhancement


from sidekit.frontend.normfeat import cms
from sidekit.frontend.normfeat import cmvn
from sidekit.frontend.normfeat import stg
from sidekit.frontend.normfeat import rasta_filt


from sidekit.frontend.features import compute_delta
from sidekit.frontend.features import framing
from sidekit.frontend.features import pre_emphasis
from sidekit.frontend.features import trfbank
from sidekit.frontend.features import mel_filter_bank
from sidekit.frontend.features import mfcc
from sidekit.frontend.features import pca_dct
from sidekit.frontend.features import shifted_delta_cepstral

from sidekit.nnet import FForwardNetwork

from sidekit.iv_scoring import cosine_scoring
from sidekit.iv_scoring import mahalanobis_scoring
from sidekit.iv_scoring import two_covariance_scoring
from sidekit.iv_scoring import PLDA_scoring

from sidekit.gmm_scoring import gmm_scoring 

from sidekit.jfa_scoring import jfa_scoring


from sidekit.sv_utils import clean_stat_server

libsvm_loaded = False
if SIDEKIT_CONFIG["libsvm"]:
    try:
        dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libsvm')
        if sys.platform == 'win32':
            libsvm = CDLL(os.path.join(dirname, r'libsvm.dll'))
            libsvm_loaded = True
        else:
            libsvm = CDLL(os.path.join(dirname, 'libsvm.so.2'))
            libsvm_loaded = True
    except:
        # For unix the prefix 'lib' is not considered.
        if find_library('svm'):
            libsvm = CDLL(find_library('svm'))
            libsvm_loaded = True
        elif find_library('libsvm'):
            libsvm = CDLL(find_library('libsvm'))
            libsvm_loaded = True
        else:
            libsvm_loaded = False
            logging.warning('WARNNG: libsvm is not installed, please refer to the' +
                            ' documentation if you intend to use SVM classifiers')

if libsvm_loaded:
    from sidekit.libsvm import *
    from sidekit.svm_scoring import *
    from sidekit.svm_training import *


if SIDEKIT_CONFIG["mpi"]:
    found_mpi4py = importlib.find_loader('mpi4py') is not None
    if found_mpi4py:
        from sidekit.sidekit_mpi import EM_split, total_variability, extract_ivector
        print("Import MPI")
        

__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2019 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__version__="1.3.2"

# __all__ = ["io",
#            "vad",
#            "normfeat",
#            "features"
#            ]
