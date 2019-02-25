# -*- coding: utf-8 -*-
"""
This package is a translation of a part of the BOSARIS toolkit.
The authors thank Niko Brummer and Agnitio for allowing them to 
translate this code and provide the community with efficient structures
and tools.

The BOSARIS Toolkit is a collection of functions and classes in Matlab
that can be used to calibrate, fuse and plot scores from speaker recognition
(or other fields in which scores are used to test the hypothesis that two
samples are from the same source) trials involving a model and a test segment.
The toolkit was written at the BOSARIS2010 workshop which took place at the
University of Technology in Brno, Czech Republic from 5 July to 6 August 2010.
See the User Guide (available on the toolkit website)1 for a discussion of the
theory behind the toolkit and descriptions of some of the algorithms used.

The BOSARIS toolkit in MATLAB can be downloaded from `the website 
<https://sites.google.com/site/bosaristoolkit/>`_.
"""


from sidekit.bosaris.idmap import IdMap
from sidekit.bosaris.ndx import Ndx
from sidekit.bosaris.plotwindow import PlotWindow
from sidekit.bosaris.key import Key
from sidekit.bosaris.scores import Scores
from sidekit.bosaris.detplot import DetPlot
from sidekit.bosaris.detplot import effective_prior
from sidekit.bosaris.detplot import logit_effective_prior
from sidekit.bosaris.detplot import fast_minDCF


__author__ = "Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brummer", "Edward de Villiers"]

