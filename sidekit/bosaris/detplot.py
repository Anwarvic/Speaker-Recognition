# -*- coding: utf-8 -*-

# This package is a translation of a part of the BOSARIS toolkit.
# The authors thank Niko Brummer and Agnitio for allowing them to
# translate this code and provide the community with efficient structures
# and tools.
#
# The BOSARIS Toolkit is a collection of functions and classes in Matlab
# that can be used to calibrate, fuse and plot scores from speaker recognition
# (or other fields in which scores are used to test the hypothesis that two
# samples are from the same source) trials involving a model and a test segment.
# The toolkit was written at the BOSARIS2010 workshop which took place at the
# University of Technology in Brno, Czech Republic from 5 July to 6 August 2010.
# See the User Guide (available on the toolkit website)1 for a discussion of the
# theory behind the toolkit and descriptions of some of the algorithms used.
#
# The BOSARIS toolkit in MATLAB can be downloaded from `the website
# <https://sites.google.com/site/bosaristoolkit/>`_.

"""
This is the 'detplot' module

    This module supplies tools for ploting DET curve.
    It includes a class for creating a plot for displaying detection performance
    with the axes scaled and labelled so that a normal Gaussian
    distribution will plot as a straight line.

    The y axis represents the miss probability.
    The x axis represents the false alarm probability.

    This file is a translation of the BOSARIS toolkit.
    For more information, refers to the license provided with this package.
"""
import numpy
import matplotlib
import os

if "DISPLAY" not in os.environ:
    matplotlib.use('PDF', warn=False, force=True)
import matplotlib.pyplot as mpl
import scipy
from collections import namedtuple
import logging

from sidekit.bosaris import PlotWindow
from sidekit.bosaris import Scores
from sidekit.bosaris import Key


__author__ = "Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brummer", "Edward de Villiers"]


colorStyle = [
    ((0, 0, 0), '-', 2),  # black
    ((0, 0, 1.0), '--', 2),  # blue
    ((0.8, 0.0, 0.0), '-.', 2),  # red
    ((0, 0.6, 0.0), ':', 2),  # green
    ((0.5, 0.0, 0.5), '-', 2),  # magenta
    ((0.3, 0.3, 0.0), '--', 2),  # orange
    ((0, 0, 0), ':', 2),  # black
    ((0, 0, 1.0), ':', 2),  # blue
    ((0.8, 0.0, 0.0), ':', 2),  # red
    ((0, 0.6, 0.0), '-', 2),  # green
    ((0.5, 0.0, 0.5), '-.', 2),  # magenta
    ((0.3, 0.3, 0.0), '-', 2),  # orange
    ]

grayStyle = [
    ((0, 0, 0), '-', 2),  # black
    ((0, 0, 0), '--', 2),  # black
    ((0, 0, 0), '-.', 2),  # black
    ((0, 0, 0), ':', 2),  # black
    ((0.3, 0.3, 0.3), '-', 2),  # gray
    ((0.3, 0.3, 0.3), '--', 2),  # gray
    ((0.3, 0.3, 0.3), '-.', 2),  # gray
    ((0.3, 0.3, 0.3), ':', 2),  # gray
    ((0.6, 0.6, 0.6), '-', 2),  # lighter gray
    ((0.6, 0.6, 0.6), '--', 2),  # lighter gray
    ((0.6, 0.6, 0.6), '-.', 2),  # lighter gray
    ((0.6, 0.6, 0.6), ':', 2),  # lighter gray
    ]

Box = namedtuple("Box", "left right top bottom")


def effective_prior(Ptar, cmiss, cfa):
    """This function adjusts a given prior probability of target p_targ,
    to incorporate the effects of a cost of miss,
    cmiss, and a cost of false-alarm, cfa.
    In particular note:
    EFFECTIVE_PRIOR(EFFECTIVE_PRIOR(p,cmiss,cfa),1,1)
            = EFFECTIVE_PRIOR(p,cfa,cmiss)

    The effective prior for the NIST SRE detection cost fuction,
    with p_targ = 0.01, cmiss = 10, cfa = 1 is therefore:
    EFFECTIVE_PRIOR(0.01,10,1) = 0.0917

    :param Ptar: is the probability of a target trial
    :param cmiss: is the cost of a miss
    :param cfa: is the cost of a false alarm

    :return: a prior
    """
    p = Ptar * cmiss / (Ptar * cmiss + (1 - Ptar) * cfa)
    return p


def logit_effective_prior(Ptar, cmiss, cfa):
    """This function adjusts a given prior probability of target p_targ,
    to incorporate the effects of a cost of miss,
    cmiss, and a cost of false-alarm, cfa.
    In particular note:
    EFFECTIVE_PRIOR(EFFECTIVE_PRIOR(p,cmiss,cfa),1,1)
            = EFFECTIVE_PRIOR(p,cfa,cmiss)

    The effective prior for the NIST SRE detection cost fuction,
    with p_targ = 0.01, cmiss = 10, cfa = 1 is therefore:
    EFFECTIVE_PRIOR(0.01,10,1) = 0.0917

    :param Ptar: is the probability of a target trial
    :param cmiss: is the cost of a miss
    :param cfa: is the cost of a false alarm

    :return: a prior
    """
    p = Ptar * cmiss / (Ptar * cmiss + (1 - Ptar) * cfa)
    return __logit__(p)


def __probit__(p):
    """Map from [0,1] to [-inf,inf] as used to make DET out of a ROC
    
    :param p: the value to map

    :return: probit(input)
    """
    y = numpy.sqrt(2) * scipy.special.erfinv(2 * p - 1)
    return y


def __logit__(p):
    """logit function.
    This is a one-to-one mapping from probability to log-odds.
    i.e. it maps the interval (0,1) to the real line.
    The inverse function is given by SIGMOID.

    log_odds = logit(p) = log(p/(1-p))

    :param p: the input value

    :return: logit(input)
    """
    p = numpy.array(p)
    lp = numpy.zeros(p.shape)
    f0 = p == 0
    f1 = p == 1
    f = (p > 0) & (p < 1)

    if lp.shape == ():
        if f:
            lp = numpy.log(p / (1 - p))
        elif f0:
            lp = -numpy.inf
        elif f1:
            lp = numpy.inf
    else:
        lp[f] = numpy.log(p[f] / (1 - p[f]))
        lp[f0] = -numpy.inf
        lp[f1] = numpy.inf
    return lp


def __DETsort__(x, col=''):
    """DETsort Sort rows, the first in ascending, the remaining in descending
    thereby postponing the false alarms on like scores.
    based on SORTROWS
    
    :param x: the array to sort
    :param col: not used here

    :return: a sorted vector of scores
    """
    assert x.ndim > 1, 'x must be a 2D matrix'
    if col == '':
        list(range(1, x.shape[1]))

    ndx = numpy.arange(x.shape[0])

    # sort 2nd column ascending
    ind = numpy.argsort(x[:, 1], kind='mergesort')
    ndx = ndx[ind]

    # reverse to descending order
    ndx = ndx[::-1]

    # now sort first column ascending
    ind = numpy.argsort(x[ndx, 0], kind='mergesort')

    ndx = ndx[ind]
    sort_scores = x[ndx, :]
    return sort_scores


def __compute_roc__(true_scores, false_scores):
    """Computes the (observed) miss/false_alarm probabilities
    for a set of detection output scores.
    
    true_scores (false_scores) are detection output scores for a set of
    detection trials, given that the target hypothesis is true (false).
    (By convention, the more positive the score,
    the more likely is the target hypothesis.)
    
    :param true_scores: a 1D array of target scores
    :param false_scores: a 1D array of non-target scores

    :return: a tuple of two vectors, Pmiss,Pfa
    """
    num_true = true_scores.shape[0]
    num_false = false_scores.shape[0]
    assert num_true > 0, "Vector of target scores is empty"
    assert num_false > 0, "Vector of nontarget scores is empty"

    total = num_true + num_false

    Pmiss = numpy.zeros((total + 1))
    Pfa = numpy.zeros((total + 1))

    scores = numpy.zeros((total, 2))
    scores[:num_false, 0] = false_scores
    scores[:num_false, 1] = 0
    scores[num_false:, 0] = true_scores
    scores[num_false:, 1] = 1

    scores = __DETsort__(scores)

    sumtrue = numpy.cumsum(scores[:, 1], axis=0)
    sumfalse = num_false - (numpy.arange(1, total + 1) - sumtrue)

    Pmiss[0] = 0
    Pfa[0] = 1
    Pmiss[1:] = sumtrue / num_true
    Pfa[1:] = sumfalse / num_false
    return Pmiss, Pfa


def __filter_roc__(pm, pfa):
    """Removes redundant points from the sequence of points (pfa,pm) so
    that plotting an ROC or DET curve will be faster.  The output ROC
    curve will be identical to the one plotted from the input
    vectors.  All points internal to straight (horizontal or
    vertical) sections on the ROC curve are removed i.e. only the
    points at the start and end of line segments in the curve are
    retained.  Since the plotting code draws straight lines between
    points, the resulting plot will be the same as the original.
    
    :param pm: the vector of miss probabilities of the ROC Convex
    :param pfa: the vector of false-alarm probabilities of the ROC Convex

    :return: a tuple of two vectors, Pmiss, Pfa
    """
    out = 0
    new_pm = [pm[0]]
    new_pfa = [pfa[0]]

    for i in range(1, pm.shape[0]):
        if (pm[i] == new_pm[out]) | (pfa[i] == new_pfa[out]):
            pass
        else:
            # save previous point, because it is the last point before the
            # change.  On the next iteration, the current point will be saved.
            out += 1
            new_pm.append(pm[i - 1])
            new_pfa.append(pfa[i - 1])

    out += 1
    new_pm.append(pm[-1])
    new_pfa.append(pfa[-1])
    pm = numpy.array(new_pm)
    pfa = numpy.array(new_pfa)
    return pm, pfa


def pavx(y):
    """PAV: Pool Adjacent Violators algorithm.
    Non-paramtetric optimization subject to monotonicity.

    ghat = pav(y)
    fits a vector ghat with nondecreasing components to the
    data vector y such that sum((y - ghat).^2) is minimal.
    (Pool-adjacent-violators algorithm).

    optional outputs:
            width: width of pav bins, from left to right
                    (the number of bins is data dependent)
            height: corresponding heights of bins (in increasing order)

    Author: This code is a simplified version of the 'IsoMeans.m' code
    made available by Lutz Duembgen at:
    http://www.imsv.unibe.ch/~duembgen/software

    :param y: input value
    """
    assert y.ndim == 1, 'Argument should be a 1-D array'
    assert y.shape[0] > 0, 'Input array is empty'
    n = y.shape[0]

    index = numpy.zeros(n)
    length = numpy.zeros(n)

    # An interval of indices is represented by its left endpoint
    # ("index") and its length "length"
    ghat = numpy.zeros(n)

    ci = 0
    index[ci] = 0
    length[ci] = 1
    ghat[ci] = y[0]

    # ci is the number of the interval considered currently.
    # ghat(ci) is the mean of y-values within this interval.
    for j in range(1, n):
        # a new index interval, {j}, is created:
        ci += 1
        index[ci] = j
        length[ci] = 1
        ghat[ci] = y[j]
        while (ci >= 1) & (ghat[numpy.max(ci - 1, 0)] >= ghat[ci]):
            # pool adjacent violators:
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = ghat[:ci + 1]
    width = length[:ci + 1]

    # Now define ghat for all indices:
    while n >= 0:
        for j in range(int(index[ci]), int(n)):
            ghat[j] = ghat[ci]

        n = index[ci] - 1
        ci -= 1

    return ghat, width, height


def rocch2eer(pmiss, pfa):
    """Calculates the equal error rate (eer) from pmiss and pfa vectors.  
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.  
    Use rocch.m to convert target and non-target scores to pmiss and
    pfa values.

    :param pmiss: the vector of miss probabilities
    :param pfa: the vector of false-alarm probabilities

    :return: the equal error rate
    """
    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]

        # xx and yy should be sorted:
        assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), \
            'pmiss and pfa have to be sorted'

        XY = numpy.column_stack((xx, yy))
        dd = numpy.dot(numpy.array([1, -1]), XY)
        if numpy.min(numpy.abs(dd)) == 0:
            eerseg = 0
        else:
            # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
            # when xx(i),yy(i) is on the line.
            seg = numpy.linalg.solve(XY, numpy.array([[1], [1]]))
            # candidate for EER, eer is highest candidate
            eerseg = 1 / (numpy.sum(seg))

        eer = max([eer, eerseg])
    return eer


def rocch(tar_scores, nontar_scores):
    """ROCCH: ROC Convex Hull.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.

    For a demonstration that plots ROCCH against ROC for a few cases, just
    type 'rocch' at the MATLAB command line.

    :param tar_scores: vector of target scores
    :param nontar_scores: vector of non-target scores

    :return: a tupple of two vectors: Pmiss, Pfa 
    """
    Nt = tar_scores.shape[0]
    Nn = nontar_scores.shape[0]
    N = Nt + Nn
    scores = numpy.concatenate((tar_scores, nontar_scores))
    # Pideal is the ideal, but non-monotonic posterior
    Pideal = numpy.concatenate((numpy.ones(Nt), numpy.zeros(Nn)))
    #
    # It is important here that scores that are the same
    # (i.e. already in order) should NOT be swapped.rb
    perturb = numpy.argsort(scores, kind='mergesort')
    #
    Pideal = Pideal[perturb]
    Popt, width, foo = pavx(Pideal)
    #
    nbins = width.shape[0]
    pmiss = numpy.zeros(nbins + 1)
    pfa = numpy.zeros(nbins + 1)
    #
    # threshold leftmost: accept everything, miss nothing
    left = 0    # 0 scores to left of threshold
    fa = Nn
    miss = 0
    #
    for i in range(nbins):
        pmiss[i] = miss / Nt
        pfa[i] = fa / Nn
        left = int(left + width[i])
        miss = numpy.sum(Pideal[:left])
        fa = N - left - numpy.sum(Pideal[left:])
    #
    pmiss[nbins] = miss / Nt
    pfa[nbins] = fa / Nn
    #
    return pmiss, pfa


def sigmoid(log_odds):
    """SIGMOID: Inverse of the logit function.
    This is a one-to-one mapping from log odds to probability.
    i.e. it maps the real line to the interval (0,1).

    p = sigmoid(log_odds)

    :param log_odds: the input value

    :return: sigmoid(input)
    """
    p = 1 / (1 + numpy.exp(-log_odds))
    return p


def fast_minDCF(tar, non, plo, normalize=False):
    """Compute the minimum COST for given target and non-target scores
    Note that minDCF is parametrized by plo:
    
        minDCF(Ptar) = min_t Ptar * Pmiss(t) + (1-Ptar) * Pfa(t) 
    
    where t is the adjustable decision threshold and:

        Ptar = sigmoid(plo) = 1./(1+exp(-plo))

    If normalize == true, then the returned value is:

        minDCF(Ptar) / min(Ptar,1-Ptar).

    Pmiss: a vector with one value for every element of plo.
    This is Pmiss(tmin), where tmin is the minimizing threshold
    for minDCF, at every value of plo. Pmiss is not altered by
    parameter 'normalize'.

    Pfa: a vector with one value for every element of plo.
    This is Pfa(tmin), where tmin is the minimizing threshold for
    minDCF, at every value of plo. Pfa is not altered by
    parameter 'normalize'.

    Note, for the un-normalized case:

        minDCF(plo) = sigmoid(plo).*Pfa(plo) + sigmoid(-plo).*Pmiss(plo)
    
    :param tar: vector of target scores
    :param non: vector of non-target scores
    :param plo: vector of prior-log-odds: plo = logit(Ptar) = log(Ptar) - log(1-Ptar)
    :param normalize: if true, return normalized minDCF, else un-normalized (optional, default = false)

    :return: the minDCF value
    :return: the miss probability for this point 
    :return: the false-alarm probability for this point
    :return: the precision-recall break-even point: Where #FA == #miss
    :return the equal error rate 
    """
    Pmiss, Pfa = rocch(tar, non)
    Nmiss = Pmiss * tar.shape[0]
    Nfa = Pfa * non.shape[0]
    prbep = rocch2eer(Nmiss, Nfa)
    eer = rocch2eer(Pmiss, Pfa)

    Ptar = sigmoid(plo)
    Pnon = sigmoid(-plo)
    cdet = numpy.dot(numpy.array([[Ptar, Pnon]]), numpy.vstack((Pmiss, Pfa)))
    ii = numpy.argmin(cdet, axis=1)
    minDCF = cdet[0, ii][0]

    Pmiss = Pmiss[ii]
    Pfa = Pfa[ii]

    if normalize:
        minDCF = minDCF / min([Ptar, Pnon])

    return minDCF, Pmiss[0], Pfa[0], prbep, eer


def plotseg(xx, yy, box, dps):
    """Prepare the plotting of a curve.
    :param xx:
    :param yy:
    :param box:
    :param dps:
    """
    assert ((xx[1] <= xx[0]) & (yy[0] <= yy[1])), 'xx and yy should be sorted'

    XY = numpy.column_stack((xx, yy))
    dd = numpy.dot(numpy.array([1, -1]), XY)
    if numpy.min(abs(dd)) == 0:
        eer = 0
    else:
        # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
        # when xx(i),yy(i) is on the line.
        seg = numpy.linalg.solve(XY, numpy.array([[1], [1]]))
        # candidate for EER, eer is highest candidate
        eer = 1.0 / numpy.sum(seg)

    # segment completely outside of box
    if (xx[0] < box.left) | (xx[1] > box.right) | (yy[1] < box.bottom) | (yy[0] > box.top):
        x = numpy.array([])
        y = numpy.array([])
    else:
        if xx[1] < box.left:
            xx[1] = box.left
            yy[1] = (1 - seg[0] * box.left) / seg[1]

        if xx[0] > box.right:
            xx[0] = box.right
            yy[0] = (1 - seg[0] * box.right) / seg[1]

        if yy[0] < box.bottom:
            yy[0] = box.bottom
            xx[0] = (1 - seg[1] * box.bottom) / seg[0]

        if yy[1] > box.top:
            yy[1] = box.top
            xx[1] = (1 - seg[1] * box.top) / seg[0]

        dx = xx[1] - xx[0]
        xdots = xx[0] + dx * numpy.arange(dps + 1) / dps
        ydots = (1 - seg[0] * xdots) / seg[1]
        x = __probit__(xdots)
        y = __probit__(ydots)

    return x, y, eer


def rocchdet(tar, non,
             dcfweights=numpy.array([]),
             pfa_min=5e-4,
             pfa_max=0.5,
             pmiss_min=5e-4,
             pmiss_max=0.5,
             dps=100,
             normalize=False):
    """ROCCHDET: Computes ROC Convex Hull and then maps that to the DET axes.
    The DET-curve is infinite, non-trivial limits (away from 0 and 1)
    are mandatory.
    
    :param tar: vector of target scores
    :param non: vector of non-target scores
    :param dcfweights: 2-vector, such that: DCF = [pmiss,pfa]*dcfweights(:)  (Optional, provide only if mindcf is
    desired, otherwise omit or use []
    :param pfa_min: limit of DET-curve rectangle. Default is 0.0005
    :param pfa_max: limit of DET-curve rectangle. Default is 0.5
    :param pmiss_min: limit of DET-curve rectangle. Default is 0.0005
    :param pmiss_max: limits of DET-curve rectangle.  Default is 0.5
    :param dps: number of returned (x,y) dots (arranged in a curve) in DET space, for every straight line-segment
    (edge) of the ROC Convex Hull. Default is 100.
    :param normalize: normalize the curve

    :return: probit(Pfa)
    :return: probit(Pmiss)
    :return: ROCCH EER = max_p mindcf(dcfweights=[p,1-p]), which is also equal to the intersection of the ROCCH
    with the line pfa = pmiss.
    :return: the mindcf: Identical to result using traditional ROC, but computed by mimimizing over the ROCCH
    vertices, rather than over all the ROC points.
    """
    assert ((pfa_min > 0) & (pfa_max < 1) & (pmiss_min > 0) & (pmiss_max < 1)), 'limits must be strictly inside (0,1)'
    assert((pfa_min < pfa_max) & (pmiss_min < pmiss_max)), 'pfa and pmiss min and max values are not consistent'

    pmiss, pfa = rocch(tar, non)
    mindcf = 0.0

    if dcfweights.shape == (2,):
        dcf = numpy.dot(dcfweights, numpy.vstack((pmiss, pfa)))
        mindcf = numpy.min(dcf)
        if normalize:
            mindcf = mindcf/min(dcfweights)

    # pfa is decreasing
    # pmiss is increasing
    box = Box(left=pfa_min, right=pfa_max, top=pmiss_max, bottom=pmiss_min)

    x = []
    y = []
    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]
        xdots, ydots, eerseg = plotseg(xx, yy, box, dps)
        x = x + xdots.tolist()
        y = y + ydots.tolist()
        eer = max(eer, eerseg)

    return numpy.array(x), numpy.array(y), eer, mindcf


class DetPlot:
    """A class for creating a plot for displaying detection performance
    with the axes scaled and labelled so that a normal Gaussian
    distribution will plot as a straight line.
    
        - The y axis represents the miss probability.
        - The x axis represents the false alarm probability.
    
    :attr __plotwindow__: PlotWindow object to plot into
    :attr __title__: title of the plot
    :attr __sys_name__: list of IDs of the systems
    :attr __tar__: list of arrays of of target scores for each system
    :attr __non__: list of arrays of the non-target scores for each system
    :attr __figure__: figure to plot into
    """

    def __init__(self, window_style='old', plot_title=''):
        """Initialize an empty DetPlot object"""
        self.__plotwindow__ = PlotWindow(window_style)
        self.__title__ = plot_title
        self.__sys_name__ = []
        self.__tar__ = []
        self.__non__ = []
        self.__figure__ = ''
        self.title = ''

    def set_title(self, title):
        """Modify the title of a DetPlot object

        :param title: title of the plot to display
        """
        self.title = title

    def create_figure(self, idx=0):
        """Create a figure to plot the DET-curve.
        Default plot everything on one single figure

        :param idx: Index of the figure to create. Default is 0.
        """
        self.__figure__ = mpl.figure(idx)
        ax = self.__figure__.add_subplot(111)
        ax.set_aspect('equal')
        mpl.axis([__probit__(self.__plotwindow__.__pfa_limits__[0]),
                  __probit__(self.__plotwindow__.__pfa_limits__[1]),
                  __probit__(self.__plotwindow__.__pmiss_limits__[0]),
                  __probit__(self.__plotwindow__.__pmiss_limits__[1])])
        xticks = __probit__(self.__plotwindow__.__xticks__)
        yticks = __probit__(self.__plotwindow__.__yticks__)
        ax.set_xticks(xticks)
        ax.set_xticklabels(self.__plotwindow__.__xticklabels__,
                           size='x-small')
        ax.set_yticks(yticks)
        ax.set_yticklabels(self.__plotwindow__.__yticklabels__, size='x-small')
        if not self.__title__ == '':
            mpl.title(self.__title__)
        mpl.grid(True)
        mpl.xlabel('False Acceptance Rate [in %]')
        mpl.ylabel('False Rejection Rate [in %]')

        # assuring, limits are kept by matplotlib after probit transform of axes
        mpl.gca().set_xlim(
            left=__probit__(self.__plotwindow__.__pfa_limits__[0]),
            right=__probit__(self.__plotwindow__.__pfa_limits__[1])
        )
        mpl.gca().set_ylim(
            bottom=__probit__(self.__plotwindow__.__pmiss_limits__[0]),
            top=__probit__(self.__plotwindow__.__pmiss_limits__[1])
    )

    def set_system(self, tar, non, sys_name=''):
        """Sets the scores to be plotted. This function must be called
        before plots are made for a system, but it can be called several
        times with different systems (with calls to plotting functions in
        between) so that curves for different systems appear on the same plot.
        
        :param tar: A vector of target scores.
        :param non: A vector of non-target scores.
        :param sys_name: A string describing the system.  This string will 
            be prepended to the plot names in the legend. 
            You can pass an empty string to this argument or omit it.
        
        """
        assert tar.ndim == 1, 'Vector of target scores should be 1-dimensional'
        assert non.ndim == 1, 'Vector of nontarget scores should be 1-dimensional'
        assert tar.shape[0] > 0, 'Vector of target scores is empty'
        assert non.shape[0] > 0, 'Vector of nontarget scores is empty'
        self.__sys_name__.append(sys_name)
        self.__tar__.append(tar)
        self.__non__.append(non)

    def set_system_from_scores(self, scores, key, sys_name=''):
        """Sets the scores to be plotted.  This function must be called
        before plots are made for a system, but it can be called several
        times with different systems (with calls to plotting functions in
        between) so that curves for different systems appear on the same plot.
        
        :param scores: A Scores object containing system scores.
        :param key: A Key object for distinguishing target and non-target scores.
        :param sys_name: A string describing the system.  This string will be 
            prepended to the plot names in the legend.  You can pass an 
            empty string to this argument or omit it.
        
        """
        assert isinstance(scores, Scores), 'First argument should be a Score object'
        assert isinstance(key, Key), 'Second argument should be a Key object'
        assert scores.validate(), 'Wrong format of Scores'
        assert key.validate(), 'Wrong format of Key'
        tar, non = scores.get_tar_non(key)
        self.set_system(tar, non, sys_name)

    def plot_steppy_det(self, idx=0, style='color', plot_args=''):
        """Plots a DET curve.
        
        :param idx: the idx of the curve to plot in case tar and non have 
            several dimensions
        :param style: style of the curve, can be gray or color
        :param plot_args: a cell array of arguments to be passed to plot 
            that control the appearance of the curve.
        """
        Pmiss, Pfa = __compute_roc__(self.__tar__[idx], self.__non__[idx])
        Pmiss, Pfa = __filter_roc__(Pmiss, Pfa)

        x = __probit__(Pfa)
        y = __probit__(Pmiss)

        # In case the plotting arguments are not specified, they are initialized
        # by using default values
        if not (isinstance(plot_args, tuple) & (len(plot_args) == 3)):
            if style == 'gray':
                plot_args = grayStyle[idx]
            else:
                plot_args = colorStyle[idx]

        fig = mpl.plot(x, y,
                       label=self.__sys_name__[idx],
                       color=plot_args[0],
                       linestyle=plot_args[1],
                       linewidth=plot_args[2])
        mpl.legend()
        if matplotlib.get_backend() == 'agg':
            mpl.savefig(self.__title__ + '.pdf')

    def plot_rocch_det(self, idx=0, style='color', target_prior=0.001, plot_args=''):
        """Plots a DET curve using the ROCCH.

        :param idx: index of the figure to plot on
        :param style: style of the DET-curve (see DetPlot description)
        :param target_prior: prior of the target trials
        :param plot_args: a list of arguments to be passed
            to plot that control the appearance of the curve.
        """
        # In case the plotting arguments are not specified, they are initialized
        # by using default values
        if not (isinstance(plot_args, tuple) & (len(plot_args) == 3)):
            if style == 'gray':
                plot_args = grayStyle[idx]
            else:
                plot_args = colorStyle[idx]

        pfa_min = self.__plotwindow__.__pfa_limits__[0]
        pfa_max = self.__plotwindow__.__pfa_limits__[1]
        pmiss_min = self.__plotwindow__.__pmiss_limits__[0]
        pmiss_max = self.__plotwindow__.__pmiss_limits__[1]

        dps = 100  # dots per segment

        tmp = numpy.array([sigmoid(__logit__(target_prior)), sigmoid(-__logit__(target_prior))])
        x, y, eer, mindcf = rocchdet(self.__tar__[idx], self.__non__[idx],
                                     tmp, pfa_min, pfa_max,
                                     pmiss_min, pmiss_max, dps, normalize=True)

        fig = mpl.plot(x, y,
                       label='{}; (eer; minDCF) = ({:.03}; {:.04})'.format(
                           self.__sys_name__[idx],
                           100. * eer,
                           100. * mindcf),
                       color=plot_args[0],
                       linestyle=plot_args[1],
                       linewidth=plot_args[2])
        mpl.legend()
        if matplotlib.get_backend() == 'agg':
            mpl.savefig(self.__title__ + '.pdf')

    def plot_mindcf_point(self, target_prior, idx=0, plot_args='ok', legend_string=''):
        """Places the mindcf point for the current system.
        
        :param target_prior: The effective target prior.
        :param idx: inde of the figure to plot in
        :param plot_args: a list of arguments to be 
            passed to 'plot' that control the appearance of the curve.
        :param legend_string: Optional. A string to describe this curve 
            in the legend.
        """
        mindcf, pmiss, pfa, prbep, eer = fast_minDCF(self.__tar__[idx],
                                                     self.__non__[idx], __logit__(target_prior), True)
        if (pfa < self.__plotwindow__.__pfa_limits__[0]) | (pfa > self.__plotwindow__.__pfa_limits__[1]):
            logging.warning('pfa of %f is not between %f and %f mindcf point will not be plotted.', format(pfa),
                            self.__plotwindow__.__pfa_limits__[0],
                            self.__plotwindow__.__pfa_limits__[1])
        elif (pmiss < self.__plotwindow__.__pmiss_limits__[0]) | (pmiss > self.__plotwindow__.__pmiss_limits__[1]):
            logging.warning('pmiss of %f is not between %f and %f. The mindcf point will not be plotted.',
                            pmiss, self.__plotwindow__.__pmiss_limits__[0],
                            self.__plotwindow__.__pmiss_limits__[1])
        else:
            fig = mpl.plot(__probit__(pfa), __probit__(pmiss), plot_args)
            if matplotlib.get_backend() == 'agg':
                mpl.savefig(self.__title__ + '.pdf')

    def plot_DR30_fa(self,
                     idx=0,
                     plot_args=((0, 0, 0), '--', 1),
                     legend_string=''):
        """Plots a vertical line indicating the Doddington 30 point for
        false alarms. This is the point left of which the number of false
        alarms is below 30, so that the estimate of the false alarm rate
        is no longer good enough to satisfy Doddington's Rule of 30.
       
        :param idx: index of the figure to plot in
        :param plot_args: A cell array of arguments to be passed to 'plot' 
            that control the appearance of the curve.
        :param legend_string: Optional. A string to describe this curve 
            in the legend.
        """
        assert isinstance(plot_args, tuple) & (len(plot_args) == 3), 'Invalid plot_args'

        pfa_min = self.__plotwindow__.__pfa_limits__[0]
        pfa_max = self.__plotwindow__.__pfa_limits__[1]
        pmiss_min = self.__plotwindow__.__pmiss_limits__[0]
        pmiss_max = self.__plotwindow__.__pmiss_limits__[1]
        pfaval = 30.0 / self.__non__[idx].shape[0]

        if (pfaval < pfa_min) | (pfaval > pfa_max):
            logging.warning('Pfa DR30 of %f is not between %f and %f Pfa DR30 line will not be plotted.',
                            pfaval, pfa_min, pfa_max)
        else:
            drx = __probit__(pfaval)
            mpl.axvline(drx,
                        ymin=__probit__(pmiss_min),
                        ymax=__probit__(pmiss_max),
                        color=plot_args[0],
                        linestyle=plot_args[1],
                        linewidth=plot_args[2],
                        label=legend_string)

    def plot_DR30_miss(self,
                       idx=0,
                       plot_args=((0, 0, 0), '--', 1),
                       legend_string=''):
        """Plots a horizontal line indicating the Doddington 30 point for
        misses.  This is the point above which the number of misses is
        below 30, so that the estimate of the miss rate is no longer good
        enough to satisfy Doddington's Rule of 30.
        
        :param idx: index of the figure to plot in
        :param plot_args: A cell array of arguments to be passed 
            to 'plot' that control the appearance of the curve.
        :param legend_string: Optional. A string to describe this curve 
            in the legend.
        """
        assert isinstance(plot_args, tuple) & (len(plot_args) == 3), 'Invalid plot_args'

        pfa_min = self.__plotwindow__.__pfa_limits__[0]
        pfa_max = self.__plotwindow__.__pfa_limits__[1]
        pmiss_min = self.__plotwindow__.__pmiss_limits__[0]
        pmiss_max = self.__plotwindow__.__pmiss_limits__[1]
        pmissval = 30.0 / self.__tar__[idx].shape[0]

        if (pmissval < pmiss_min) | (pmissval > pmiss_max):
            logging.warning('Pmiss DR30 of is not between %f and %f Pfa DR30 line will not be plotted.',
                            pmissval, pmiss_min, pmiss_max)
        else:
            dry = __probit__(pmissval)
            mpl.axhline(y=dry,
                        xmin=__probit__(pfa_min),
                        xmax=__probit__(pfa_max),
                        color=plot_args[0],
                        linestyle=plot_args[1],
                        linewidth=plot_args[2],
                        label=legend_string)

    def plot_DR30_both(self,
                       idx=0,
                       plot_args_fa=((0, 0, 0), '--', 1),
                       plot_args_miss=((0, 0, 0), '--', 1),
                       legend_string=''):
        """Plots two lines indicating Doddington's Rule of 30 points: one
        for false alarms and one for misses.  See the documentation of
        plot_DR30_fa and plot_DR30_miss for details.
        
        :param idx: index of the figure to plot in
        :param plot_args_fa: A tuple of arguments to be passed to 'plot' that control
            the appearance of the DR30_fa point.
        :param plot_args_miss: A tuple of arguments to be passed to 'plot' that control
            the appearance of the DR30_miss point.
        :param legend_string: Optional. A string to describe this curve
            in the legend.
        """
        self.plot_DR30_fa(idx, plot_args_fa, 'pfa DR30')
        self.plot_DR30_miss(idx, plot_args_miss, 'pmiss DR30')

    def display_legend(self):
        # to complete
        pass

    def save_as_pdf(self, outfilename):
        # to complete
        pass

    def add_legend_entry(self, lh, legend_string, append_name):
        # to complete
        pass


