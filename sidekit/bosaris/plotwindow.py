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
This is the 'plotwindow' module
"""
import numpy


__author__ = "Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brummer", "Edward de Villiers"]


class PlotWindow:
    """A class that is used to define the parameters of a plotting window.

    :attr __pfa_limits__: ndarray of two values that determine the limits of the pfa axis.
            Default is [0.0005, 0.5]
    :attr __pmiss_limits__: ndarray of two values that determine the limits of the pmiss axis.
                Default is [0.0005, 0.5]
    :attr __xticks__: coordonates of the ticks on the horizontal axis
    :attr __xticklabels__: labels of the ticks on the horizontal axis in a ndarray of strings
    :attr __yticks__: coordonates of the ticks on the vertical axis
    :attr __yticklabels__: labels of the ticks on the vertical axis in a ndarray of strings
    """

    def __init__(self, input_type=''):
        """Initialize PlotWindow object to one of the pre-defined ploting type.
        - 'new'
        - 'old'
        - 'big'
        - 'sre10'

        :param input_type: the type of DET plot to display. Default is 'old'
        """

        if input_type == '':
            self.__pfa_limits__ = numpy.array([5e-4, 5e-1])
            self.__pmiss_limits__ = numpy.array([5e-4, 5e-1])
            self.__xticks__ = numpy.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4])
            self.__xticklabels__ = numpy.array(['0.1', '0.2', '0.5', ' 1 ', ' 2 ', ' 5 ', '10 ', '20 ', '30 ', '40 '])
            self.__yticks__ = numpy.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4])
            self.__yticklabels__ = numpy.array(['0.1', '0.2', '0.5', ' 1 ', ' 2 ', ' 5 ', '10 ', '20 ', '30 ', '40 '])
        elif input_type == 'new':
            self.axis_new()
        elif input_type == 'old':
            self.axis_old()
        elif input_type == 'big':
            self.axis_big()
        elif input_type == 'sre10':
            self.axis_sre10()
        else:
            raise Exception('Error, wrong type of PlotWindow')

    def make_plot_window_from_values(self, pfa_limits, pmiss_limits, xticks, xticklabels, yticks, yticklabels):
        """Initialize PlotWindow from provided values

        :param pfa_limits: ndarray of two values that determine the limits of the pfa axis.
        :param pmiss_limits: ndarray of two values that determine the limits of the pmiss axis.
        :param xticks: coordonates of the ticks on the horizontal axis.
        :param xticklabels: labels of the ticks on the horizontal axis in a ndarray of strings.
        :param yticks: coordonates of the ticks on the vertical axis.
        :param yticklabels: labels of the ticks on the vertical axis in a ndarray of strings.
        """
        self.__pfa_limits__ = pfa_limits
        self.__pmiss_limits__ = pmiss_limits
        self.__xticks__ = xticks
        self.__xticklabels__ = xticklabels
        self.__yticks__ = yticks
        self.__yticklabels__ = yticklabels

    def axis_new(self):
        """Set axis value to new ones

        - pfa ranges from 0.000005 to 0.005
        - pmiss ranges from 0.01 to 0.99
        """
        self.__pfa_limits__ = numpy.array([5e-6, 5e-3])
        self.__pmiss_limits__ = numpy.array([1e-2, 0.99])
        self.__xticks__ = numpy.array([1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
        self.__xticklabels__ = numpy.array(['1e-3', '2e-3', '5e-3', '0.01', '0.02', '0.05', '0.1 ', '0.2 '])
        self.__yticks__ = numpy.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
        self.__yticklabels__ = numpy.array([' 2 ', ' 5 ', '10 ', '20 ', '30 ',
                                            '40 ', '50 ', '60 ', '70 ', '80 ', '90 ', '95 ', '98 '])

    def axis_old(self):
        """Set axis value to old ones (NIST-SRE08 style)

        - pfa ranges from 0.0005 to 0.5
        - pmiss ranges from 0.0005 to 0.5
        """
        self.__pfa_limits__ = numpy.array([5e-4, 5e-1])
        self.__pmiss_limits__ = numpy.array([5e-4, 5e-1])
        self.__xticks__ = numpy.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4])
        self.__xticklabels__ = numpy.array(['0.1', '0.2', '0.5', ' 1 ', ' 2 ', ' 5 ', '10 ', '20 ', '30 ', '40 '])
        self.__yticks__ = numpy.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4])
        self.__yticklabels__ = numpy.array(['0.1', '0.2', '0.5', ' 1 ', ' 2 ', ' 5 ', '10 ', '20 ', '30 ', '40 '])

    def axis_big(self):
        """Set axis value to big ones

        - pfa ranges  from 0.000005 to 0.99
        - pmiss ranges from 0.000005 to0.99
        """
        self.__pfa_limits__ = numpy.array([5e-6, 0.99])
        self.__pmiss_limits__ = numpy.array([5e-6, 0.99])
        self.__yticks__ = numpy.array([5e-6, 5e-5, 5e-4, 0.5e-2, 2.5e-2, 10e-2,
                                       25e-2, 50e-2, 72e-2, 88e-2, 96e-2, 99e-2])
        self.__yticklabels__ = numpy.array(['5e-4', '5e-3', '0.05', '0.5 ',
                                            '2.5 ', ' 10 ', ' 25 ', ' 50 ', ' 72 ', ' 88 ', ' 96 ', ' 99 '])
        self.__xticks__ = numpy.array([5e-5, 5e-4, 0.5e-2, 2.5e-2, 10e-2, 25e-2, 50e-2, 72e-2, 88e-2, 96e-2, 99e-2])
        self.__xticklabels__ = numpy.array(['5e-3', '0.05', '0.5 ', '2.5 ',
                                            ' 10 ', ' 25 ', ' 50 ', ' 72 ', ' 88 ', ' 96 ', ' 99 '])

    def axis_sre10(self):
        """Set axis value to NIST-SRE10 style

        - pfa ranges from 0.000003 to 0.5
        - pmiss ranges from 0.0003 to 0.9
        """
        self.__pfa_limits__ = numpy.array([3e-6, 5e-1])
        self.__pmiss_limits__ = numpy.array([3e-4, 9e-1])
        self.__xticks__ = numpy.array([1e-5, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 4e-1])
        self.__xticklabels__ = numpy.array(['0.001', ' 0.01', '  0.1', '  0.2',
                                            '  0.5', '    1', '    2', '    5', '   10', '   20', '   40'])
        self.__yticks__ = numpy.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 4e-1, 8e-1])
        self.__yticklabels__ = numpy.array(['0.1', '0.2', '0.5', ' 1 ', ' 2 ', ' 5 ', ' 10', ' 20', ' 40', ' 80'])
