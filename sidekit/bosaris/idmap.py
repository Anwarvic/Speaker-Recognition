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
This is the 'idmap' module
"""
import sys
import numpy
import logging
import copy
import h5py

from sidekit.sidekit_wrappers import check_path_existance


__author__ = "Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brummer", "Edward de Villiers"]


class IdMap:
    """A class that stores a map between identifiers (strings).  One
    list is called 'leftids' and the other 'rightids'.  The class
    provides methods that convert a sequence of left ids to a
    sequence of right ids and vice versa.  If `leftids` or `rightids`
    contains duplicates then all occurrences are used as the index
    when mapping.

    :attr leftids: a list of classes in a ndarray
    :attr rightids: a list of segments in a ndarray
    :attr start: index of the first frame of the segment
    :attr stop: index of the last frame of the segment
    """

    def __init__(self, idmap_filename=''):
        """Initialize an IdMap object

        :param idmap_filename: name of a file to load. Default is ''.
        In case the idmap_filename is empty, initialize an empty IdMap object.
        """
        self.leftids = numpy.empty(0, dtype="|O")
        self.rightids = numpy.empty(0, dtype="|O")
        self.start = numpy.empty(0, dtype="|O")
        self.stop = numpy.empty(0, dtype="|O")

        if idmap_filename == '':
            pass
        else:
            tmp = IdMap.read(idmap_filename)
            self.leftids = tmp.leftids
            self.rightids = tmp.rightids
            self.start = tmp.start
            self.stop = tmp.stop

    def __repr__(self):
        ch = '-' * 30 + '\n'
        ch += 'left ids:' + self.leftids.__repr__() + '\n'
        ch += 'right ids:' + self.rightids.__repr__() + '\n'
        ch += 'seg start:' + self.start.__repr__() + '\n'
        ch += 'seg stop:' + self.stop.__repr__() + '\n'
        ch += '-' * 30 + '\n'
        return ch

    @check_path_existance
    def write(self, output_file_name):
        """ Save IdMap in HDF5 format

        :param output_file_name: name of the file to write to
        """
        assert self.validate(), "Error: wrong IdMap format"
        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("leftids", data=self.leftids.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("rightids", data=self.rightids.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            # WRITE START and STOP
            start = copy.deepcopy(self.start)
            start[numpy.isnan(self.start.astype('float'))] = -1
            start = start.astype('int32', copy=False)

            stop = copy.deepcopy(self.stop)
            stop[numpy.isnan(self.stop.astype('float'))] = -1
            stop = stop.astype('int32', copy=False)

            f.create_dataset("start", data=start,
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("stop", data=stop,
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)

    @check_path_existance
    def write_txt(self, output_file_name):
        """Saves the Id_Map to a text file.
        
        :param output_file_name: name of the output text file
        """
        with open(output_file_name, 'w') as outputFile:
            for left, right, start, stop in zip(self.leftids, self.rightids, self.start, self.stop):
                line = ' '.join(filter(None, (left, right, str(start), str(stop)))) + '\n'
                outputFile.write(line)

    def map_left_to_right(self, leftidlist):
        """Maps an array of ids to a new array of ids using the given map.  
        The input ids are matched against the leftids of the map and the
        output ids are taken from the corresponding rightids of the map.
        
        Beware: if leftids are not unique in the IdMap, only the last value 
        corresponding is kept

        :param leftidlist: an array of strings to be matched against the
            leftids of the idmap.  The rightids corresponding to these
            leftids will be returned.

        :return: an array of strings that are the mappings of the
            strings in leftidlist.
        """
        tmp_dict = dict(zip(self.leftids, self.rightids))
        inter = numpy.intersect1d(self.leftids, leftidlist)
        rightids = numpy.empty(inter.shape[0], '|O')
        
        idx = 0
        for left in leftidlist:
            if left in inter:
                rightids[idx] = tmp_dict[left]
                idx += 1

        lost_ids = numpy.unique(leftidlist).shape[0] - inter.shape[0]
        if lost_ids:
            logging.warning('{} ids could not be mapped'.format(lost_ids))

        return rightids

    def map_right_to_left(self, rightidlist):
        """Maps an array of ids to a new array of ids using the given map.  
        The input ids are matched against the rightids of the map and the
        output ids are taken from the corresponding leftids of the map.

        Beware: if rightids are not unique in the IdMap, only the last value 
        corresponding is kept

        :param rightidlist: An array of strings to be matched against the
            rightids of the idmap.  The leftids corresponding to these
            rightids will be returned.

        :return: an array of strings that are the mappings of the
            strings in rightidlist.
        """
        tmp_dict = dict(zip(self.rightids, self.leftids))
        inter = numpy.intersect1d(self.rightids, rightidlist)
        leftids = numpy.empty(inter.shape[0], '|O')
        
        idx = 0
        for right in rightidlist:
            if right in inter:
                leftids[idx] = tmp_dict[right]
                idx += 1        
        
        lost_ids = numpy.unique(rightidlist).shape[0] - inter.shape[0]
        if lost_ids:
            logging.warning('{} ids could not be mapped'.format(lost_ids))

        return leftids

    def filter_on_left(self, idlist, keep):
        """Removes some of the information in an idmap.  Depending on the
        value of 'keep', the idlist indicates the strings to retain or
        the strings to discard.

        :param idlist: an array of strings which will be compared with
            the leftids of the current.
        :param keep: A boolean indicating whether idlist contains the ids to
            keep or to discard.

        :return: a filtered version of the current IdMap.
        """
        # get the list of ids to keep
        if keep:
            keepids = numpy.unique(idlist)
        else:
            keepids = numpy.setdiff1d(self.leftids, idlist)
        
        keep_idx = numpy.in1d(self.leftids, keepids)
        out_idmap = IdMap()
        out_idmap.leftids = self.leftids[keep_idx]
        out_idmap.rightids = self.rightids[keep_idx]
        out_idmap.start = self.start[keep_idx]
        out_idmap.stop = self.stop[keep_idx]
        
        return out_idmap

    def filter_on_right(self, idlist, keep):
        """Removes some of the information in an idmap.  Depending on the
        value of 'keep', the idlist indicates the strings to retain or
        the strings to discard.

        :param idlist: an array of strings which will be compared with
            the rightids of the current IdMap.
        :param keep: a boolean indicating whether idlist contains the ids to
            keep or to discard.

        :return: a filtered version of the current IdMap.
        """
        # get the list of ids to keep
        if keep:
            keepids = numpy.unique(idlist)
        else:
            keepids = numpy.setdiff1d(self.rightids, idlist)
        
        keep_idx = numpy.in1d(self.rightids, keepids)
        out_idmap = IdMap()
        out_idmap.leftids = self.leftids[keep_idx]
        out_idmap.rightids = self.rightids[keep_idx]
        out_idmap.start = self.start[keep_idx]
        out_idmap.stop = self.stop[keep_idx]
        return out_idmap

    def validate(self, warn=False):
        """Checks that an object of type Id_Map obeys certain rules that
        must alows be true.
        
        :param warn: boolean. If True, print a warning if strings are
            duplicated in either left or right array

        :return: a boolean value indicating whether the object is valid.

        """
        ok = (self.leftids.shape == self.rightids.shape == self.start.shape == self.stop.shape) & self.leftids.ndim == 1

        if warn & (self.leftids.shape != numpy.unique(self.leftids).shape):
            logging.warning('The left id list contains duplicate identifiers')
        if warn & (self.rightids.shape != numpy.unique(self.rightids).shape):
            logging.warning('The right id list contains duplicate identifiers')
        return ok

    def set(self, left, right, start=None, stop=None):
        self.leftids = copy.deepcopy(left)
        self.rightids = copy.deepcopy(right)

        if start is not None:
            self.start = copy.deepcopy(start)
        else:
            self.start = numpy.empty(self.rightids.shape, '|O')

        if stop is not None:
            self.stop = copy.deepcopy(stop)
        else:
            self.stop = numpy.empty(self.rightids.shape, '|O')

    @staticmethod
    def read(input_file_name):
        """Read IdMap in hdf5 format.

        :param input_file_name: name of the file to read from
        """
        with h5py.File(input_file_name, "r") as f:
            idmap = IdMap()

            idmap.leftids = f.get("leftids").value
            idmap.rightids = f.get("rightids").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                idmap.leftids = idmap.leftids.astype('U255', copy=False)
                idmap.rightids = idmap.rightids.astype('U255', copy=False)

            tmpstart = f.get("start").value
            tmpstop = f.get("stop").value
            idmap.start = numpy.empty(f["start"].shape, '|O')
            idmap.stop = numpy.empty(f["stop"].shape, '|O')
            idmap.start[tmpstart != -1] = tmpstart[tmpstart != -1]
            idmap.stop[tmpstop != -1] = tmpstop[tmpstop != -1]

            assert idmap.validate(), "Error: wrong IdMap format"
            return idmap

    @classmethod
    @check_path_existance
    def read_txt(cls, input_file_name):
        """Read IdMap in text format.

        :param input_file_name: name of the file to read from
        """
        idmap = IdMap()

        with open(input_file_name, "r") as f:
            columns = len(f.readline().split(' '))

        if columns == 2:
            idmap.leftids, idmap.rightids = numpy.loadtxt(input_file_name,
                                                          dtype={'names': ('left', 'right'), 'formats': ('|O', '|O')},
                                                          usecols=(0, 1), unpack=True)
            idmap.start = numpy.empty(idmap.rightids.shape, '|O')
            idmap.stop = numpy.empty(idmap.rightids.shape, '|O')
        
        # If four columns
        elif columns == 4:
            idmap.leftids, idmap.rightids, idmap.start, idmap.stop = numpy.loadtxt(
                input_file_name,
                dtype={'names': ('left', 'right', 'start', 'stop'),
                       'formats': ('|O', '|O', 'int', 'int')}, unpack=True)
    
        if not idmap.validate():
            raise Exception('Wrong format of IdMap')
        assert idmap.validate(), "Error: wrong IdMap format"
        return idmap

    def merge(self, idmap2):
        """ Merges the current IdMap with another IdMap or a list of IdMap objects..

        :param idmap2: Another Id_Map object.

        :return: an Id_Map object that contains the information from the two
            input Id_Maps.
        """
        idmap = IdMap()
        if self.validate() & idmap2.validate():
            # create tuples of (model,seg) for both IdMaps for quick comparaison
            tup1 = [(mod, seg) for mod, seg in zip(self.leftids, self.rightids)]
            tup2 = [(mod, seg) for mod, seg in zip(idmap2.leftids, idmap2.rightids)]

            # Get indices of common sessions
            existing_sessions = set(tup1).intersection(set(tup2))
            # Get indices of sessions which are not common in idmap2
            idx_new = numpy.sort(numpy.array([idx for idx, sess in enumerate(tup2) if sess not in tup1]))
            if len(idx_new) == 0:
                idx_new = numpy.zeros(idmap2.leftids.shape[0], dtype='bool')

            idmap.leftids = numpy.concatenate((self.leftids, idmap2.leftids[idx_new]), axis=0)
            idmap.rightids = numpy.concatenate((self.rightids, idmap2.rightids[idx_new]), axis=0)
            idmap.start = numpy.concatenate((self.start, idmap2.start[idx_new]), axis=0)
            idmap.stop = numpy.concatenate((self.stop, idmap2.stop[idx_new]), axis=0)

        else:
            raise Exception('Cannot merge IdMaps, wrong type')

        if not idmap.validate():
            raise Exception('Wrong format of IdMap')

        return idmap
