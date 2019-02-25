# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
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

:mod:`sidekit_wrappers` provides wrappers for different purposes.
The aim when using wrappers is to simplify the development of new function
in an efficient manner
"""
import os
import numpy
import copy
import logging
from sidekit import PARALLEL_MODULE

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def coroutine(func):
    """
    Decorator that allows to forget about the first call of a coroutine .next()
    method or .send(None)
    This call is done inside the decorator
    :param func: the coroutine to decorate
    """
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start


def deprecated(func):
    """

    :param func:
    :return:
    """
    count = [0]

    def wrapper(*args, **kwargs):
        count[0] += 1
        if count[0] == 1:
            logging.warning(func.__name__ + ' is deprecated')
        return func(*args, **kwargs)
    return wrapper


def check_path_existance(func):
    """ Decorator for a function wich prototype is:
    
        func(features, outputFileName)
        
        This decorator gets the path included in 'outputFileName' if any 
        and check if this path exists; if not the path is created.
        :param func: function to decorate
    """
    def wrapper(*args, **kwargs):
        dir_name = os.path.dirname(args[1])  # get the path
        # Create the directory if it dosn't exist
        if not os.path.exists(dir_name) and (dir_name is not ''):
            os.makedirs(dir_name)            
        # Do the job
        func(*args, **kwargs)
    return wrapper


def process_parallel_lists(func):
    """
    Decorator that is used to parallelize process.
    This decorator takes a function with any list of arguments including 
    "num_thread" and parallelize the process by creating "num_thread" number of
    parallel process or threads.
    
    The choice of process or threas depends on the value of the global variable
    "PARALLEL_MODULE" that is defined in  ./sidekit/__init__.py
      
    Parallelization is done as follow:
        - all arguments have to be given to the decorator with their names
          any other case might limit the parallelization.
        - the function that is decorated is called by "num_thread" concurrent
          process (or threads) with the list of arguments that is given 
          to the decorator except special arguments (see below)

    Special arguments:
        Special arguments are the one that lead to parallelization.
        There are 3 types of special arguments which name end with a special 
        suffix:
        
        - arguments which names are "*_list" or "*_indices" are lists 
          or numpy arrays that will be split (equally or almost) and
          each sub-list will be passed as an argument for a process/thread
        
        - arguments which names are "_acc" are duplicated and each thread is
          given a copy of this accumulator. At the end of the function, all
          accumulators will be summed to return a unique accumulatore; thus
          any object passed as a "*_acc" argument has to implement 
          a "+" operator

        - arguments which names are "*_server" are duplicated using a deepcopy
          of the original argument. This is mostly used to pass servers such
          as FeaturesServer as arguments
    :param func: function to decorate
    
    """
    def wrapper(*args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        
        if len(args) > 1:
            print("Warning, some arguments are not named, computation might not be parallelized")
        
        num_thread = 1
        if "num_thread" in kwargs.keys():
            num_thread = kwargs["num_thread"]
        
        # On créé un dictionnaire de paramètres kwargs pour chaque thread
        if PARALLEL_MODULE in ['threading', 'multiprocessing'] and num_thread > 1:

            # If arguments end with _list or _indices,
            # set number of Threads to the minimum length of the lists and raise a warning
            list_length = numpy.inf
            for k, v in kwargs.items():
                # If v is a list or a numpy.array
                if k.endswith("_list") or k.endswith("_indices"):
                    list_length = min(list_length, len(list(v)))
            num_thread = min(num_thread, list_length)

            # Create a list of dictionaries, one per thread, and initialize
            # them with the keys
            parallel_kwargs = []
            for ii in range(num_thread):
                parallel_kwargs.append(dict(zip(kwargs.keys(), 
                                            [None]*len(kwargs.keys()))))
 
            for k, v in kwargs.items():
                
                # If v is a list or a numpy.array
                if k.endswith("_list") or k.endswith("_indices"):
                    sub_lists = numpy.array_split(v, num_thread)
                    for ii in range(num_thread):
                        parallel_kwargs[ii][k] = sub_lists[ii]  # the ii-th sub_list is used for the thread ii

                elif k == "num_thread":
                    for ii in range(num_thread):
                        parallel_kwargs[ii][k] = 1
 
                # If v is an accumulator (meaning k ends with "_acc")
                # v is duplicated for each thread
                elif k.endswith("_acc"):
                    for ii in range(num_thread):
                        parallel_kwargs[ii][k] = v

                # Duplicate servers for each thread
                elif k.endswith("_server") or k.endswith("_extractor"):
                    for ii in range(num_thread):
                        parallel_kwargs[ii][k] = copy.deepcopy(v)
                        
                # All other parameters are just given to each thread
                else:
                    for ii in range(num_thread):
                        parallel_kwargs[ii][k] = v
            
            if PARALLEL_MODULE is 'multiprocessing':
                import multiprocessing
                jobs = []
                multiprocessing.freeze_support()
                for idx in range(num_thread):
                    p = multiprocessing.Process(target=func, args=args, kwargs=parallel_kwargs[idx])
                    jobs.append(p)
                    p.start()
                for p in jobs:
                    p.join()
            
            elif PARALLEL_MODULE is 'threading':
                import threading
                jobs = []
                for idx in range(num_thread):
                    p = threading.Thread(target=func, args=args, kwargs=parallel_kwargs[idx])
                    jobs.append(p)
                    p.start()
                for p in jobs:
                    p.join()
        
            elif PARALLEL_MODULE is 'MPI':
                # TODO
                print("ParallelProcess using MPI is not implemented yet")
                pass
           
            # Sum accumulators if any
            for k, v in kwargs.items():
                if k.endswith("_acc"):
                    for ii in range(num_thread):
                        if isinstance(kwargs[k], list):
                            kwargs[k][0] += parallel_kwargs[ii][k][0]
                        else:
                            kwargs[k] += parallel_kwargs[ii][k]

        else:
            logging.debug("No Parallel processing with this module")
            func(*args, **kwargs)
        
    return wrapper


def accepts(*types, **kw):
    """Function decorator. Checks decorated function's arguments are
    of the expected types.
    
    Sources: https://wiki.python.org/moin/PythonDecoratorLibrary#Type_Enforcement_.28accepts.2Freturns.29
    
    Parameters:
        types -- The expected types of the inputs to the decorated function.
            Must specify type for each parameter.
        kw    -- Optional specification of 'debug' level (this is the only valid
            keyword argument, no other should be given).
            debug = ( 0 | 1 | 2 )
            
    """
    if not kw:
        # default level: MEDIUM
        debug = 1
    else:
        debug = kw['debug']
    try:
        def decorator(f):
            def newf(*args):
                if debug is 0:
                    return f(*args)
                assert len(args) == len(types)
                argtypes = tuple([a.__class__.__name__ for a in args])
                if argtypes != types:
                    print("argtypes = {} and types = {}".format(argtypes, types))
                    msg = info(f.__name__, types, argtypes, 0)
                    if debug is 1:
                        print('TypeWarning: ', msg)
                    elif debug is 2:
                        raise TypeError(msg)
                return f(*args)
            newf.__name__ = f.__name__
            return newf
        return decorator
    except KeyError as key:
        raise KeyError(key + "is not a valid keyword argument")
    except TypeError(msg):
        raise TypeError(msg)


def info(fname, expected, actual, flag):
    """Convenience function returns nicely formatted error/warning msg.
    :param fname: function to decorate
    :param expected: expected format of the function
    :param actual: actual format of the function to check
    :param flag: flag
    """
    _format = lambda types: ', '.join([str(t).split("'")[0] for t in types])
    expected, actual = _format(expected), _format(actual)
    msg = "'{}' method ".format(fname)\
          + ("accepts", "returns")[flag] + " ({}), but ".format(expected)\
          + ("was given", "result is")[flag] + " ({})".format(actual)
    return msg
