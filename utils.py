import os
import yaml
import subprocess
import h5py
import numpy as np



def safe_makedir(dirname):
    """This function takes a directory name as an argument"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def explore_file(filepath):
    """
    This function is used to explore any hdf5 file.
    """
    h5 = h5py.File(filepath, mode="r")
    print(h5.keys())
    for key, value in h5.items():
        print("Key:", key)
        print("Value Type:", value)
        value = np.array(value)
        print("Value Shape:", value.shape)
        print("Value:", value)
        print("="*25)


def convert_wav(inpath, outpath, no_channels, sampling_rate, bit_precision,
                showWarning=False):
    """
    Convert the waves to a pre-defined sampling rate, number of channels and
    bit-precision using SoX tool. So, it should be installed!!
    """
    parent, _ = os.path.split(outpath)
    safe_makedir(parent)
    command = "sox {} -r {} -c {} -b {} {}".format( inpath,
                                                    sampling_rate,
                                                    no_channels,
                                                    bit_precision,
                                                    outpath)
    if showWarning:
        subprocess.call(command, shell=True) 
    else:
        with open(os.devnull, 'w') as FNULL:
            subprocess.call(command, shell=True, stdout=FNULL,
                                                    stderr=subprocess.STDOUT)


def parse_yaml(filepath="conf.yaml"):
    """
    This method parses the YAML configuration file and returns the parsed info
    as python dictionary.
    Args:
        filepath (string): relative path of the YAML configuration file
    """
    with open(filepath, 'r') as fin:
        try:
            conf_dictionary = yaml.safe_load(fin)
            return conf_dictionary
        except Exception as exc:
            print("ERROR while parsing YAML conf.")
            print(exc)