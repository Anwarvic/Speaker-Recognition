import os
import subprocess
import h5py
import numpy as np




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


def preprocessAudioFile(inwav, outwav, sample_rate, n_channels, bit=16):
    """
    This function is used to preprocess audio for SideKit.
    As we can see, the only default value set is for precision (16-bit)
    and that's because SideKit has problems for other precisions.
    And all audio files need to be the same criteria.
    NOTE:
    It only reads wav files in PCM format, so make sure that's the case!!
    """
    parent, _ = os.path.split(outwav)
    if not os.path.exists(parent):
        os.mkdir(parent)
    command = "sox {} -r {} -c {} -b {} {}"\
                    .format(inwav, sample_rate, n_channels, bit, outwav)
    subprocess.call(command, shell=True) 