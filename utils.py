import os
import subprocess



def preprocessAudioFile(inwav, outwav, sample_rate, n_channels, bit=16):
    """
    This function is used to preprocess audio for SideKit.
    As we can see, the only default value set is for precision (16-bit)
    and that's because SideKit has problems for other precisions.
    And all audio files need to be the same criteria.
    NOTE:
    wave.py only reads wav files in PCM format, so make sure that's the case!!
    """
    parent, _ = os.path.split(outwav)
    if not os.path.exists(parent):
        os.mkdir(parent)
    command = "sox {} -r {} -c {} -b {} {}"\
                    .format(inwav, sample_rate, n_channels, bit, outwav)
    subprocess.call(command, shell=True) 