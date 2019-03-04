import os
import sidekit
import numpy as np
from multiprocessing import cpu_count
import logging
logging.basicConfig(level=logging.INFO)

class FeaturesExtractor():

    def __init__(self):
        # The parent directory of the project
        self.BASE_DIR = "/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env"
        # Number of parallel threads
        self.NUM_THREADS = cpu_count()
        # The features need to be extracted
        #  -> log-energy: energy
        #  -> "cep" for cepstral coefficients, its size is ceps_number which is CEPS_NUMBER
        #  -> "fb" for filter-banks, its size is FILTER_BANK_ISE
        #  -> "energy" for the log-energy, its size is 1
        #  -> "bnf"
        self.FEAUTRES = ["vad", "energy", "cep", "fb"]
        
        #filter bank can either be "log" and "lin" for linear
        self.FILTER_BANK = "log"
        self.FILTER_BANK_SIZE = 24
        self.LOWER_FREQUENCY = 300
        self.HIGHER_FREQUENCY = 3400
        #  -> vad: type of voice activity detection algorithm to use.       
        self.VAD = "snr" #can be either "energy", "snr", "percentil" or "lbl".
        self.SNR_RATIO = 40 if self.VAD == "snr" else None
        # cepstral coefficients
        self.WINDOW_SIZE = 0.025
        self.WINDOW_SHIFT = 0.01
        self.CEPS_NUMBER = 19


    def reviewMemberVariables(self):
        """
        This method is used to modify the values of some of the member
        variables based on the features inserted.
        """
        # Review fb
        if "fb" not in self.FEAUTRES:
            self.FILTER_BANK      = None
            self.FILTER_BANK_SIZE = None
            self.LOWER_FREQUENCY  = None
            self.HIGHER_FREQUENCY = None

        # Review cep
        if "cep" not in self.FEAUTRES:
            self.WINDOW_SIZE = None
            self.WINDOW_SHIFT = None
            self.CEPS_NUMBER = None
        
        # Review vad
        if "vad" not in self.FEAUTRES:
            self.VAD = None
            self.SNR_RATIO = None


    def extractFeatures(self, group):
        """
        This function computes the acoustic parameters of audio files insied 
        "self.BASE_DIR/group":
        for a list of audio files and save them to disk in a HDF5 format
        """
        in_files = os.listdir(os.path.join(self.BASE_DIR, "audio", group))
        feat_dir = os.path.join(self.BASE_DIR, "feat", group)
        # Feature extraction
        # lower_frequency: lower frequency (in Herz) of the filter bank
        # higher_frequency: higher frequency of the filter bank
        # filter_bank: type of fiter scale to use, can be "lin" or "log" (for linear of log-scale)
        # filter_bank_size: number of filters banks
        # window_size: size of the sliding window to process (in seconds)
        # shift: time shift of the sliding window (in seconds)
        # ceps_number: number of cepstral coefficients to extract
        # snr: signal to noise ratio used for "snr" vad algorithm
        # vad: Type of voice activity detection algorithm to use.
        #      It Can be "energy", "snr", "percentil" or "lbl".
        # save_param: list of strings that indicate which parameters to save. The strings can be:
        # for bottle-neck features and "vad" for the frame selection labels.
        # keep_all_features: boolean, if True, all frames are writen; if False, keep only frames according to the vad label
        extractor = sidekit.FeaturesExtractor(audio_filename_structure=os.path.join(self.BASE_DIR, "audio", group, "{}"),
                                              feature_filename_structure=os.path.join(feat_dir, "{}.h5"),
                                              lower_frequency=self.LOWER_FREQUENCY,
                                              higher_frequency=self.HIGHER_FREQUENCY,
                                              filter_bank=self.FILTER_BANK,
                                              filter_bank_size=self.FILTER_BANK_SIZE,
                                              window_size=self.WINDOW_SIZE,
                                              shift=self.WINDOW_SHIFT,
                                              ceps_number=self.CEPS_NUMBER,
                                              vad=self.VAD,
                                              snr=self.SNR_RATIO,
                                              save_param=self.FEAUTRES,
                                              keep_all_features=True)

        # Prepare file lists
        # show_list: list of IDs of the show to process
        show_list = np.unique(np.hstack([in_files]))
        # channel_list: list of channel indices corresponding to each show
        channel_list = np.zeros_like(show_list, dtype = int)

        # save the features in feat_dir where the resulting-files parameters
        # are always concatenated in the following order:
        # (energy, fb, cep, bnf, vad_label).
        # SKIPPED: list to track faulty-files
        SKIPPED = []
        for show, channel in zip(show_list, channel_list):
            try:
                extractor.save(show, channel)
            except RuntimeError:
                logging.info("SKIPPED")
                SKIPPED.append(show)
                continue
        logging.info("Number of skipped files: "+str(len(SKIPPED)))
        for show in SKIPPED:
            logging.debug(show)
        #BUG: The following lines do the exact same thing
        # as the few ones above, but with using multi-processing where
        # num_thread: is the number of parallel process to run
        # This method freezes after sometime, so you can try it
        # extractor.save_list(show_list=show_list,
        #                     channel_list=channel_list,
        #                     num_thread=self.NUM_THREADS)





if __name__ == "__main__":
    ex = FeaturesExtractor()
    ex.extractFeatures("enroll")
    ex.extractFeatures("test")