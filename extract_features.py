import os
import sidekit
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from multiprocessing import cpu_count
from utils import parse_yaml

class FeaturesExtractor():

    def __init__(self, conf_path):
        #parse the YAML configuration file
        self.conf = parse_yaml(conf_path)
        self.audio_dir = os.path.join(self.conf['outpath'], "audio") #input dir
        self.feat_dir = os.path.join(self.conf['outpath'], "feat")
        # Number of parallel threads
        self.NUM_THREADS = cpu_count()

        self.FEAUTRES = self.conf['features']
        self.FILTER_BANK = self.conf['filter_bank']
        self.FILTER_BANK_SIZE = self.conf['filter_bank_size']
        self.LOWER_FREQUENCY = self.conf['lower_frequency']
        self.HIGHER_FREQUENCY = self.conf['higher_frequency']
        self.VAD = self.conf['vad']
        self.SNR_RATIO = self.conf['snr_ratio'] if self.VAD=="snr" else None
        # cepstral coefficients
        self.WINDOW_SIZE = self.conf['window_size']
        self.WINDOW_SHIFT = self.conf['window_shift']
        self.CEPS_NUMBER = self.conf['cepstral_coefficients']
        # reset unnecessary ones based on given configuration
        self.review_member_variables()


    def review_member_variables(self):
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


    def extract_features(self, group):
        """
        This function computes the acoustic parameters of audio files insied 
        "self.audio_dir/group" save them to disk in a HDF5 format
        Args:
            group (String): the name of the group that we want to extract its
                featurs. It could be either 'data', 'enroll' or 'test'.
        """
        assert group in ["enroll", "test"],\
            "Invalid group name!! Choose either 'enroll', 'test'"
        in_files = os.listdir(os.path.join(self.audio_dir, group))
        feat_dir = os.path.join(self.feat_dir, group)
        # Feature extraction
        # lower_frequency: lower frequency (in Herz) of the filter bank
        # higher_frequency: higher frequency of the filter bank
        # filter_bank: type of fiter scale to use, can be "lin" or "log"
        #              (for linear of log-scale)
        # filter_bank_size: number of filters banks
        # window_size: size of the sliding window to process (in seconds)
        # shift: time shift of the sliding window (in seconds)
        # ceps_number: number of cepstral coefficients to extract
        # snr: signal to noise ratio used for "snr" vad algorithm
        # vad: Type of voice activity detection algorithm to use.
        #      It Can be "energy", "snr", "percentil" or "lbl".
        # save_param: list of strings that indicate which parameters to save. 
        # keep_all_features: boolean, if True, all frames are writen; if False,
        #                    keep only frames according to the vad label
        extractor = sidekit.FeaturesExtractor(
            audio_filename_structure=os.path.join(self.audio_dir, group, "{}"),
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
        # BUG: The following lines do the exact same thing as the previous
        # for-loop above, but with using multi-processing where 'num_thread' is
        # the number of parallel processes to run.
        # The following method is buggy and freezes after sometime. I don't
        # recommend using it, but you can give it a try:
        # extractor.save_list(show_list=show_list,
        #                     channel_list=channel_list,
        #                     num_thread=self.NUM_THREADS)





if __name__ == "__main__":
    conf_filename = "py3env/conf.yaml"
    ex = FeaturesExtractor(conf_filename)
    ex.extract_features("enroll")
    ex.extract_features("test")