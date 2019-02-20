import os
import subprocess
import sidekit
import numpy as np
from tqdm import tqdm
from utils import *
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")



class SpeakerRecognizer():

    def __init__(self):
        ############ Global Variables ###########
        self.SAMPLE_RATE = 44100
        self.NUM_CHANNELS = 2
        self.PRECISION = 16 #I mean 16-bit
        self.NUM_THREADS = mp.cpu_count() #(4 default)
        # Number of Guassian Distributions
        self.NUM_GUASSIANS = 32
        ############ Dirs #########
        self.base_dir = "/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env"
        self.input_dir = os.path.join(self.base_dir, "data")
        self.all_files = os.listdir(self.input_dir)
        self.wav_dir = os.path.join(self.base_dir, "wav")
        self.feat_dir = os.path.join(self.base_dir, "feat")
        self.ubm_dir = os.path.join(self.base_dir, "ubm")
        self.test_dir = os.path.join(self.base_dir, "test")


    

    def extractFeatures(self):
        if not os.path.exists(self.wav_dir):
            raise Exception("Couldn't extract features, apply preprocess method first!!") 

        # Feature extraction
        extractor = sidekit.FeaturesExtractor(audio_filename_structure=os.path.join(self.wav_dir, "{}"),
                                              feature_filename_structure=os.path.join(self.feat_dir, "{}.h5"),
                                              sampling_frequency=self.SAMPLE_RATE,
                                              lower_frequency=200,
                                              higher_frequency=3800,
                                              filter_bank="log",
                                              filter_bank_size=24,
                                              window_size=0.04,
                                              shift=0.01,
                                              ceps_number=20,
                                              vad="snr",
                                              snr=40,
                                              pre_emphasis=0.97,
                                              save_param=["vad", "energy", "cep", "fb"],
                                              keep_all_features=True)

        # Prepare file lists
        show_list = np.unique(np.hstack([self.all_files]))
        channel_list = np.zeros_like(show_list, dtype = int)
        
        # save the features in feat_dir
        extractor.save_list(show_list=show_list,
                            channel_list=channel_list,
                            num_thread=self.NUM_THREADS)
    


    def train(self, SAVE_FLAG=True):
        #Universal Background Model Training
        #NOTE: https://projets-lium.univ-lemans.fr/sidekit/tutorial/ubmTraining.html
        ubm_list = os.listdir(os.path.join(self.base_dir, "feat"))
        for i in range(len(ubm_list)):
            ubm_list[i] = ubm_list[i].split(".h5")[0]

        server = sidekit.FeaturesServer(feature_filename_structure=os.path.join(self.feat_dir, "{}.h5"),
                                        sources=None,
                                        dataset_list=["vad", "energy", "cep", "fb"],
                                        feat_norm="cmvn",
                                        global_cmvn=None,
                                        dct_pca=False,
                                        dct_pca_config=None,
                                        sdc=False,
                                        sdc_config=None,
                                        delta=True,
                                        double_delta=True,
                                        delta_filter=None,
                                        context=None,
                                        traps_dct_nb=None,
                                        rasta=True,
                                        keep_all_features=True)

        print("Training...")
        ubm = sidekit.Mixture()
        ubm.EM_split(features_server=server,
                     feature_list=ubm_list,
                     distrib_nb=self.NUM_GUASSIANS,
                     iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
                     num_thread=self.NUM_THREADS,
                     save_partial=True,
                     ceil_cov=10,
                     floor_cov=1e-2
                     )
        if SAVE_FLAG:
            modelname = "ubm_{}.h5".format(self.NUM_GUASSIANS)
            print("Saving the model {} at {}".format(modelname, self.ubm_dir))
            ubm.write(os.path.join(self.ubm_dir, modelname))


    def evaluate(self):
        pass




if __name__ == "__main__":
    ubm = SpeakerRecognizer()
    ubm.preprocess()
    ubm.extractFeatures()
    ubm.train()