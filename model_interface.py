import os
import sidekit
from multiprocessing import cpu_count
import logging
logging.basicConfig(level=logging.INFO)



class SidekitModel():

    def __init__(self):
        ############ Global Variables ###########
        # use 0 to disable multi-processing
        self.NUM_THREADS = cpu_count()
        # Number of Guassian Distributions
        self.NUM_GUASSIANS = 128
        # The parent directory of the project
        self.BASE_DIR = "/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env"
    

    def createFeatureServer(self, group=None):
        if group:
            feat_dir = os.path.join(self.BASE_DIR, "feat", group)
        else:
            feat_dir = os.path.join(self.BASE_DIR, "feat")
        # feature_filename_structure: structure of the filename to use to load HDF5 files
        # dataset_list: string of the form ["cep", "fb", vad", energy", "bnf"]
        # feat_norm: type of normalization to apply as post-processing
        # delta: if True, append the first order derivative
        # double_delta: if True, append the second order derivative
        # rasta: if True, perform RASTA filtering
        # keep_all_features: boolean, if True, keep all features, if False, keep frames according to the vad labels
        server = sidekit.FeaturesServer(feature_filename_structure=os.path.join(feat_dir, "{}.h5"),
                                        dataset_list=["vad", "energy", "cep", "fb"],
                                        feat_norm="cmvn",
                                        delta=True,
                                        double_delta=True,
                                        rasta=True,
                                        keep_all_features=True)
        logging.info("Feature-Server is created")
        logging.debug(server)
        return server
    
    def train(self):
        pass
    
    def evaluate(self):
        pass
    
    def plotDETcurve(self):
        pass