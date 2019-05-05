import os
import sidekit
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from multiprocessing import cpu_count
from utils import parse_yaml



class SidekitModel():

    def __init__(self, conf_filepath):
        self.conf = parse_yaml(conf_filepath)
        # use 0 to disable multi-processing
        self.NUM_THREADS = cpu_count()
        # The parent directory of the project
        self.BASE_DIR = self.conf['outpath']
    

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
        # keep_all_features: boolean, if True, keep all features; if False,
        #       keep frames according to the vad labels
        server = sidekit.FeaturesServer(
                feature_filename_structure=os.path.join(feat_dir, "{}.h5"),
                dataset_list=self.conf['features'],
                feat_norm="cmvn", #cepstral mean-variance normalization
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
    
    def getAccuracy(self, speakers, test_files, scores, mode=2, threshold=0):
        """
        This private method is used to get the accuracy of a model
        given five pieces of information:
        -> speakers: list of speakers
        -> test_files: list of filenames that used to evaluate model
        -> scores: score numpy matrix obtained by the model
        -> mode: which is one of these values [0, 1, 2] where:
            -> 0: means get verification accuracy.
            -> 1: means get recognition accuracy.
            -> 2: means get both accuracy, verification and recognition.
        -> threshold: the value above which we will consider the verification
                is done correctly. In other words, if the score>threshold, then
                the answer is considered; otherwise, the answer is not considered
        And it should return the accuracy of the model in percentage
        """
        assert mode in [0, 1, 2],\
            "The model variable must be one of these values[0, 1, 2]"
        assert scores.shape == (len(speakers), len(test_files)),\
            "The dimensions of the input don't match"
        accuracy = 0.
        speakers = [sp.decode() for sp in speakers]
        max_indices = np.argmax(scores, axis=0)
        max_scores = np.max(scores, axis=0)
        for idx, test_filename in enumerate(test_files):
            test_filename = test_filename.decode() #convert from byte to string
            actual_speaker = test_filename.split("/")[-1].split(".")[0]
            predicted_speaker = speakers[max_indices[idx]]
            #TODO: 
            ########## JUST VERIFICATION ##########
            if mode == 0:
                if max_scores[idx] < threshold:
                    continue
                else:
                    accuracy += 1.
            ########## JUST RECOGNITION ##########
            elif mode == 1:
                #skip speakers outside the training
                if actual_speaker not in speakers:
                    continue
                else:
                    if predicted_speaker == actual_speaker:
                        accuracy += 1.
            ########## VERIFICATION & RECOGNITION ##########
            elif mode == 2:
                #skip speakers outside the training
                if max_scores[idx] < threshold:
                    continue
                else:
                    if predicted_speaker == actual_speaker:
                        accuracy += 1.

        return accuracy/len(test_files)
