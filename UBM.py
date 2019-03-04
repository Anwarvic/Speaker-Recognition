import os
import subprocess
import sidekit
import numpy as np
from tqdm import tqdm
from utils import preprocessAudioFile
import multiprocessing as mp
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)




class SpeakerRecognizer():
    
    def __init__(self):
        ############ Global Variables ###########
        # use 0 to disable multi-processing
        self.NUM_THREADS = mp.cpu_count()
        # Number of Guassian Distributions
        self.NUM_GUASSIANS = 128
        # The parent directory of the project
        self.base_dir = "/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env"
    

    def __createFeatureServer(self, group=None):
        if group:
            feat_dir = os.path.join(self.base_dir, "feat", group)
        else:
            feat_dir = os.path.join(self.base_dir, "feat")
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


    def train(self, SAVE_FLAG=True):
        #Universal Background Model Training
        #SEE: https://projets-lium.univ-lemans.fr/sidekit/tutorial/ubmTraining.html
        train_list = os.listdir(os.path.join(self.base_dir, "audio", "enroll"))
        for i in range(len(train_list)):
            train_list[i] = train_list[i].split(".h5")[0]
        server = self.__createFeatureServer("enroll")
        logging.info("Training...")
        ubm = sidekit.Mixture()
        # Expectation-Maximization estimation of the Mixture parameters.
        ubm.EM_split(features_server=server, #sidekit.FeaturesServer used to load data
                     feature_list=train_list, # list of feature files to train the model
                     distrib_nb=self.NUM_GUASSIANS, # final number of Gaussian distributions
                     iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8), # list of iteration number for each step of the learning process
                     num_thread=self.NUM_THREADS, # number of thread to launch for parallel computing
                     save_partial=False # if False, it only saves the last model
                    )
        # -> 1 iteration of EM with 1 distribution
        # -> 2 iterations of EM with 2 distributions
        # -> 2 iterations of EM with 4 distributions
        # -> 4 iterations of EM with 8 distributions
        # -> 4 iterations of EM with 16 distributions
        # -> 4 iterations of EM with 32 distributions
        # -> 4 iterations of EM with 64 distributions
        # -> 8 iterations of EM with 128 distributions
        # -> 8 iterations of EM with 256 distributions
        # -> 8 iterations of EM with 512 distributions
        # -> 8 iterations of EM with 1024 distributions
        model_dir = os.path.join(self.base_dir, "ubm")
        if SAVE_FLAG:
            modelname = "ubm_{}.h5".format(self.NUM_GUASSIANS)
            logging.info("Saving the model {} at {}".format(modelname, model_dir))
            ubm.write(os.path.join(model_dir, modelname))
        # Read idmap for the enrolling data
        enroll_idmap = sidekit.IdMap.read(os.path.join(self.base_dir, "task", "idmap_enroll.h5"))
        # Create Statistic Server to store/process the enrollment data
        enroll_stat = sidekit.StatServer(statserver_file_name=enroll_idmap,
                                         ubm=ubm)
        logging.debug(enroll_stat)

        # Compute the sufficient statistics for a list of sessions whose indices are segIndices.
        server.feature_filename_structure = os.path.join(self.base_dir, "feat", "{}.h5")
        #BUG: don't use self.NUM_THREADS when assgining num_thread as it's prune to race-conditioning
        enroll_stat.accumulate_stat(ubm=ubm,
                                    feature_server=server,
                                    seg_indices=range(enroll_stat.segset.shape[0])
                                   )
        if SAVE_FLAG:
            # Save the status of the enroll data
            filename = "enroll_stat_{}.h5".format(self.NUM_GUASSIANS)
            enroll_stat.write(os.path.join(self.base_dir, "ubm", filename))



    def evaluate(self, explain=True):
        ############################# READING ############################
        # Create Feature server
        server = self.__createFeatureServer()
        # Read the index for the test datas
        test_ndx = sidekit.Ndx.read(os.path.join(self.base_dir, "task", "test_ndx.h5"))
        # Read the UBM model
        ubm = sidekit.Mixture()
        model_name = "ubm_{}.h5".format(self.NUM_GUASSIANS)
        ubm.read(os.path.join(self.base_dir, "ubm", model_name))

        ############################ Evaluating ###########################
        filename = "enroll_stat_{}.h5".format(self.NUM_GUASSIANS)
        enroll_stat = sidekit.StatServer.read(os.path.join(self.base_dir, "ubm", filename))
        # MAP adaptation of enrollment speaker models
        enroll_sv = enroll_stat.adapt_mean_map_multisession(ubm=ubm,
                                                            r=3 # MAP regulation factor
                                                           )
        # Compute scores
        scores_gmm_ubm = sidekit.gmm_scoring(ubm=ubm,
                                             enroll=enroll_sv,
                                             ndx=test_ndx,
                                             feature_server=server,
                                             num_thread=self.NUM_THREADS
                                            )
        # Save the model's Score object
        filename = "test_scores_{}.h5".format(self.NUM_GUASSIANS)
        scores_gmm_ubm.write(os.path.join(self.base_dir, "result", filename))
        
        #write Analysis
        if explain:
            filename = "test_scores_explained_{}.txt".format(self.NUM_GUASSIANS)
            fout = open(os.path.join(self.base_dir, "result", filename), "a")
            fout.truncate(0) #clear content
            modelset = list(scores_gmm_ubm.modelset)
            segest = list(scores_gmm_ubm.segset)
            scores = np.array(scores_gmm_ubm.scoremat)
            for seg_idx, seg in enumerate(segest):
                fout.write("Wav: {}\n".format(seg))
                for speaker_idx, speaker in enumerate(modelset):
                    fout.write("\tSpeaker {}:\t{}\n".format(speaker, scores[speaker_idx, seg_idx]))
                fout.write("\n")
            fout.close()



    def plotDETcurve(self):
        # Read test scores
        filename = "test_scores_{}.h5".format(self.NUM_GUASSIANS)
        scores_dir = os.path.join(self.base_dir, "result", filename)
        scores_gmm_ubm = sidekit.Scores.read(scores_dir)
        # Read the key
        key = sidekit.Key.read_txt(os.path.join(self.base_dir, "task", "test_trials.txt"))

        # Make DET plot
        logging.info("Drawing DET Curve")
        dp = sidekit.DetPlot(window_style='sre10', plot_title='Scores GMM-UBM')
        dp.set_system_from_scores(scores_gmm_ubm, key, sys_name='GMM-UBM')
        dp.create_figure()
        dp.plot_rocch_det(0)
        dp.plot_DR30_both(idx=0)
        prior = sidekit.logit_effective_prior(0.01, 10, 1)
        dp.plot_mindcf_point(prior, idx=0)
        graphname = "DET_GMM_UBM_{}.png".format(self.NUM_GUASSIANS)
        dp.__figure__.savefig(os.path.join(self.base_dir, "result", graphname))



    def __getAccuracy(self, speakers, test_files, scores, mode=2, threshold=0):
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
            actual_speaker = test_filename.split("/")[-1].split("_")[0]
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



    def getAccuracy(self):
        """
        This function is used to get the accuracy of the model. 
        It reads the "test_scores_{}.h5" file that we got  using the 
        evaluate() method where {} is the number of Gaussians
        used in the model. For example, if the number of Gaussian 
        distributions is 32, then the file read will be "test_scores_32.h5",
        This method should return the Accuracy of the model in percentage.
        """
        import h5py

        filename = "test_scores_{}.h5".format(self.NUM_GUASSIANS)
        filepath = os.path.join(self.base_dir, "result", filename)
        h5 = h5py.File(filepath, mode="r")
        modelset = list(h5["modelset"])
        segest = list(h5["segset"])
        scores = np.array(h5["scores"])
        
        #get Accuracy
        accuracy = self.__getAccuracy(modelset, segest, scores, mode=2, threshold=0)
        return accuracy



if __name__ == "__main__":
    ubm = SpeakerRecognizer()
    ubm.train()
    ubm.evaluate()
    ubm.plotDETcurve()
    # ubm.NUM_GUASSIANS = 64
    print( "Accuracy: {}%".format(ubm.getAccuracy()) )