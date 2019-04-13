import os
import sidekit
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)

from model_interface import SidekitModel



class UBM(SidekitModel):
    """Universal Background Model"""
    
    def __init__(self):
        super().__init__()
        # Number of Guassian Distributions
        self.NUM_GUASSIANS = 16


    def train(self, SAVE=True):
        #SEE: https://projets-lium.univ-lemans.fr/sidekit/tutorial/ubmTraining.html
        train_list = os.listdir(os.path.join(self.BASE_DIR, "audio", "enroll"))
        for i in range(len(train_list)):
            train_list[i] = train_list[i].split(".h5")[0]
        server = self.createFeatureServer("enroll")
        logging.info("Training...")
        ubm = sidekit.Mixture()
        # Set the model name
        ubm.name = "ubm_{}.h5".format(self.NUM_GUASSIANS) 
        # Expectation-Maximization estimation of the Mixture parameters.
        ubm.EM_split(features_server=server, #sidekit.FeaturesServer used to load data
                     feature_list=train_list, # list of feature files to train the model
                     distrib_nb=self.NUM_GUASSIANS, # final number of Gaussian distributions
                     iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8), # list of iteration number for each step of the learning process
                     num_thread=self.NUM_THREADS, # number of thread to launch for parallel computing
                     save_partial=False # if False, it only saves the last model
                    )
        # -> 2 iterations of EM with 2   distributions
        # -> 2 iterations of EM with 4   distributions
        # -> 4 iterations of EM with 8   distributions
        # -> 4 iterations of EM with 16  distributions
        # -> 4 iterations of EM with 32  distributions
        # -> 4 iterations of EM with 64  distributions
        # -> 8 iterations of EM with 128 distributions
        # -> 8 iterations of EM with 256 distributions
        # -> 8 iterations of EM with 512 distributions
        # -> 8 iterations of EM with 1024 distributions
        model_dir = os.path.join(self.BASE_DIR, "ubm")
        logging.info("Saving the model {} at {}".format(ubm.name, model_dir))
        ubm.write(os.path.join(model_dir, ubm.name))

        # Read idmap for the enrolling data
        enroll_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "enroll_idmap.h5"))
        # Create Statistic Server to store/process the enrollment data
        enroll_stat = sidekit.StatServer(statserver_file_name=enroll_idmap,
                                         ubm=ubm)
        logging.debug(enroll_stat)

        # Compute the sufficient statistics for a list of sessions whose indices are segIndices.
        server.feature_filename_structure = os.path.join(self.BASE_DIR, "feat", "{}.h5")
        #BUG: don't use self.NUM_THREADS when assgining num_thread as it's prune to race-conditioning
        enroll_stat.accumulate_stat(ubm=ubm,
                                    feature_server=server,
                                    seg_indices=range(enroll_stat.segset.shape[0])
                                   )
        if SAVE:
            # Save the status of the enroll data
            filename = "enroll_stat_{}.h5".format(self.NUM_GUASSIANS)
            enroll_stat.write(os.path.join(self.BASE_DIR, "stat", filename))



    def evaluate(self, explain=True):
        ############################# READING ############################
        # Create Feature server
        server = self.createFeatureServer()
        # Read the index for the test datas
        test_ndx = sidekit.Ndx.read(os.path.join(self.BASE_DIR, "task", "test_ndx.h5"))
        # Read the UBM model
        ubm = sidekit.Mixture()
        model_name = "ubm_{}.h5".format(self.NUM_GUASSIANS)
        ubm.read(os.path.join(self.BASE_DIR, "ubm", model_name))

        ############################ Evaluating ###########################
        filename = "enroll_stat_{}.h5".format(self.NUM_GUASSIANS)
        enroll_stat = sidekit.StatServer.read(os.path.join(self.BASE_DIR, "stat", filename))
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
        filename = "ubm_scores_{}.h5".format(self.NUM_GUASSIANS)
        scores_gmm_ubm.write(os.path.join(self.BASE_DIR, "result", filename))
        
        #write Analysis
        if explain:
            filename = "ubm_scores_explained_{}.txt".format(self.NUM_GUASSIANS)
            fout = open(os.path.join(self.BASE_DIR, "result", filename), "a")
            fout.truncate(0) #clear content
            modelset = list(scores_gmm_ubm.modelset)
            segset = list(scores_gmm_ubm.segset)
            scores = np.array(scores_gmm_ubm.scoremat)
            for seg_idx, seg in enumerate(segset):
                fout.write("Wav: {}\n".format(seg))
                for speaker_idx, speaker in enumerate(modelset):
                    fout.write("\tSpeaker {}:\t{}\n".format(speaker, scores[speaker_idx, seg_idx]))
                fout.write("\n")
            fout.close()



    def plotDETcurve(self):
        # Read test scores
        filename = "ubm_scores_{}.h5".format(self.NUM_GUASSIANS)
        scores_dir = os.path.join(self.BASE_DIR, "result", filename)
        scores_gmm_ubm = sidekit.Scores.read(scores_dir)
        # Read the key
        key = sidekit.Key.read_txt(os.path.join(self.BASE_DIR, "task", "test_trials.txt"))

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
        dp.__figure__.savefig(os.path.join(self.BASE_DIR, "result", graphname))


    def getAccuracy(self, result_mode):
        """
        This function is used to get the accuracy of the model. 
        It reads the "test_scores_{}.h5" file that we got  using the 
        evaluate() method where {} is the number of Gaussians
        used in the model. For example, if the number of Gaussian 
        distributions is 32, then the file read will be "test_scores_32.h5",
        This method should return the Accuracy of the model in percentage.
        """
        assert result_mode in [0, 1, 2], "Accuracy mode must be either 0, 1 or 2!!"
        import h5py

        filename = "ubm_scores_{}.h5".format(self.NUM_GUASSIANS)
        filepath = os.path.join(self.BASE_DIR, "result", filename)
        h5 = h5py.File(filepath, mode="r")
        modelset = list(h5["modelset"])
        segest = list(h5["segset"])
        scores = np.array(h5["scores"])
        
        #get Accuracy
        accuracy = super().getAccuracy(modelset, segest, scores, mode=result_mode, threshold=0)
        return accuracy



if __name__ == "__main__":
    ubm = UBM()
    ubm.train()
    ubm.evaluate()
    ubm.plotDETcurve()
    for mode in [0, 1, 2]:
        print( "Accuracy: {}%".format(ubm.getAccuracy(mode)*100) )