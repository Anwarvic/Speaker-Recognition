import os
import sidekit
import numpy as np
from glob import glob
from multiprocessing import cpu_count
from glob import glob
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)

from model_interface import SidekitModel


class IVector(SidekitModel):
    """Identity Vectors"""
    
    def __init__(self):
        super().__init__()
        # Set parameters of your system
        self.NUM_GUASSIANS = 64  # number of Gaussian distributions for each GMM
        self.BATCH_SIZE = 30
        self.RANK_TV = 400  # Rank of the total variability matrix
        self.TV_ITERATIONS = 10  # number of iterations to run over the variability matrix
        #DON'T KNOW YET
        self.ENABLE_PLDA = True
    
    

    def __create_stats(self):
        # Read tv_idmap, and plda_idmap
        tv_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "tv_idmap.h5"))
        plda_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "plda_idmap.h5"))
        # Create a joint StatServer for TV and PLDA training data
        back_idmap = plda_idmap.merge(tv_idmap)
        if not back_idmap.validate():
            raise RuntimeError("Error merging tv_idmap & plda_idmap")
        
        # Load UBM
        model_name = "ubm_{}.h5".format(self.NUM_GUASSIANS)
        ubm = sidekit.Mixture()
        ubm.read(os.path.join(self.BASE_DIR, "ubm", model_name))
        back_stat = sidekit.StatServer( statserver_file_name=back_idmap, 
                                        ubm=ubm
                                      )
        # Create Feature Server
        fs = self.createFeatureServer()
        
        # Jointly compute the sufficient statistics of TV and PLDA data
        back_filename = 'back_stat_{}.h5'.format(self.NUM_GUASSIANS)
        if not os.path.isfile(os.path.join(self.BASE_DIR, "stat", back_filename)):
            #BUG: don't use self.NUM_THREADS when assgining num_thread as it's prune to race-conditioning
            back_stat.accumulate_stat( ubm=ubm,
                                    feature_server=fs,
                                    seg_indices=range(back_stat.segset.shape[0])
                                    )
            back_stat.write(os.path.join(self.BASE_DIR, "stat", back_filename))
        
        # Load the sufficient statistics from TV training data
        tv_filename = 'tv_stat_{}.h5'.format(self.NUM_GUASSIANS)
        if not os.path.isfile(os.path.join(self.BASE_DIR, "stat", tv_filename)):
            tv_stat = sidekit.StatServer.read_subset(os.path.join(self.BASE_DIR, "stat", back_filename),
                                                     tv_idmap
                                                    )
            tv_stat.write(os.path.join(self.BASE_DIR, "stat", tv_filename))
        
        # Load sufficient statistics and extract i-vectors from PLDA training data
        plda_filename = 'plda_stat_{}.h5'.format(self.NUM_GUASSIANS)
        if not os.path.isfile(os.path.join(self.BASE_DIR, "stat", plda_filename)):
            plda_stat = sidekit.StatServer.read_subset( os.path.join(self.BASE_DIR, "stat", back_filename),
                                                        plda_idmap
                                                    )
            plda_stat.write(os.path.join(self.BASE_DIR, "stat", plda_filename))
        
        # Load sufficient statistics from test data
        filename = 'test_stat_{}.h5'.format(self.NUM_GUASSIANS)
        if not os.path.isfile(os.path.join(self.BASE_DIR, "stat", filename)):
            test_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "test_idmap.h5"))
            test_stat = sidekit.StatServer( statserver_file_name=test_idmap, 
                                            ubm=ubm
                                        )
            # Create Feature Server
            fs = self.createFeatureServer()
            # Jointly compute the sufficient statistics of TV and PLDA data
            #BUG: don't use self.NUM_THREADS when assgining num_thread as it's prune to race-conditioning
            test_stat.accumulate_stat( ubm=ubm,
                                    feature_server=fs,
                                    seg_indices=range(test_stat.segset.shape[0])
                                    )
            test_stat.write(os.path.join(self.BASE_DIR, "stat", filename))



    def train_tv(self):
        """
        This method is used to train the Total Variability (TV) matrix
        and save it into 'ivector' directory !! 
        """
        # Create status servers
        self.__create_stats()

        # Load UBM model
        model_name = "ubm_{}.h5".format(self.NUM_GUASSIANS)
        ubm = sidekit.Mixture()
        ubm.read(os.path.join(self.BASE_DIR, "ubm", model_name))

        # Train TV matrix using FactorAnalyser
        filename = "tv_matrix_{}".format(self.NUM_GUASSIANS)
        outputPath = os.path.join(self.BASE_DIR, "ivector", filename)
        tv_filename = 'tv_stat_{}.h5'.format(self.NUM_GUASSIANS)
        fa = sidekit.FactorAnalyser()
        fa.total_variability_single(os.path.join(self.BASE_DIR, "stat", tv_filename),
                                    ubm,
                                    tv_rank=self.RANK_TV,
                                    nb_iter=self.TV_ITERATIONS,
                                    min_div=True,
                                    tv_init=None,
                                    batch_size=self.BATCH_SIZE,
                                    save_init=False,
                                    output_file_name=outputPath
                                   )
        # tv = fa.F # TV matrix
        # tv_mean = fa.mean # Mean vector
        # tv_sigma = fa.Sigma # Residual covariance matrix

        # Clear files produced at each iteration
        filename_regex = "tv_matrix_{}_it-*.h5".format(self.NUM_GUASSIANS)
        lst = glob(os.path.join(self.BASE_DIR, "ivector", filename_regex))
        for f in lst:
            os.remove(f)
    

    def evaluate(self):
        """
        This method is used to score our trained model. 
        """
        # Load UBM model
        model_name = "ubm_{}.h5".format(self.NUM_GUASSIANS)
        ubm = sidekit.Mixture()
        ubm.read(os.path.join(self.BASE_DIR, "ubm", model_name))

        # Load TV matrix
        filename = "tv_matrix_{}".format(self.NUM_GUASSIANS)
        outputPath = os.path.join(self.BASE_DIR, "ivector", filename)
        fa = sidekit.FactorAnalyser(outputPath+".h5")

        # Extract i-vectors from enrollment data
        logging.info("Extracting i-vectors from enrollment data")
        enroll_stat = sidekit.StatServer.read(os.path.join(self.BASE_DIR, 'stat', 'enroll_stat_32.h5'))
        enroll_iv = fa.extract_ivectors_single( ubm=ubm,
                                                stat_server=enroll_stat,
                                                uncertainty=False
                                              )
    
        # Extract i-vectors from test data
        logging.info("Extracting i-vectors from test data")
        test_stat = sidekit.StatServer.read(os.path.join(self.BASE_DIR, 'stat', 'test_stat.h5'))
        test_iv = fa.extract_ivectors_single(ubm=ubm,
                                             stat_server=test_stat,
                                             uncertainty=False
                                            )

        # Do cosine distance scoring and write results
        logging.info("Calculating cosine score")
        test_ndx = sidekit.Ndx.read(os.path.join(self.BASE_DIR, "task", "test_ndx.h5"))
        scores_cos = sidekit.iv_scoring.cosine_scoring( enroll_iv,
                                                        test_iv,
                                                        test_ndx,
                                                        wccn=None
                                                      )
        # Write scores
        filename = "ivector_scores_cos_{}.h5".format(self.NUM_GUASSIANS)
        scores_cos.write(os.path.join(self.BASE_DIR, "result", filename))
        
    # def _________________plda():
        # plda = os.path.join(self.BASE_DIR, "stat", "plda_stat")
        # # Load sufficient statistics and extract i-vectors from PLDA training data
        # plda_iv = fa.extract_ivectors(ubm=ubm,
        #                               stat_server_filename = plda,
        #                               batch_size=self.BATCH_SIZE,
        #                               num_thread=self.NUM_THREADS
        #                              )

    def getAccuracy(self):
        import h5py

        filename = "ivector_scores_cos_{}.h5".format(self.NUM_GUASSIANS)
        filepath = os.path.join(self.BASE_DIR, "result", filename)
        h5 = h5py.File(filepath, mode="r")
        modelset = list(h5["modelset"])
        segest = list(h5["segset"])
        scores = np.array(h5["scores"])
        
        #get Accuracy
        accuracy = super().getAccuracy(modelset, segest, scores, mode=2, threshold=0)
        return accuracy




if __name__ == "__main__":
    iv = IVector()
    iv.train_tv()
    # print( "Accuracy: {}%".format(iv.getAccuracy()*100) )
    # import h5py
    
    # filepath = "/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env/result/ivector_scores.h5"
    # h5 = h5py.File(filepath, mode="r")
    # modelset = list(h5.get("modelset"))
    # segset = list(h5.get("segset"))
    # scores = np.array(h5.get("scores"))
    # filename = "ivector_scores_explained_{}.txt".format(iv.NUM_GUASSIANS)
    # fout = open(os.path.join(iv.BASE_DIR, "result", filename), "a")
    # fout.truncate(0) #clear content
    # for seg_idx, seg in enumerate(segset):
    #     fout.write("Wav: {}\n".format(seg))
    #     for speaker_idx, speaker in enumerate(modelset):
    #         fout.write("\tSpeaker {}:\t{}\n".format(speaker, scores[speaker_idx, seg_idx]))
    #     fout.write("\n")
    # fout.close()