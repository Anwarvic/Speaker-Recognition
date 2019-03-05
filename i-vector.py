import os
import sidekit
import numpy as np
from multiprocessing import cpu_count
from glob import glob
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)



class IVector():
    
    def __init__(self):
        ############ Global Variables ###########
        # use 0 to disable multi-processing
        self.NUM_THREADS = cpu_count()
        # The parent directory of the project
        self.BASE_DIR = "/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env"
        # Set parameters of your system
        self.NUM_GUASSIANS = 32  # number of Gaussian distributions for each GMM
        self.BATCH_SIZE = 30
        self.RANK_TV = 400  # Rank of the total variability matrix
        self.TV_ITERATIONS = 10  # number of iterations to run over the variability matrix
        # self.FEAUTURE_EXTENTION = 'h5'  # Extension of the feature files
    
    
    def __createFeatureServer(self, group=None):
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


    def data_init(self):
        # Read tv_idmap, and plda_idmap
        tv_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "idmap_tv.h5"))
        plda_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "idmap_plda.h5"))
        # Load UBM
        ubm = sidekit.Mixture()
        model_name = "ubm_{}.h5".format(self.NUM_GUASSIANS)
        ubm.read(os.path.join(self.BASE_DIR, "ubm", model_name))
        # Create Feature Server
        fs = self.__createFeatureServer()

        # Create a joint StatServer for TV and PLDA training data
        back_idmap = plda_idmap.merge(tv_idmap)
        if not back_idmap.validate():
            logging.warning("Error merging tv_idmap & plda_idmap")
            return
        back_stat = sidekit.StatServer( statserver_file_name=back_idmap, 
                                        ubm=ubm
                                      )
        # Jointly compute the sufficient statistics of TV and PLDA data
        #BUG: don't use self.NUM_THREADS when assgining num_thread as it's prune to race-conditioning
        back_stat.accumulate_stat( ubm=ubm,
                                   feature_server=fs,
                                   seg_indices=range(back_stat.segset.shape[0])
                                   )
        back_stat.write(os.path.join(self.BASE_DIR, "task", 'stat_back.h5'))
        # Load the sufficient statistics from TV training data
        tv_stat = sidekit.StatServer.read_subset(os.path.join(self.BASE_DIR, "task", 'stat_back.h5'), tv_idmap)
        tv_stat.write(os.path.join(self.BASE_DIR, "task", 'tv_stat.h5'))
        # Train TV matrix using FactorAnalyser
        filename = "tv_matrix_{}".format(self.NUM_GUASSIANS)
        outputPath = os.path.join(self.BASE_DIR, "ivector", filename)
        fa = sidekit.FactorAnalyser()
        fa.total_variability_single(os.path.join(self.BASE_DIR, "task", 'tv_stat.h5'),
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



if __name__ == "__main__":
    iv = IVector()
    iv.data_init()
    # iv.play()