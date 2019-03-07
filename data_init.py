import os
import sidekit
import numpy as np
from tqdm import tqdm




class Initializer():
    """
    This class if for structure the data (training/test) into h5 files
    that will be used later for training and evaluating our models
    
    #NOTE:All outputs of this script can be found in the directory TASK_DIR
    """

    def __init__(self):
        BASE_DIR = "/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env"
        self.AUDIO_DIR = os.path.join(BASE_DIR, "audio")
        self.TASK_DIR = os.path.join(BASE_DIR, "task")
        self.enrolled_speakers = self.__get_speakers()


    def __get_speakers(self):
        """
        This private method is supposed to return the unique speakers' IDs
        who are enrolled in the training data.
        """
        enroll_dir = os.path.join(self.AUDIO_DIR, "enroll") # enrollment data directory
        enroll_files = os.listdir(enroll_dir)
        enroll_models = [files.split('_')[0] for files in enroll_files] # list of model IDs
        return sorted(set(enroll_models))


    def create_idMap(self, group):
        """
        IdMap are used to store two lists of strings and to map between them.
        Most of the time, IdMap are used to associate names of segments (sessions)
        stored in leftids; with the ID of their class (that could be a speaker ID)
        stored in rightids.
        NOTE: Duplicated entries are allowed in each list.
        Additionally, and in order to allow more flexibility, IdMap includes two other vectors:
        'start'and 'stop' which are vectors of floats and can be used to store boudaries of
        audio segments.
        An IdMap object is often used to store together: speaker IDs, segment IDs,
        start and stop time of the segment and to initialize a StatServer.
        """
        # Make enrollment (IdMap) file list
        group_dir = os.path.join(self.AUDIO_DIR, group) # enrollment data directory
        group_files = os.listdir(group_dir)
        group_models = [files.split('_')[0] for files in group_files] # list of model IDs
        group_segments = [group+"/"+f for f in group_files]
        
        # Generate IdMap
        group_idmap = sidekit.IdMap()
        group_idmap.leftids = np.asarray(group_models)
        group_idmap.rightids = np.asarray(group_segments)
        group_idmap.start = np.empty(group_idmap.rightids.shape, '|O')
        group_idmap.stop = np.empty(group_idmap.rightids.shape, '|O')
        if group_idmap.validate():
            #TODO: possibily adding tv_idmap.h5 and plda_idmap.h5
            group_idmap.write(os.path.join(self.TASK_DIR, group+'_idmap.h5'))
        else:
            raise RuntimeError('Problems with creating idMap file')


    def create_test_trials(self):
        """
        Ndx objects store trials index information, i.e., combination of 
        model and segment IDs that should be evaluated by the system which 
        will produce a score for those trials.

        The trialmask is a m-by-n matrix of boolean where m is the number of
        unique models and n is the number of unique segments. If trialmask(i,j)
        is true then the score between model i and segment j will be computed.
        """
        # Make list of test segments
        test_data_dir = os.path.join(self.AUDIO_DIR, "test") # test data directory
        test_files = os.listdir(test_data_dir)
        test_files = ["test/"+f for f in test_files]

        # Make lists for trial definition, and write to file
        test_models = []
        test_segments = []
        test_labels = []

        for model in tqdm(self.enrolled_speakers, desc="Processing Enrolled-speakers"):
            for segment in sorted(test_files):
                test_model = segment.split("_")[0].split("/")[-1]
                test_models.append(model)
                test_segments.append(segment)
                # Compare gender and speaker ID for each test file
                if test_model == model:
                    test_labels.append('target')
                else:
                    test_labels.append('nontarget')
            
        with open(os.path.join(self.TASK_DIR, "test_trials.txt"), "w") as fh:
            for i in range(len(test_models)):
                fh.write(test_models[i]+' '+test_segments[i]+' '+test_labels[i]+'\n')


    def create_Ndx(self):
        """
        Key are used to store information about which trial is a target trial
        and which one is a non-target (or impostor) trial. tar(i,j) is true
        if the test between model i and segment j is target. non(i,j) is true
        if the test between model i and segment j is non-target.
        """
        # Define Key and Ndx from text file
        # SEE: https://projets-lium.univ-lemans.fr/sidekit/_modules/sidekit/bosaris/key.html
        key = sidekit.Key.read_txt(os.path.join(self.TASK_DIR, "test_trials.txt"))
        ndx = key.to_ndx()
        if ndx.validate():
            ndx.write(os.path.join(self.TASK_DIR, 'test_ndx.h5'))
        else:
            raise RuntimeError('Problems with creating idMap file')


    def structure(self):
        """
        This is the main method for this class, it calls all previous
        methods... that's basically what it does :)
        """
        self.create_idMap("enroll")
        self.create_idMap("test")
        self.create_test_trials()
        self.create_Ndx()
        print("DONE!!")




if __name__ == "__main__":
    init = Initializer()
    init.structure()