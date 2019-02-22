import os
import sidekit
import numpy as np
from tqdm import tqdm




#NOTE:All outputs of this file can be found in the directory TASK_DIR



######################## GLOBAL VARIABLES ########################
INDIR = "/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env"
TASK_DIR = os.path.join(INDIR, "task")
if not TASK_DIR:
    os.mkdir(TASK_DIR)
##################################################################

def create_idMap(save_flag=True):
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
    # Make ubm file list
    # data_dir = os.path.join(INDIR, "audio", "data") # data directory
    # data_files = os.listdir(data_dir)
    # # Remove extension and add path prefix
    # data_files = ["data/"+f for f in data_files]

    # Make enrollment (IdMap) file list
    enroll_dir = os.path.join(INDIR, "audio", "enroll") # enrollment data directory
    enroll_files = os.listdir(enroll_dir)
    enroll_models = [files.split('_')[0] for files in enroll_files] # list of model IDs
    enroll_segments = ["enroll/"+f for f in enroll_files]
    
    # Generate IdMap
    enroll_idmap = sidekit.IdMap()
    enroll_idmap.leftids = np.asarray(enroll_models)
    enroll_idmap.rightids = np.asarray(enroll_segments)
    enroll_idmap.start = np.empty(enroll_idmap.rightids.shape, '|O')
    enroll_idmap.stop = np.empty(enroll_idmap.rightids.shape, '|O')
    if enroll_idmap.validate():
        if save_flag:
            enroll_idmap.write(os.path.join(TASK_DIR, 'idmap_enroll.h5'))
        return enroll_models
    else:
         raise RuntimeError('Problems with creating idMap file')




def create_test_trials(enroll_models):
    """
    Ndx objects store trials index information, i.e., combination of 
    model and segment IDs that should be evaluated by the system which 
    will produce a score for those trials.

    The trialmask is a m-by-n matrix of boolean where m is the number of
    unique models and n is the number of unique segments. If trialmask(i,j)
    is true then the score between model i and segment j will be computed.
    """
    # Make list of test segments
    test_data_dir = os.path.join(INDIR, "audio", "test") # test data directory
    test_files = os.listdir(test_data_dir)
    test_files = ["test/"+f for f in test_files]

    # Make lists for trial definition, and write to file
    test_models = []
    test_segments = []
    test_labels = []

    for model in tqdm(enroll_models, desc="Processing enrolled-speakers"):
        for segment in sorted(test_files):
            test_model = segment.split("_")[0].split("/")[-1]
            test_models.append(test_model)
            test_segments.append(segment)
            # Compare gender and speaker ID for each test file
            if test_model == model:
                test_labels.append('target')
            else:
                test_labels.append('nontarget')
        
    with open(os.path.join(TASK_DIR, "test_trials.txt"), "w") as fh:
        for i in range(len(test_models)):
            fh.write(test_models[i]+' '+test_segments[i]+' '+test_labels[i]+'\n')



def create_Ndx(save_flag = True):
    """
    Key are used to store information about which trial is a target trial
    and which one is a non-target (or impostor) trial. tar(i,j) is true
    if the test between model i and segment j is target. non(i,j) is true
    if the test between model i and segment j is non-target.
    """
    # Define Key and Ndx from text file
    # SEE: https://projets-lium.univ-lemans.fr/sidekit/_modules/sidekit/bosaris/key.html
    key = sidekit.Key.read_txt(os.path.join(TASK_DIR, "test_trials.txt"))
    
    ndx = key.to_ndx()
    if save_flag:
        ndx.write(os.path.join(TASK_DIR, 'test_ndx.h5'))


if __name__ == "__main__":
    enroll_speakers = set(create_idMap()) #unique speaker IDs
    create_test_trials(sorted(list(enroll_speakers)))
    create_Ndx()