import os
import random
import shutil
from glob import glob
from tqdm import tqdm
from utils import preprocessAudioFile

######################## NOTE ########################
"""
To build speaker verification model, one needs speech data
from each speaker that is to be known by the system.
The set of known speakers are in speaker recognition known
as the (enrollment speakers), and a speaker is enrolled into the
system when enrollment data from the speaker is processed to build
its model.

After the enrollment process, the performance of the speaker 
verification system can be evaluated using test data, which in
an open set scenario, will consist of data from speakers in and
outside the enrollment set. The set of all speakers involved in
testing the system will be referred to as the test speakers
"""
##################################################################


######################## GLOBAL VARIABLES ########################
#ids of speakers to be excluded
EXCLUDED_IDS = {11, 36, 44}

#ids of words to be excluded
EXCLUDED_WORDS = set([])
ENROLL_NUM = 10
TEST_NUM = 5
assert ENROLL_NUM+TEST_NUM <= 50,\
    """The sum(ENROLL_NUM, TEST_NUM) can't be > 50
    which is the maximum number of speakers in the corpus"""

ENROLL_REPS = 3
TEST_REPS = 7
assert ENROLL_REPS+TEST_REPS <= 10,\
    """The sum(ENROLL_REPS, TEST_REPS) MUST be <= 10
    which is the maximum number of reps"""
##################################################################

def splitSpeakers(enroll_num, test_num):
    """
    This function is used to split the speaker IDs to two sets
    (enroll, test) basted on the given ration after filtering some
    IDs which are defined inside EXCLUDED_IDS which are the female.
    
    NOTE: The number of speakers in the Arabic Corpus of Isolated Words
    are 50 whose IDs vary from [1-50]
    """
    # assert ratio >=0 and ratio <=1, "Given ratio MUST be between 0 and 1"
    random.seed(0)
    total_ids = set(range(1, 51)) #IDs from 1 to 50
    #exclude the EXCLUDED_IDS
    remaining_ids = list(total_ids - EXCLUDED_IDS)
    #shuffle
    random.shuffle(remaining_ids)
    enrollIDs = remaining_ids[0 : enroll_num]
    testIDs = remaining_ids[enroll_num : enroll_num+test_num]
    return enrollIDs, testIDs


def copyFiles(inDir, outDir, gender, speakerID, start, end):
    """
    This function lists all the speakers in inDir directory,
    then split these speakers based on enrollIDs and testIDs
    and copy them into outDir.
    While copying, we need to consider these:
    -> Exclude the EXCLUDED_WORDS
    -> Use the ENROLL_REPS, TEST_REPS
    -> Maintain the structure of wav filenames which is
       <gender><speaker>_<session>_<sentence>.wav
    -> Maintain the wav aspects which are:
       sample_rate = 16000
       n_channels = 1
       precision = 16-bit (PCM)
    """
    #iterate over reps
    for rep in range(start, end):
        rep = str(rep).zfill(2) #1 -> 01
        #iterate over words
        for wordID in set(range(1, 21)) - EXCLUDED_WORDS:
            wordID = str(wordID).zfill(2)
            wavFilename = "{}.{}.{}.wav".format(speakerID, rep, wordID)
            newWavFilename = "{}_{}_{}.wav".format(gender+speakerID, rep, wordID)
            inWav = os.path.join(inDir, speakerID, wavFilename)
            outWav = os.path.join(outDir, newWavFilename)
            preprocessAudioFile(inWav, outWav, sample_rate=16000, n_channels=1, bit=16)




def copyData(inDir, outDir, enrollIDs, testIDs):
    """
    Copy the Arabic Corpus of Isolated Words into their
    associated directory. The whole data data will be in 'data'
    directory, the enrolled data will be in 'enroll', and the
    test data will be in 'test'.
    """
    gender = "M" #since they're all male
    ##################################################################
    ############################## Data #############################
    ##################################################################
    #iterate over speakers
    dataIDs = set(range(1, 51)) - EXCLUDED_IDS
    data_outDir = os.path.join(outDir, "data")
    if not os.path.exists(data_outDir):
        os.mkdir(data_outDir)
    for speakerID in tqdm(dataIDs, desc="Whole Data"):
        speakerID = "S"+str(speakerID).zfill(2)
        start = 1
        end = 10+1
        copyFiles(inDir, data_outDir, gender, speakerID, start, end)
    ##################################################################
    ############################# ENROLL #############################
    ##################################################################
    #iterate over speakers
    enroll_outDir = os.path.join(outDir, "enroll")
    if not os.path.exists(enroll_outDir):
        os.mkdir(enroll_outDir)
    for speakerID in tqdm(enrollIDs, desc="Enroll Speakers"):
        speakerID = "S"+str(speakerID).zfill(2)
        start = 1
        end = ENROLL_REPS+1
        copyFiles(inDir, enroll_outDir, gender, speakerID, start, end)
    ##################################################################
    ############################## TEST ##############################
    ##################################################################
    #iterate over speakers
    # NOTE: speakers here are enrolled speakers + other speakers
    test_outDir = os.path.join(outDir, "test")
    if not os.path.exists(test_outDir):
        os.mkdir(test_outDir)
    for speakerID in tqdm(enrollIDs+testIDs, desc="Testing Files"):
        speakerID = "S"+str(speakerID).zfill(2)
        start = ENROLL_REPS+1
        end = ENROLL_REPS+TEST_REPS+1
        copyFiles(inDir, test_outDir, gender, speakerID, start, end)








if __name__ == "__main__":
    INDIR = "/media/anwar/D/Data/ASR/Arabic_Corpus_of_Isolated_Words/mono_16k"
    OUTDIR = "/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env/audio"
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    #Split speakers to 80% enroll, and 20% test
    enrollIDs, testIDs = splitSpeakers(ENROLL_NUM, TEST_NUM)
    print(enrollIDs) #[2, 35, 39, 1, 22, 25, 16, 14, 8, 30]
    print(testIDs)   #[47, 50, 49, 6, 45]
    #copy the data from INDIR to OUTDIR
    copyData(INDIR, OUTDIR, enrollIDs, testIDs)
