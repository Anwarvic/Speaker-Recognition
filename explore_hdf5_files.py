from utils import explore_file


"""
wavFile:
<KeysViewHDF5 ['MS01_01_01.wav', 'compression']>
Key: MS01_01_01.wav
Value Type: <HDF5 group "/MS01_01_01.wav" (16 members)>
Value Shape: (16,)
Value: ['cep' 'cep_header' 'cep_mean' 'cep_min_range' 'cep_std' 'energy'
 'energy_header' 'energy_mean' 'energy_min_range' 'energy_std' 'fb'
 'fb_header' 'fb_mean' 'fb_min_range' 'fb_std' 'vad']
=========================
Key: compression
Value Type: <HDF5 dataset "compression": shape (), type "<i8">
Value Shape: ()
Value: 2
=========================
"""


"""
test_ndx:
<KeysViewHDF5 ['modelset', 'segset', 'trial_mask']>
Key: modelset
Value Type: <HDF5 dataset "modelset": shape (15,), type "|S4">
Value Shape: (15,)
Value: [b'MS01' b'MS02' b'MS06' b'MS08' b'MS14' b'MS16' b'MS22' b'MS25' b'MS30'
 b'MS35' b'MS39' b'MS45' b'MS47' b'MS49' b'MS50']
=========================
Key: segset
Value Type: <HDF5 dataset "segset": shape (2100,), type "|S19">
Value Shape: (2100,)
Value: [b'test/MS01_04_01.wav' b'test/MS01_04_02.wav' b'test/MS01_04_03.wav' ...
 b'test/MS50_10_18.wav' b'test/MS50_10_19.wav' b'test/MS50_10_20.wav']
=========================
Key: trial_mask
Value Type: <HDF5 dataset "trial_mask": shape (15, 2100), type "|i1">
Value Shape: (15, 2100)
Value: [[1 1 1 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 1 1]]
=========================
"""


"""
enroll_stat
<KeysViewHDF5 ['modelset', 'segset', 'start', 'stat0', 'stat1', 'stop']>
Key: modelset
Value Type: <HDF5 dataset "modelset": shape (600,), type "|S255">
Value Shape: (600,)
Value: [b'MS01' b'MS01' b'MS01' b'MS01' b'MS01' b'MS01' b'MS01' b'MS01' b'MS01'
 b'MS01' b'MS01' b'MS01' b'MS01' b'MS01' b'MS01' b'MS01' b'MS01' b'MS02'
 b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02'
 ...
 b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02'
 b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02' b'MS02'
 b'MS02' b'MS02' b'MS02' b'MS08' b'MS08' b'MS08' b'MS08' b'MS08' b'MS08'
 b'MS35' b'MS35' b'MS35' b'MS35' b'MS35' b'MS35']
=========================
Key: segset
Value Type: <HDF5 dataset "segset": shape (600,), type "|S255">
Value Shape: (600,)
Value: [b'enroll/MS01_01_01.wav' b'enroll/MS01_01_02.wav'
 b'enroll/MS01_01_03.wav' b'enroll/MS01_01_04.wav'
 b'enroll/MS01_01_05.wav' b'enroll/MS01_01_06.wav'
 b'enroll/MS01_01_07.wav' b'enroll/MS01_01_08.wav'
 b'enroll/MS01_01_09.wav' b'enroll/MS01_01_10.wav'
 b'enroll/MS01_01_11.wav' b'enroll/MS01_01_12.wav'
 b'enroll/MS01_01_13.wav' b'enroll/MS01_01_14.wav'
 b'enroll/MS01_01_15.wav' b'enroll/MS01_01_16.wav'
...
 b'enroll/MS01_01_17.wav' b'enroll/MS02_01_19.wav'
 b'enroll/MS30_03_18.wav' b'enroll/MS30_03_19.wav'
 b'enroll/MS30_03_20.wav' b'enroll/MS35_01_01.wav'
 b'enroll/MS35_01_02.wav' b'enroll/MS35_01_03.wav'
 b'enroll/MS35_01_04.wav' b'enroll/MS35_01_05.wav'
 b'enroll/MS35_01_06.wav' b'enroll/MS35_01_07.wav'
 b'enroll/MS35_01_08.wav' b'enroll/MS35_01_09.wav'
 b'enroll/MS35_01_10.wav' b'enroll/MS35_01_11.wav']
=========================
Key: start
Value Type: <HDF5 dataset "start": shape (600,), type "|i1">
Value Shape: (600,)
Value: [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 ...
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
=========================
Key: stat0
Value Type: <HDF5 dataset "stat0": shape (600, 32), type "<f4">
Value Shape: (600, 32)
Value: [[4.22525363e-06 1.25457397e-22 2.14592557e-08 ... 1.00015926e+00
  3.09200931e+00 2.17751869e-13]
 [1.03120673e+00 4.87949045e-24 1.90984929e+00 ... 1.99999046e+00
  4.61276150e+00 3.80742177e-03]
 [3.02973557e+00 3.89771771e+00 3.99923158e+00 ... 1.99999213e+00
  3.99661136e+00 1.00212145e+00]
 ...
 [8.65445194e-13 4.43241373e-03 1.54910449e-05 ... 9.99999464e-01
  1.00000119e+00 1.00281215e+00]
 [1.44533169e+00 1.10637935e-16 1.89461768e+00 ... 1.99983180e+00
  9.04784203e-01 2.36073756e+00]
 [1.00068641e+00 3.09467077e-01 7.32917118e+00 ... 1.99967194e+00
  3.37822127e+00 2.00291371e+00]]
=========================
Key: stat1
Value Type: <HDF5 dataset "stat1": shape (600, 4224), type "<f4">
Value Shape: (600, 4224)
Value: [[-6.7632050e-06 -2.8391512e-06 -1.7485643e-06 ...  1.5283616e-13
   2.2366531e-13  1.5222272e-13]
 [-8.0579400e-01  2.7705008e-02  1.0520424e+00 ...  1.4134807e-03
   4.7722678e-03  5.0291903e-03]
 [-1.5057135e+00  1.8970174e+00  6.9014454e-01 ...  3.3700544e-01
   6.4156878e-01  4.2040855e-01]
 ...
 [-4.4079622e-13 -4.9817258e-13  3.3988304e-13 ...  5.0988328e-01
   7.8357536e-01  9.5903724e-01]
 [-4.8816833e-01  7.5340039e-01 -1.5388818e+00 ...  7.5796938e-01
  -5.3048772e-03 -8.0529141e-01]
 [-8.5066539e-01  4.7747120e-01 -1.8504832e+00 ...  7.9523855e-01
   3.7151437e-02  2.0318198e-01]]
=========================
Key: stop
Value Type: <HDF5 dataset "stop": shape (600,), type "|i1">
Value Shape: (600,)
Value: [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 ...
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
=========================
"""



if __name__ == "__main__":
    explore_file("/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env/feat/enroll/MS01_01_01.wav.h5")
    # explore_file("/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env/task/idmap_enroll.h5")
    # explore_file("/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env/task/test_ndx.h5")
    # explore_file("/media/anwar/E/Voice_Biometrics/SIDEKIT-1.3/py3env/task/enroll_stat.h5")
