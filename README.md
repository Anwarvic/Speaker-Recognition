# Speaker Recognition using SideKit
This repo contains my Speaker Recognition/Verification project using SideKit.

Speaker recognition is the identification of a person given an audio file. It is used to answer the question "Who is speaking?" Speaker verification (also called speaker authentication) is simliar to speaker recognition, but instead of return the speaker who is speaking, it returns whether the speaker (who is claiming to be a certain one) is truthful or not. Speaker Verification is considered to be a little easier than speaker recognition.

Recognizing the speaker can simplify the task of translating speech in systems that have been trained on specific voices or it can be used to authenticate or verify the identity of a speaker as part of a security process. Speaker recognition has a history dating back some four decades and uses the acoustic features of speech that have been found to differ between individuals. These acoustic patterns reflect both anatomy and learned behavioral patterns.

## SideKit
SIDEKIT is an open source package for Speaker and Language recognition. The aim of SIDEKIT is to provide an educational and efficient toolkit for speaker/language recognition including the whole chain of treatment that goes from the audio data to the analysis of the system performance.

Authors:	Anthony Larcher & Kong Aik Lee & Sylvain Meignier
Version:	1.3.1 of 2019/01/22
You can check the official documentation, altough I don't recommend it, from [here](https://projets-lium.univ-lemans.fr/sidekit/). Also [here](https://projets-lium.univ-lemans.fr/sidekit/api/index.html) is the API documentation.


To run SIDEKIT on your machine, you need to:

- Install the dependencies by running `pip install -r requirements.txt`
- Install `pytorch` by running the following command: `pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl`.
- Install `torchvision` by running `pip3 install torchvision`.
- Install tkinter by running `sudo apt-get install python3.6-tk`.
- Install libSVM by running `sudo apt-get install libsvm-dev`. This library dedicated to SVM classifiers.

**IMPORTANT NOTE**:

There is no need to install SIDEKIT as the library isn't stable and requires some manuevering so I cloned the project from gitLab using`git clone https://git-lium.univ-lemans.fr/Larcher/sidekit.git` and did some editing. So, you just need to clone my project and you are ready to go!!

## Download Dataset
This project is just a proof-of-concept, so it was built using the merged vesion of a small open-source dataset called the "Arabic Corpus of Isolated Words" made by the [University of Stirling](http://www.cs.stir.ac.uk/) located in the Central Belt of Scotland. This dataset can be downloaded from [here](https://www.kaggle.com/mohamedanwarvic/merged-arabic-corpus-of-isolated-words). 

This dataset is a voice-recorded dataset of 50 Native-Arabic speakers saying 20 words about 10 times. It has been recorded with a 44100 Hz sampling rate and 16-bit resolution. This dataset can be used for tasks Speaker Recognition, Speaker Verification, Voice biometrics, ... etc.

This dataset (1GB) is divided into:

- 50 Speakers (starting from S01 ending at S50.
- 47 of these speakers are Male and just 3 are females and they are: S11, S36, and S44.
- Each speaker is in a separate directory named after the speaker id.
- Each speaker recorded 2 waves, about 10 times (sessions). So, each speaker should contain 20 wave files.
- These two waves are names as `{speaker_id}.{session_id}.digits.wav` which contain the recordings of the Arabic digits from zero to 9. And `{speaker_id}.{session_id}.words.wav` which contain the recordings of some random Arabic words.

After downloading the dataset and extracting it, you will find about  50 folders with the name of "S+speakerId" like so S01, S02, ... S50. Each one of these folders should contain around 20 audio files for every speaker, each audio file contains the audio of the speaker speaking 10 in a single WAV file words. This is repeated for 10 times/sessions. And these words are:
```
first_wav_words = {
        "01": "صِفْرْ", 
        "02":"وَاحِدْ",
        "03":"إِثنَانِْ",
        "04":"ثَلَاثَةْ",
        "05":"أَربَعَةْ",
        "06":"خَمْسَةْ",
        "07":"سِتَّةْ",
        "08":"سَبْعَةْ",
        "09":"ثَمَانِيَةْ",
        "10":"تِسْعَةْ"
}

second_wav_words = {
        "01":"التَّنْشِيطْ",
        "02":"التَّحْوِيلْ",
        "03":"الرَّصِيدْ",
        "04":"التَّسْدِيدْ",
        "05":"نَعَمْ",
        "06":"لَا",
        "07":"التَّمْوِيلْ",
        "08":"الْبَيَانَاتْ",
        "09":"الْحِسَابْ",
        "10":"إِنْهَاءْ"
}
```

## How it Works
The sideKit pipeline consists of six steps as shown in the following image:

<p align="center">
<img src="http://www.mediafire.com/convkey/cc16/r56t49ybirn455izg.jpg" /> 
</p>

As we can see, the pipeline consists of six main steps:

- **Preprocessing**: In this step, we perform some processing over the wav files to be consistent like changing the bit-rate, sampling rare, number of channels, ... etc. Besides dividing the data into *training (or enroll)* and *testing*.
- **Structure**: In this step, we produce some files that will be helpful when training and evaluating our model.
- **Feature Extraction**: In this step, we extract pre-defind features from the wav files.
- **Choosing A Model**: In this step, we choose a certain model, out of four, to be trained. We have five models that can be trained:
	- UBM
	- SVM with GMM (NOT READY YET)
	- I-vector
	- Deep Learning (NOT READY YET)
- **Training**: This step is pretty self-explanatory ... com'on.
- **Evaluating**: This step is used to evaluate our model using a test set.

All the configuration options for all previous steps can be found in a YAML file called `conf.yaml`. We will discuss most of these configurations, each at its associated section.

Now, let's talk about each one of these processes in more details:

### 1. Preprocessing
The file responsible for data pre-processing is `data_init.py` in which I split the whole data into two groups (one for training -enroll- and the other for testing). Then doing some preprocessing over the two sets to match the case that I'm creating this model for, like: 

- Setting the sample rate to 44100.
- Setting the number of channels to one (mono).
- Setting the precision to 16-bit.

In the configuration file `conf.yaml`, you can modify only these:

- `inpath`: the absolute path of the directory where the audio data exist.
- `outpath`: the absolute path of the directory where the audio data exist.
- `EXCLUDED_WORDS`: which contains the word's ID you want to exclude.
- `ENROLL_NUM`: which are the number of speakers to be included in the training (enroll).
- `TEST_NUM`: which are the number of speakers to be included in the test outside the enroll. I set the `ENROLL_NUM=10` and `TEST_NUM=5` which means that the training will be done on just 10 speakers and the test will be done using 15 speakers.
- `ENROLL_REPS`: which determines how many repetition a word will be repeated in the training. Three is pretty realistic.
`TEST_REPS`: it determines how many repetitions will be tested over a single word.

And you need also to set these two variables:

- `INDIR`: which is the abolute path to the location of the downloaded data.
- `OUTDIR`: which is the absolute path to the location where your project is located.

After running this file, a directory named `audio` will be created where three sub-directories will be found. These three sub-directories are `data` which contains the whole data preprocessed, `enroll` which contains only the enrolled data for trainined, and `test` which contains the data for testing.

**Note:**

- The number of files inside `data` should equal `47 * 20 * 10 = 9400` where `47` is the total number of speakers minus the excluded ones, `20` is the total number of words, and `10` is the total number of repetition of a single word. 
- The number of files inside `enroll` should equal `ENROLL_NUM * number of used words * ENROLL_REPS`. In my case, they are `10 * 20 * 3 = 600`.
- The number of files inside `test` should equal `(TEST_NUM+ENROLL_NUM) * number of used words * TEST_REPS`. In my case, they are `15 * 20 * 7 = 2100`.

### 2. Feature Extraction
The file responsible for the feature extraction is `extract_features.py` in which I extract features from the audio files and the extracted features will be located at a new folder called `feat` at the project directory. Note that this file needs the data to be located at `audio` directory and it also needs the enroll/training data to be at `audio/enroll` and the test data to be at `audio/test`.

To be able to use this file, you need to set these member variables:

- `BASE_DIR`: it's the absolute path to the project's directory.
- `NUM_THREADS`: the number of threads to be running in parallel
- `FEAUTRES`: it is a list of features that will to be extracted from the audio files. The list of features I used are These features are `fb`: Filter Banks, `cep`: Cepstral Coefficients, `eneregy` and `vad`: Voice Activity Detection. If you chose `vad` within the least of features, you need to set the alogrithm that will be used between either `snr` or `energy`. I chose `snr`: Signal-to-Noise Ratio.
- `FILTER_BANK`: The type of filter-banknig used. It can be either `log` or `lin`:linear.
- `FILTER_BANK_SIZE`: Size of the filter bank.
- `LOWER_FREQUENCY`: the lower frequency (in Hz) of the filter bank.
- `HIGHER_FREQUENCY`: the higher frequency (in Hz) of the filter bank.
- `VAD`: The Voice Activity Detection algorithm used.
- `SNR_RATIO`: The ratio of the SNR algortihm (in db).
- `WINDOW_SIZE`: the size of the window for cep features.
- `WINDOW_SHIFT`: The step that the window is moving (in sec).
- `CEPS_NUMBER`: the size of the cep vector.

There is also a method called `reviewMemberVariables` that resets these member varibales back to `None` based on the `FEATURES` used in your model.

You can download the features used in my model from [here](http://www.mediafire.com/file/03o7i80o7a2taza/feat.zip/file). After downloading, you should extract them in the projects directory.

### 3. Structure









TO BE CONTINUED :)