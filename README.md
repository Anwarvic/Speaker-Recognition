# Speaker Recognition using SideKit
This repo contains my Speaker Recognition/Verification project using SideKit.

Speaker recognition is the identification of a person given an audio file. It is used to answer the question "Who is speaking?" Speaker verification (also called speaker authentication) is simliar to speaker recognition, but instead of return the speaker who is speaking, it returns whether the speaker who is claiming to be a certain on is truthful or not. Speaker Verification is considered to be a little easier than speaker recognition.

Recognizing the speaker can simplify the task of translating speech in systems that have been trained on specific voices or it can be used to authenticate or verify the identity of a speaker as part of a security process. Speaker recognition has a history dating back some four decades and uses the acoustic features of speech that have been found to differ between individuals. These acoustic patterns reflect both anatomy and learned behavioral patterns.

## SideKit 1.3.1
SIDEKIT is an open source package for Speaker and Language recognition. The aim of SIDEKIT is to provide an educational and efficient toolkit for speaker/language recognition including the whole chain of treatment that goes from the audio data to the analysis of the system performance.
Authors:	Anthony Larcher & Kong Aik Lee & Sylvain Meignier
Version:	1.3.1 of 2019/01/22
You can check the official documentation, altough I don't recommend it, from [here](https://projets-lium.univ-lemans.fr/sidekit/). Also [here](https://projets-lium.univ-lemans.fr/sidekit/api/index.html) is the API documentation.

To run SIDEKIT on your machine, you need to install:

- Install the dependencies by running `pip install -r requirements.txt`
- Install `pytorch` by running the following command: `pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl`.
- Install `torchvision` by running `pip3 install torchvision`.
- Install tkinter by running `sudo apt-get install python3.6-tk`.
- Install libSVM by running `sudo apt-get install libsvm-dev`. This library dedicated to SVM classifiers.

Now, let's install SIDEKIT. We can either install it using `pip install sidekit`. However, the library isn't stable and requires some manuevering so I cloned the project from gitLab using`git clone https://git-lium.univ-lemans.fr/Larcher/sidekit.git`.

## Download Dataset
This project is just a proof-of-concept, so it was built using a small open-source dataset called the "Arabic Corpus of Isolated Words" made by the [University of Stirling](http://www.cs.stir.ac.uk/) located in the Central Belt of Scotland. This dataset can be downloaded from the official website right [here](http://www.cs.stir.ac.uk/~lss/arabic/). The "Arabic speech corpus for isolated words" contains about 10,000 utterances (9992 utterances to be precise) of 20 words spoken by 50 native male Arabic speakers. It has been recorded with a 44100 Hz sampling rate and 16-bit resolution in the raw format (.wav files). This corpus is free for noncommercial uses.

After downloading the dataset and extracting it, you will find about 50 folders with the name of "S+speakerId" like so S01, S02, ... S50. Each one of these folders contains around 200 audio files, each audio file contains the audio of the speaker speaking just one word. Notice that the naming of these audio files has certain information that we surely need. So for example the audio file named as "S01.02.03.wav", this means that the wav was created by the speaker whose id is "1", saying the word "03" which is "اثنان", for the "second" repetition. Each speaker has around 200 wav files, saying 20 different words 10 times. And these words are:
```
d = {
        "01": "صِفْرْ", 
        "02":"وَاحِدْ",
        "03":"إِثنَانِْ",
        "04":"ثَلَاثَةْ",
        "05":"أَربَعَةْ",
        "06":"خَمْسَةْ",
        "07":"سِتَّةْ",
        "08":"سَبْعَةْ",
        "09":"ثَمَانِيَةْ",
        "10":"تِسْعَةْ",
        "11":"التَّنْشِيطْ",
        "12":"التَّحْوِيلْ",
        "13":"الرَّصِيدْ",
        "14":"التَّسْدِيدْ",
        "15":"نَعَمْ",
        "16":"لَا",
        "17":"التَّمْوِيلْ",
        "18":"الْبَيَانَاتْ",
        "19":"الْحِسَابْ",
        "20":"إِنْهَاءْ"
        }
```
## How it Works
The sideKit pipeline consists of five steps as shown in the following image:
<p align="center">
<img src="http://www.mediafire.com/convkey/cc16/r56t49ybirn455izg.jpg" /> 
</p>
As we can see, the pipeline consists of six main steps:

- **Preprocessing**: In this step, we perform some processing over the wav files to be consistent like changing the bit-rate, sampling rare, number of channels, ... etc. Besides dividing the data into *training (or enroll)* and *testing*.
- **Feature Extraction**: In this step, we extract pre-defind features from the wav files.
- **Structure**: In this step, we produce some files that will be helpful when training and evaluating our model.
- **Choosing A Model**: In this step, we choose a certain model, out of four, to be trained. We have five models that can be trained:
	- UBM
	- SVM with GMM
	- I-vector
	- Deep Learning
- **Training**: This step is pretty self-explanatory ... com'on.
- ** Evaluating**: This step is used to evaluate our model. We have two ways of evaluating a model. The first one is to draw the DET (Detection Error Tradeoff) graph. And the second is getting the accuracy percentage.
Let's talk about each one of these in more details:
### 1. Preprocessing
The file responsible for data pre-processing is `data_preprocessing.py` in which I split the whole data into two groups (one for training (enroll) and the other for testing) beside doing some preprocessing over the wav files, to match the case that I'm creating this model for, like: 

- Setting the sample rate to 16000.
- Setting the number of channels to one (mono).
- Setting the precision to 16-bit.

In this file, you need to set only these Global Variables:

- `EXCLUDED_IDS`: which includes the speaker you want to exclude from the whole data. In my case, I was working on only males, so I removed these speakers `11`, `36` and `44`.
- `EXCLUDED_WORDS`: which contains the word's ID you want to exclude.
- `ENROLL_NUM`: which are the number of speakers to be included in the training (enroll).
- `TEST_NUM`: which are the number of speakers to be included in the test outside the enroll. I set the `ENROLL_NUM=10` and `TEST_NUM=5` which means that the training will be done on just 10 speakers and the test will be done using 15 speakers.
- `ENROLL_REPS`: which determines how many repetition a word will be repeated in the training. Three is pretty realistic.
`TEST_REPS`: it determines how many repetitions will be tested over a single word.

And you need also to set these two member variables:

- `INDIR`: which is the abolute path to the location of the downloaded data.
- `OUTDIR`: which is the absolute path to the location where your project is located.

After running this file, a directory named `audio` will be created where three sub-directories will be found. These three sub-directories are `data` which contains the whole data preprocessed, `enroll` which contains only the enrolled data for trainined, and `test` which contains the data for testing.

**Note:**

- The number of files inside `data` should equal `47 * 20 * 10 = 9400` where `47` is the total number of speakers - the excluded ones, `20` is the total number of words, and `10` is the total number of repetition of a single word. 
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

You can download the features used in my model from [here](http://www.mediafire.com/file/03o7i80o7a2taza/feat.zip/file). After downloading, you can extract them in the projects directory.

### 3. Structure









TO BE CONTINUED :)