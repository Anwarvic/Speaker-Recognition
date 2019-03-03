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
![SideKit pipeline](http://www.mediafire.com/convkey/cc16/r56t49ybirn455izg.jpg) 
As we can see, the pipeline consists of six main steps:

- Preprocessing: In this step, we perform some processing over the wav files to be consistent like changing the bit-rate, sampling rare, number of channels, ... etc. Besides dividing the data into **training (or enroll)** and **testing**.
- Feature Extraction: In this step, we extract pre-defind features from the wav files.
- Structure: In this step, we produce some files that will be helpful when training and evaluating our model.
- Choosing A Model: In this step, we choose a certain model, out of four, to be trained. We have five models that can be trained:
	- UBM
	- SVM with GMM
	- I-vector
	- Deep Learning
- Training: This step is pretty self-explanatory ... com'on.
- Evaluating: This step is used to evaluate our model. We have two ways of evaluating a model. The first one is to draw the DET (Detection Error Tradeoff) graph. And the second is getting the accuracy percentage.
Let's talk about each one of these in more details:
### 1. Preprocessing