# Speaker Recognition/Verification using SideKit

## Installation
To run SIDEKIT on your machine, you need to install:

- Install the dependencies by running `pip install -r requirements.txt`
- Install `pytorch` by running the following command: `pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl`.
- Install `torchvision` by running `pip3 install torchvision`.
- Install tkinter by running `sudo apt-get install python3.6-tk`.
- Install libSVM by running `sudo apt-get install libsvm-dev`. This library dedicated to SVM classifiers.

Now, let's install SIDEKIT. We can either install it using `pip install sidekit` or we can clone it from GitLab which I recommend, since the library isn't stable and requires some maneuvering.

Notice that I did that for you, but if you want to start from scratch, you can clone the repo using`git clone https://git-lium.univ-lemans.fr/Larcher/sidekit.git`.

## Download Dataset
Here, we are going to try our model over the Arabic Corpus of Isolated Words made by the University of Stirling. This dataset can be downloaded from the official website right [here](http://www.cs.stir.ac.uk/~lss/arabic/).