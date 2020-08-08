# Deep-STORM

This code accompanies the paper: ["Deep-STORM: Super resolution single molecule microscopy by deep learning"](https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-5-4-458&id=385495)

# Contents

- [Overview](#overview)
- [System requirements](#system-requirements)
- [Installation instructions](#installation-instructions)
- [Learning a localization model](#learning-a-localization-model)
- [Demo examples](#demo-examples)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

# Overview

Deep-STORM is a single molecule localization microscopy code for training a custom fully convolutional neural network estimator and recovering super-resolution images from dense blinking movies:

![](Figures/DemoExpData.gif "This movie shows representative experimental frames from the EPFL challenge website with localizations overliad as red dots (Fig. 5c main text).")

# System requirements

* The software was tested on a *Linux* system with Ubuntu version 16.04, and a *Windows* system with Windows 10 Home.  
* Training and evaluation were run on a standard workstation equipped with 32 GB of memory, an Intel(R) Core(TM) i7 − 8700, 3.20 GHz CPU, and a NVidia GeForce Titan Xp GPU with 12 GB of video memory.

# Installation instructions

* Prerequisites
This software was tested on a Windows 10 64-bit operating system, with the following packages:
    1. ImageJ 1.51u with ThunderSTORM plugin 1.3 installed.
    2. MatlabR2017b with image-processing toolbox.
    3. Anaconda distribution 5.1 for windows 10 (64-bit) with Tensorflow 1.4.0, and Keras 1.0.0 installed.

* ImageJ and ThunderSTORM
    1. Download and install ImageJ 1.51u - The software is freely available at "https://imagej.nih.gov/ij/download.html"
    2. Download ThunderSTORM plugin ".jar" file from "https://zitmen.github.io/thunderstorm/"
    3. Place the downloaded ".jar" file into ImageJ plugins directory
    * To verify ThunderSTORM is setup correctly, open ImageJ and navigate to the Plugins directory. The ThunderSTORM plugin should appear in the list.

* MatlabR2017b 
    * can be downloaded at "https://www.mathworks.com/products/matlab.html"

* Anaconda
    * Download and install the anaconda distribution for windows at "https://www.anaconda.com/download/"
    * Open up the Anaconda prompt, and create a new conda environment named "deepstorm" using the command: "conda create -n deepstorm pip python=3.5"
    when conda asks you to proceed type "y"
	3.3  - Activate the newly created environment using the command: "activate deepstorm"
	3.4  - Install Tensorflow cpu or gpu-version 1.4.0 in the deepstorm environment using the command: "pip install --ignore-installed --upgrade tensorflow" or
	       "pip install --ignore-installed --upgrade tensorflow-gpu" depending on whether your system have a cuda capable GPU.
	       For more information on tensorflow installation see "https://www.tensorflow.org/install/install_windows"
	3.5  - Install Keras 2.1.3 using the command: "pip install keras"
	3.6  - Install scipy 1.0.0 using the command: "conda install -c anaconda scipy"
	3.7  - Install scikit-learn using the command: "conda install scikit-learn"
	3.8  - Install scikit-image using the command: "conda install scikit-image"
	3.9  - Install matplotlib using the command: "conda install -c conda-forge matplotlib"
	3.10 - Install h5py using the command: "conda install -c anaconda h5py"
	3.11 - Install argparser using the command: "pip install argparse"
	*To verify all the above mentioned packages are installed in the new environment "deepstorm" run the command: "conda list".
	Now the conda environment with all needed dependencies is ready for use, and the prompt can be closed using the command: "exit()".
 
 
 
1. Download this repository as a zip file (or clone it using git).
2. Go to the downloaded directory and unzip it (or clone it using git).
3. The [conda](https://docs.conda.io/en/latest/) environment for this project is given in `environment_<os>.yml` where `<os>` should be substituted with your operating system. For example, to replicate the environment on a linux system use the command: `conda env create -f environment_linux.yml` from within the downloaded directory.
This should take a couple of minutes.
4. After activation of the environment using: `conda activate deep-storm`, you're set to go!



# Learning a localization model

* The software includes Matlab and Python codes, both for training and reconstruction. For training a net, the user needs to perform the following steps:
    1. Simulate a tiff stack of data frames, with known ground truth positions in an accompanying csv file, using ImageJ ThunderSTORM plugin. The simulated images and positions are saved for handling in Matlab.
    2. Generate the training examples matfile in matlab using the script `GenerateTrainingExamples.m`.
    3. Open up the anaconda command prompt, and activate the previously created `deepstorm` environment by using the command: `activate deepstorm`.
    4. Train a convolutional neural network for the created examples using the command: `python Training.py --filename <path_of_the_generated_training_examples_mfile> --weights_name <path_for_saving_the_trained_weights_hdf5_file> --meanstd_name <path_for_saving_the_normalization_factors_mfile>`

* To use the trained network on data for image reconstruction from a blinking movie, run the command: `python  Testing.py --datafile <path_to_tiff_stack_for_reconstruction> --weights_name <path_to_the_trained_model_weights_as_hdf5_file> --meanstd_name <path_to_the_saved_normalization_factors_as_mfile> --savename <path_for_saving_the_Superresolution_reconstruction_matfile> --upsampling_factor <desired_upsampling_factor> --debug <boolean (0/1) for saving individual predictions>`
    * Note: The inputs `upsampling_factor` and `debug` are optional. By default, the `upsampling_factor` is set to 8, and `debug` is set to 0.
 
# Demo examples
 
* There are 2 different demo examples that demonstrate the use of this code:
    1. `demo1 - Simulated Microtubules` - learning a CNN for localizing simulated microtubules structures obtained from the EPFL 2013 Challenge (Fig. 4 main text). It takes approximately 2 hours to train a model from scratch on a Titan Xp. See the pdf instructions inside the folder `demo 1 - Simulated Microtubules` for a detailed step by step application of the software, with snapshots and intermediate outputs.
    2. `demo2 - Real Microtubules` - pre-trained CNN for localizing experimental microtubules (Fig. 6 main text).

# Citation

If you use this code for your research, please cite our paper:
```
@article{nehme2018deep,
  title={Deep-STORM: super-resolution single-molecule microscopy by deep learning},
  author={Nehme, Elias and Weiss, Lucien E and Michaeli, Tomer and Shechtman, Yoav},
  journal={Optica},
  volume={5},
  number={4},
  pages={458--464},
  year={2018},
  publisher={Optical Society of America}
}
```

# License
 
This project is covered under the [**MIT License**](https://github.com/EliasNehme/Deep-STORM/blob/master/LICENSE).

# Contact

To report any bugs, suggest improvements, or ask questions, please contact me at "seliasne@campus.technion.ac.il"
