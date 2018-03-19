DeepSTORM v1.0.0

Software Description:
Deep-STORM is a single molecule localization microscopy code for training a custom fully convolutional neural network estimator and recovering super-resolution images from blinking movies. 
More details on the method can be found in the following paper:
"E. Nehme, L.E. Weiss, T. Michaeli, and Y. Shechtman. Deep-STORM: Super resolution single molecule microscopy by deep learning" https://arxiv.org/pdf/1801.09631.pdf

Getting Started:
These instructions will get you a copy of the project up and running on your local machine. 

Prerequisites:
This software was tested on a Windows 10 64-bit operating system, with the following packages:

1. ImageJ 1.51u with ThunderSTORM plugin 1.3 installed
2. MatlabR2017b with image-processing toolbox.
3. Anaconda distribution 5.1 for windows 10 (64-bit) with Tensorflow 1.4.0, and Keras 1.0.0 installed

Installation:
	1. ImageJ and ThunderSTORM:
	1.1 - Download and install ImageJ 1.51u - The software is freely available at "https://imagej.nih.gov/ij/download.html"
	1.2 - Download ThunderSTORM plugin ".jar" file from "https://zitmen.github.io/thunderstorm/"
	1.3 - Place the downloaded ".jar" file into ImageJ plugins directory
	*To verify ThunderSTORM is setup correctly, open ImageJ and navigate to the Plugins directory. The ThunderSTORM plugin should appear in the list.

	2. MatlabR2017b: can be downloaded at "https://www.mathworks.com/products/matlab.html"

	3. Anaconda:
	3.1  - Download and install the anaconda distribution for windows at "https://www.anaconda.com/download/"
	3.2  - Open up the Anaconda prompt, and create a new conda environment named "deepstorm" using the command: "conda create -n deepstorm pip python=3.5"
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

Usage:
The software includes Matlab and Python codes, both for training and reconstruction. For training a net, the user needs to perform the following steps:
1. Simulate a tiff stack of data frames, with known ground truth positions in an accompanying csv file, using ImageJ ThunderSTORM plugin. 
   The simulated images and positions are saved for handling in Matlab.
2. Generate the training examples matfile in matlab using the script "GenerateTrainingExamples.m".
3. Open up the anaconda command prompt, and activate the previously created "deepstorm" environment by using the command: "activate deepstorm".
4. Train a convolutional neural network for the created examples using the command: " python Training.py --filename "path of the generated training examples m-file" \
--weights_name "path for saving the trained weights hdf5-file" --meanstd_name "path for saving the normalization factors m-file" "

To use the trained network on data for image reconstruction from a blinking movie, run the command: " python  Testing.py --datafile "path to tiff stack for reconstruction" \
--weights_name "path to the trained model weights as hdf5-file" --meanstd_name "path to the saved normalization factors as m-file" \ 
--savename "path for saving the Superresolution reconstruction matfile" --upsampling_factor "desired upsampling factor" --debug "boolean (0/1) for saving individual predictions" "
Note: The inputs upsampling_factor and debug are optional. By default, the upsampling factor is set to 8, and debug=0.

Demo example:
See the pdf instructions inside the folder "demo 1 - Simulated Microtubules" for a detailed step by step application of the software, with snapshots and intermediate outputs.

*To report any bugs, suggest improvements, or ask questions, please contact me at "seliasne@campus.technion.ac.il".
