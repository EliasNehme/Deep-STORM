#%% This script tests the trained fully convolutional network based on the 
# saved training weights, and normalization created using train_model.

# Import Libraries and model
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import argparse
from CNN_Model import buildModel, project_01, normalize_im
import scipy.io as sio
import time
from os.path import abspath

def test_model(datafile, weights_file, meanstd_file, savename, \
               upsampling_factor=8, debug=0):
    """
    This function tests a trained model on the desired test set, given the 
    tiff stack of test images, learned weights, and normalization factors.
    
    # Inputs
    datafile          - the tiff stack of test images 
    weights_file      - the saved weights file generated in train_model
    meanstd_file      - the saved mean and standard deviation file generated in train_model
    savename          - the filename for saving the recovered SR image
    upsampling_factor - the upsampling factor for reconstruction (default 8)
    debug             - boolean whether to save individual frame predictions (default 0)
    
    # Outputs
    function saves a mat file with the recovered image, and optionally saves 
    individual frame predictions in case debug=1. (default is debug=0)    
    """
    
    # load the tiff data
    Images = io.imread(datafile)
    
    # get dataset dimensions
    (K, M, N) = Images.shape
    
    # upsampling using a simple nearest neighbor interp.
    Images_upsampled = np.zeros((K, M*upsampling_factor, N*upsampling_factor))
    for i in range(Images.shape[0]):
        Images_upsampled[i,:,:] = np.kron(Images[i,:,:], np.ones((upsampling_factor,upsampling_factor)))   
    Images = Images_upsampled
    
    # upsampled frames dimensions
    (K, M, N) = Images.shape
    
    # Build the model for a bigger image
    model = buildModel((M, N, 1))

    # Load the trained weights
    model.load_weights(weights_file)
    
    # load mean and std
    matfile = sio.loadmat(meanstd_file)
    test_mean = np.array(matfile['mean_test'])
    test_std = np.array(matfile['std_test'])   

    # Setting type
    Images = Images.astype('float32')
    
    # Normalize each sample by it's own mean and std
    Images_norm = np.zeros(Images.shape,dtype=np.float32)
    for i in range(Images.shape[0]):
        Images_norm[i,:,:] = project_01(Images[i,:,:])
        Images_norm[i,:,:] = normalize_im(Images_norm[i,:,:], test_mean, test_std)

    # Reshaping
    Images_norm = np.expand_dims(Images_norm,axis=3)    
    
    # Make a prediction and time it
    start = time.time()
    predicted_density = model.predict(Images_norm, batch_size=1)
    end = time.time()
    print(end - start)
    
    # threshold negative values
    predicted_density[predicted_density < 0] = 0
    
    # resulting sum images
    WideField = np.squeeze(np.sum(Images_norm, axis=0))
    Recovery = np.squeeze(np.sum(predicted_density, axis=0))
    
    # Look at the sum image
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    ax1.imshow(WideField)
    ax1.set_title('Wide Field')
    ax2.imshow(Recovery)
    ax2.set_title('Sum of Predictions')
    f.subplots_adjust(hspace=0)
    plt.show()
    
    # Save predictions to a matfile to open later in matlab
    mdict = {"Recovery": Recovery}
    sio.savemat(savename, mdict)
    
    # save predicted density in each frame for debugging purposes
    if debug:
        mdict = {"Predictions": predicted_density}
        sio.savemat(savename + '_predictions.mat', mdict)
    
    return f


if __name__ == '__main__':
    
    # start a parser
    parser = argparse.ArgumentParser()
    
    # path of the tiff stack to be reconstructed
    parser.add_argument('--datafile', help="path to tiff stack for reconstruction")
    
    # path of the optimal model weights and normalization factors, saved after 
    # training with the function "train_model.py" is completed. 
    parser.add_argument('--weights_name', help="path to the trained model weights as hdf5-file")
    parser.add_argument('--meanstd_name', help="path to the saved normalization factors as m-file")
    
    # path for saving the Superresolution reconstruction matfile
    parser.add_argument('--savename', type=str, help="path for saving the Superresolution reconstruction matfile")
    
    # upsampling factor for the superresolution reconstruction 
    parser.add_argument('--upsampling_factor', type=int, default=8, help="desired upsampling factor")
                        
    # boolean debugging constant, turned to debug=1 in order to save a matfile 
    # with individual frame predictions. In case debug=1 the resulting individual 
    # predictions will be saved to a matfile whose path is "savename_predictions.mat" 
    parser.add_argument('--debug', type=int, default=0, help="boolean (0/1) for saving individual predictions")
    
    # parse the input arguments
    args = parser.parse_args()
    
    # run the testing/reconstruction process
    test_model(abspath(args.datafile), abspath(args.weights_name), \
               abspath(args.meanstd_name), abspath(args.savename), \
               args.upsampling_factor, args.debug)
    
    