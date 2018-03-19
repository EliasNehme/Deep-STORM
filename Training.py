#%% This script trains the fully convolutional network using keras with a 
# tensorflow backend, based on the saved training examples created in MATLAB.

# Import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import argparse
from os.path import abspath
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from CNN_Model import LossHistory, buildModel, project_01, normalize_im
    
# define a function that trains a model for a given data SNR and density
def train_model(filename, weights_name, meanstd_name):
    
    """
    This function trains a CNN model on the desired training set, given the 
    upsampled training images and labels generated in MATLAB.
    
    # Inputs
    filename      - the name of the training matfile generated in MATLAB
    weights_name  - the name of the hdf5 file for saving the weights
    meanstd_name  - the name of the mat file for saving the normalization factors
    
    # Outputs
    function saves the weights of the trained model to a hdf5, and the 
    normalization factors to a mat file. These will be loaded later for testing 
    the model in test_model.    
    """
    
    # for reproducibility
    np.random.seed(123)

    # Load training data and divide it to training and validation sets
    matfile = h5py.File(filename, 'r')
    patches = np.array(matfile['patches'])
    heatmaps = 100.0 * np.array(matfile['heatmaps'])
    X_train, X_test, y_train, y_test = train_test_split(patches, heatmaps, test_size=0.3, random_state=42)
    print('Number of Training Examples: %d' % X_train.shape[0])
    print('Number of Validation Examples: %d' % X_test.shape[0])
       
    # Setting type
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    #===================== Training set normalization ==========================
    # normalize training images to be in the range [0,1] and calculate the 
    # training set mean and std
    mean_train = np.zeros(X_train.shape[0],dtype=np.float32)
    std_train = np.zeros(X_train.shape[0], dtype=np.float32)
    for i in range(X_train.shape[0]):
        X_train[i, :, :] = project_01(X_train[i, :, :])
        mean_train[i] = X_train[i, :, :].mean()
        std_train[i] = X_train[i, :, :].std()

    # resulting normalized training images
    mean_val_train = mean_train.mean()
    std_val_train = std_train.mean()
    X_train_norm = np.zeros(X_train.shape, dtype=np.float32)
    for i in range(X_train.shape[0]):
        X_train_norm[i, :, :] = normalize_im(X_train[i, :, :], mean_val_train, std_val_train)
    
    # patch size
    psize =  X_train_norm.shape[1]

    # Reshaping
    X_train_norm = X_train_norm.reshape(X_train.shape[0], psize, psize, 1)

    # ===================== Test set normalization ==========================
    # normalize test images to be in the range [0,1] and calculate the test set 
    # mean and std
    mean_test = np.zeros(X_test.shape[0],dtype=np.float32)
    std_test = np.zeros(X_test.shape[0], dtype=np.float32)
    for i in range(X_test.shape[0]):
        X_test[i, :, :] = project_01(X_test[i, :, :])
        mean_test[i] = X_test[i, :, :].mean()
        std_test[i] = X_test[i, :, :].std()

    # resulting normalized test images
    mean_val_test = mean_test.mean()
    std_val_test = std_test.mean()
    X_test_norm = np.zeros(X_test.shape, dtype=np.float32)
    for i in range(X_test.shape[0]):
        X_test_norm[i, :, :] = normalize_im(X_test[i, :, :], mean_val_test, std_val_test)
        
    # Reshaping
    X_test_norm = X_test_norm.reshape(X_test.shape[0], psize, psize, 1)

    # Reshaping labels
    Y_train = y_train.reshape(y_train.shape[0], psize, psize, 1)
    Y_test = y_test.reshape(y_test.shape[0], psize, psize, 1)

    # Set the dimensions ordering according to tensorflow consensous
    K.set_image_dim_ordering('tf')

    # Save the model weights after each epoch if the validation loss decreased
    checkpointer = ModelCheckpoint(filepath=weights_name, verbose=1,
                                   save_best_only=True)

    # Change learning when loss reaches a plataeu
    change_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00005)
    
    # Model building and complitation
    model = buildModel((psize, psize, 1))
    
    # Create an image data generator for real time data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0.,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
        zoom_range=0.,
        shear_range=0.,
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        fill_mode='constant',
        data_format=K.image_data_format())

    # Fit the image generator on the training data
    datagen.fit(X_train_norm)
    
    # loss history recorder
    history = LossHistory()

    # Inform user training begun
    print('Training model...')

    # Fit model on the batches generated by datagen.flow()
    train_history = model.fit_generator(datagen.flow(X_train_norm, Y_train, batch_size=16), \
                                        steps_per_epoch=400, epochs=100, verbose=1, \
                                        validation_data=(X_test_norm, Y_test), \
                                        callbacks=[history, checkpointer, change_lr])    

    # Inform user training ended
    print('Training Completed!')
    
    # plot the loss function progression during training
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('Iteration #')
    plt.ylabel('Loss Function')
    plt.title("Loss function progress during training")
    plt.show()
    
    # Save datasets to a matfile to open later in matlab
    mdict = {"mean_test": mean_val_test, "std_test": std_val_test}
    sio.savemat(meanstd_name, mdict)
    return
    
if __name__ == '__main__':
    
    # start a parser
    parser = argparse.ArgumentParser()
    
    # path of the training data: patches and heatmaps, created in MATLAB using
    # the function "GenerateTrainingExamples.m"
    parser.add_argument('--filename', type=str, help="path to generated training data m-file")
    
    # path for saving the optimal model weights and normalization factors after 
    # training with the function "train_model.py" is completed.
    parser.add_argument('--weights_name', type=str, help="path to save model weights as hdf5-file")
    parser.add_argument('--meanstd_name', type=str, help="path to save normalization factors as m-file")
    
    # parse the input arguments
    args = parser.parse_args()
    
    # run the training process
    train_model(abspath(args.filename), abspath(args.weights_name), abspath(args.meanstd_name))
    
    