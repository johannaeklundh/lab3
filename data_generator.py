import numpy as np
import h5py
from tensorflow import keras
import matplotlib.pyplot as plt

#-------------------------------
# Data generator class
#-------------------------------
class DataGenerator:
    def __init__(self, verbose=True):
        self.verbose = verbose

    # Generate training, validation, and testing data
    def generate(
        self, 
        dataset='mnist',   # Dataset type
        N_train=None,      # Number of training samples (if not specified, all samples will be used)
        N_valid=0.1        # Fraction of training samples to use as validation data
    ):
        self.N_train = N_train
        self.N_valid = N_valid
        self.dataset = dataset

        # MNIST dataset, provided through Keras
        if dataset == 'mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
            self.split_data()
            self.normalize()

            # Grayscale images, so we need to add an axis
            self.x_train = np.expand_dims(self.x_train, -1)
            self.x_valid = np.expand_dims(self.x_valid, -1)
            self.x_test = np.expand_dims(self.x_test, -1)

        # CIFAR10 dataset, provided through Keras
        elif dataset == 'cifar10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()
            self.split_data()
            self.normalize()

        # Cropped subset of PatchCamelyon dataset, loaded from disk
        elif dataset == 'patchcam':
            with h5py.File('patchcam/train.h5','r') as f:
                self.x_train = f['x'][:]
                self.y_train = f['y'][:]
            with h5py.File('patchcam/valid.h5','r') as f:
                self.x_valid = f['x'][:]
                self.y_valid = f['y'][:]
            with h5py.File('patchcam/test_x.h5','r') as f:
                self.x_test = f['x'][:]
                self.y_test = []

            self.normalize()
        else:
            raise Exception("Unknown dataset", dataset) 

        # Number of classes
        self.K = len(np.unique(self.y_train))
        
        # Number of color channels
        self.C = self.x_train.shape[3]
        
        # One hot encoding of class labels
        self.y_train_oh = keras.utils.to_categorical(self.y_train, self.K)
        self.y_valid_oh = keras.utils.to_categorical(self.y_valid, self.K)
        self.y_test_oh = keras.utils.to_categorical(self.y_test, self.K)

        if self.verbose:
            print('Data specification:')
            print('\tDataset type:          ', self.dataset)
            print('\tNumber of classes:     ', self.K)
            print('\tNumber of channels:    ', self.C)
            print('\tTraining data shape:   ', self.x_train.shape)
            print('\tValidation data shape: ', self.x_valid.shape)
            print('\tTest data shape:       ', self.x_test.shape)

    # Training/validation data split
    def split_data(self):
        # Random shuffle of training data
        N = self.x_train.shape[0]
        ind = np.random.permutation(N)
        self.x_train = self.x_train[ind]
        self.y_train = self.y_train[ind]

        # Validation data as subset of training data
        self.N_valid = int(N*self.N_valid)
        N = N - self.N_valid
        self.x_valid = self.x_train[-self.N_valid:]
        self.y_valid = self.y_train[-self.N_valid:]

        # Subset of training data, if specified
        if self.N_train and self.N_train < N:
            self.x_train = self.x_train[:self.N_train]
            self.y_train = self.y_train[:self.N_train]
        else:
            self.x_train = self.x_train[:N]
            self.y_train = self.y_train[:N]
            self.N_train = N

    # Normalization of images, from 8-bit images in the range [0,255]
    # to floating point representation in the range [-1,1]
    def normalize(self):
        self.x_train = 2*self.x_train.astype("float32") / 255 - 1.0
        self.x_valid = 2*self.x_valid.astype("float32") / 255 - 1.0
        self.x_test = 2*self.x_test.astype("float32") / 255 - 1.0
        
    # Show some training samples
    def plot(
        self,
        xx = 12,        # Number of columns
        yy = 3,         # Number of rows
        save_path=None  # Specify a filename if you want to export the plot
    ):
        plt.figure(figsize=(18,yy*2))
        cm = 'gray' if self.C==1 else 'viridis'
        for i in range(xx*yy):
            plt.subplot(yy,xx,i+1)
            plt.imshow((self.x_train[i]+1)/2, cmap=cm)
            plt.title('label=%d'%(self.y_train[i]))
            plt.axis('off')
        plt.show()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
