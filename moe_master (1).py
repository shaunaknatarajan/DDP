import PIL
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
import sklearn.model_selection as model_selection
import os
import time
import matplotlib.pyplot as plt
import pywt
import keras
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image

def load_y_data(y_path):
    y = np.loadtxt(y_path, dtype=np.int32).reshape(-1,1)
    # change labels range from 1-6 t 0-5, for sparse_categorical_crossentropy loss function implementation
    return y-1

def load_X_data(X_path):
    X_signal_paths = [X_path + file for file in os.listdir(X_path)]
    X_signals = [np.loadtxt(path, dtype=np.float32) for path in X_signal_paths]
    return np.transpose(np.array(X_signals), (1, 2, 0))

PATH = '/home/ravi/churn-analysis/data_A5_Saturn'
LABEL_NAMES = ["1","2","3","4","6","7", "10", "11"]
#Stationary: 7 Cementing: 4 Tripping-in: 1 Tripping-out: 2 Casing: 3 Liner: 5 Tubing: 8 Tripping out Drilling: 6 
#Out of surface: 10 Conditioning: 11

# load  input data
X_all= load_X_data(PATH + '/channel_data/')

# load target label
y_all = load_y_data(PATH + '/Y_Data/RigState.txt')  


# In[3]:


np.unique(y_all, return_counts= True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_all, y_all, test_size=0.2, random_state =42)


X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

print("data details:")

print(f"shapes (n_samples, n_steps, n_signals) of X_train: {X_train.shape} and X_val: {X_val.shape} and X_test: {X_test.shape}")

X_full = np.concatenate([X_train, X_val, X_test])
print(f"all X's have following mean: {format(X_full.mean(), '.2f')} and standard derivation: {format(X_full.std(), '.2f')} ")

import pywt
import matplotlib.pyplot as plt

def split_indices_per_label(y):
    #ly = np.unique(y)
    indicies_per_label = [[] for x in range(0,10)]
    # loop over the labels
    for i in range(10): 
        indicies_per_label[i] = np.where(y == i)[0]
    return indicies_per_label

def plot_cwt_coeffs_per_label(X, label_indicies, label_names, signal, sample, scales, wavelet):
    
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(12,5))
    
    for ax, indices, name in zip(axs.flat, label_indicies, label_names):
        # apply continuous wavelet transform
        coeffs, freqs = pywt.cwt(X[indices[sample],:, signal], scales, wavelet = wavelet)
        # create scalogram
        ax.imshow(np.abs(coeffs), cmap = 'coolwarm', aspect = 'auto')
        ax.set_title(name)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Scale')
        ax.set_xlabel('Time')
    plt.tight_layout()

# 
train_labels_indicies = split_indices_per_label(y_train)
#

signal = 6 
sample = 1 
scales = np.arange(1,64)
wavelet = 'mexh' # mother wavelet

plot_cwt_coeffs_per_label(X_train, train_labels_indicies, LABEL_NAMES, signal, sample, scales, wavelet)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

from skimage.transform import resize

def create_cwt_images(X, n_scales, rescale_size, wavelet_name = "mexh"):
    n_samples = X.shape[0] 
    n_signals = X.shape[2] 
    
    # range of scales from 1 to n_scales
    scales = np.arange(1, n_scales + 1) 
    
   
    X_cwt = np.ndarray(shape=(n_samples, rescale_size, rescale_size, n_signals), dtype = 'float32')
    
    for sample in range(n_samples):
        if sample % 1000 == 0:
            print(sample)
        for signal in range(n_signals):
            serie = X[sample, :, signal]
            # continuous wavelet transform 
            coeffs, freqs = pywt.cwt(serie, scales, wavelet_name)
            # resize cwt coeffs
            rescale_coeffs = resize(np.abs(coeffs), (rescale_size, rescale_size), mode = 'constant')
            X_cwt[sample,:,:,signal] = rescale_coeffs
            
    return X_cwt

rescale_size = 64

n_scales = 32

X_train_cwt = create_cwt_images(X_train, n_scales, rescale_size)
#print(f"shapes (n_samples, x_img, y_img, z_img) of X_train_cwt: {X_train_cwt.shape}")

X_val_cwt = create_cwt_images(X_val, n_scales, rescale_size)
print('shapes (n_samples, x_img, y_img, z_img) of X_val_cwt: {X_val_cwt.shape}')

X_test_cwt = create_cwt_images(X_test, n_scales, rescale_size)
print('shapes (n_samples, x_img, y_img, z_img) of X_test_cwt: {X_test_cwt.shape}')

import tensorflow as tf
print("TensorFlow version: {tf.__version__}")

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

def build_cnn_model(activation, input_shape):
    model = Sequential()
    model.add(Conv2D(32, 5, activation = activation, padding = 'same', input_shape = input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 5, activation = activation, padding = 'same', kernel_initializer = "he_normal"))
    model.add(MaxPooling2D())  
    model.add(Conv2D(64, 5, activation = activation, padding = 'same', kernel_initializer = "he_normal"))
    model.add(MaxPooling2D())  
    model.add(Flatten())
    model.add(Dense(128, activation = activation, kernel_initializer = "he_normal"))
    model.add(Dense(64, activation = activation, kernel_initializer = "he_normal"))
    model.add(Dense(10, activation = 'softmax')) # 8 classes
    
    # summarize the model
    print(model.summary())
    return model

def compile_and_fit_model(model, X_train, y_train, X_test, y_test, batch_size, n_epochs):

    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])
    
    
    callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_sparse_categorical_accuracy', save_best_only=True)]
    
    
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val))
    
    return model, history

input_shape = (X_train_cwt.shape[1], X_train_cwt.shape[2], X_train_cwt.shape[3])


cnn_model = build_cnn_model("relu", input_shape)

trained_cnn_model, cnn_history = compile_and_fit_model(cnn_model, X_train_cwt, y_train, X_val_cwt, y_val, 368, 18)

trained_cnn_model.save('/home/ravi/RR_Diff_Binary_mexh_32_64_A5_Saturn_Twenty_val_model.h5')

import seaborn as sns
from sklearn import metrics

def create_confusion_matrix(y_pred, y_test):    
    #calculate the confusion matrix
    confmat = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(confmat, cmap=plt.cm.Blues, alpha=0.5)

    n_labels = len(LABEL_NAMES)
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)

   
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=i, y=j, s=confmat[i, j], va='center', ha='center')
    
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

y_pred = trained_cnn_model.predict_classes(X_test_cwt)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

create_confusion_matrix(y_pred, y_test)
