from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.preprocessing import image
import keras.utils as image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
#!pip install Pillow
import sklearn
import os, glob
import matplotlib.pyplot as plt

import pywt
#!pip install keras
import keras
#!pip install seaborn
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image

path = os.getcwd()
print(path)
l= 50
scales = np.arange(1,33)
wavelet = 'morl'
vmin =1
vmax = 0
'''
dfPowerData = pd.read_csv("power_10Hz.csv")

df = dfPowerData['VA_POWER']

diff_df = diff(df)  # differenced power signal data. Anomalies in these scalograms are more pronounced than the ones with the original signal

#THIS is to generate scalograms
n = int(len(diff_df)/l)
for i in tqdm(range(1,n)):
    dff = diff_df[l*(i-1): l*i]
    data_in_use = np.array(dff) 
    data_in_use[np.abs(data_in_use) < 0.001] = 0
    if data_in_use.max() == 0.0 and data_in_use.min() == 0.0:
      continue
    coef, freqs=pywt.cwt(data_in_use,scales,wavelet = wavelet)
    mod_coeff = np.abs(coef)
    file_path = '/home/user/Desktop/jupyter1/DDP/Scalograms' + str(i) + '_sc.png'
    if np.amin(mod_coeff) < vmin:
      vmin = np.amin(mod_coeff)
    else:
      pass
    if np.amax(mod_coeff) > vmax:
      vmax = np.amax(mod_coeff)
    else:
      pass
  
    plt.figure(figsize = (20,20))
    plt.imshow(np.abs(coef), origin = 'lower', cmap = 'coolwarm', aspect = 'auto',
          vmin = vmin, vmax = vmax)
    plt.title('Scalogram_' + str(i))
    plt.ylabel('Scale')
    plt.xlabel('Time')
    plt.savefig(file_path)
    plt.tight_layout()
    plt.close()
'''
#Anomalies on a partial signal: scalograms to look at are the ones with numbers 4396, 4403, 4440, 4938-45, 4972-76, 10277, 10279, 12974-988

# This is to extract features using Resnet50. Pick the output at the last but one layer of the network
from pandas.core.arrays import numeric
import re
df_for_feature_vectors = pd.DataFrame(columns = None)
model = ResNet50(weights = 'imagenet', include_top = False, pooling = 'avg')
file_path = '/home/user/Desktop/jupyter1/DDP/Scalograms/'
df_for_feature_vectors = pd.DataFrame()
for infile in tqdm(glob.glob(file_path + '*.png')):
  fileindex = re.findall('[0-9]+', infile)
  fileindex = [int(i) for i in fileindex][1]
  img = image.load_img(infile)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis =0)
  x = preprocess_input(x)
  features = model.predict(x)[0]
  features_arr = np.char.mod('%f', features)
  df_for_feature_vectors = pd.concat((df_for_feature_vectors, pd.Series(features_arr).rename('FeatureVector_' +str(fileindex))),axis = 1)
  #df_for_feature_vectors['FeatureVector_' +str(fileindex)]= features_arr
'''
  if k != n:
    if k % 1000 == 0: 
      print("files_completed " + str(k))
      df_for_feature_vectors.to_csv("Power_difference_features" + str(k) + ".csv")
  else:
    df_for_feature_vectors.to_csv("Power_difference_features_All.csv")
'''
df_for_feature_vectors.to_csv("Power_difference_features_All.csv")
