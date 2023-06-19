from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

path = os.getcwd()
print(path)
l= 50
sub_samp = 25
scales = np.arange(1,33)
wavelet = 'morl'
vmin =1
vmax = 0
df = pd.read_csv("power_10Hz.csv", delimiter = ",",encoding = 'unicode_escape').drop(columns = 'Unnamed: 0')
#diff_df = df.diff().drop(index = 0).reset_index().drop(columns = 'index')
n = int(len(df)/l)  

diff_df = df.diff().drop(index = 0).reset_index().drop(columns = 'index')
'''
indices = [1100500,1108000,1234000,1234500,2564500,2569000,3243000,3244500,3246250, 3263500,3268500,3269750,3481750]

indices_l = [int(x/(l*sub_samp)) for x in indices]

l
'''
for i in tqdm(range(1,n+1)):
#for i in tqdm(indices_l):
    dff = diff_df[l*(i-1): l*i]
    
    data_in_use = np.array(dff) 
    data_in_use[np.abs(data_in_use) < 0.001] = 0
    if data_in_use.max() == 0.0 and data_in_use.min() == 0.0:
      continue
    coef, freqs=pywt.cwt(data_in_use,scales,wavelet = wavelet)
    mod_coeff = np.abs(coef)
    file_path = "/home/user/Desktop/jupyter1/DDP/Scalograms/Scalogram" + str(i) + '_sc.png'
    if np.amin(mod_coeff) < vmin:
      vmin = np.amin(mod_coeff)
    else:
      pass
    if np.amax(mod_coeff) > vmax:
      vmax = np.amax(mod_coeff)
    else:
      pass
  
    plt.figure(figsize = (32,32))
    plt.imshow(np.abs(coef), origin = 'lower', cmap = 'BrBG', aspect = 'auto',
          vmin = vmin, vmax = vmax)
    plt.title('Scalogram_' + str(i))
    plt.ylabel('Scale')
    plt.xlabel('Time')
    plt.savefig(file_path)
    plt.tight_layout()
    plt.close()
