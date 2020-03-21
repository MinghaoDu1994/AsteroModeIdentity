#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 23:34:46 2018

@author: duminghao
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from tqdm import tqdm
import shutil

#%%

dataPath=os.getcwd()
pkbgPath=dataPath+'/freq_csv/'
setPath =dataPath+'/set/'
tempPath=dataPath+'/temp/'
psdPath =dataPath+'/simups_60000/'
#basic_para=pd.read_csv(dataPath+'/final_subgiants1.csv',header=0)
reshape_num=1000   #interpolate to this number
l=[]               # the label set

shutil.rmtree(setPath)
shutil.rmtree(tempPath)


def file_name(file_path,file_type):

    files_name=[]  
    files_path=[]
    for root, dirs, files in os.walk(file_path):  
        for file in files:              
            if os.path.splitext(file)[1] == '.%s' % file_type:  
                files_name.append(file)
                files_path.append(os.path.join(root,file))

    return files_name,files_path   

def mkdir(path):    
    folder = os.path.exists(path)    
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹  
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径  
        print("---  new folder...  ---")  
        print("   ---  OK  ---"  )  
    else:  
        print("---  There is this folder!  ---")
        
def same_length(sample_path):
    sample_idvd=[]
    global sample
    global ppm_new
    sample_idvd=pd.read_csv('%s' %sample_path,header=0,index_col=0)
    length=int(sample_idvd.shape[0])
    x=np.arange(0,length)
#    print(len(x))
    y=sample_idvd['Ppm']
#    print(len(y))
    f_xy=interpolate.interp1d(x,y)   #1d line interpolation
    x_new=np.arange(0,length-1,length/reshape_num)
    ppm_new=f_xy(x_new)

def power_cut(name):
    mkdir(setPath+'%s' %name)    
    for i in range(len(star_detail_info['Freq'])):
#        if star_detail_info['L'][i] == 3:
#            continue                           #ignore the l=3 peak because of the so poor data
        model_cent_freq=star_detail_info['Freq'][i]
#        l.append(star_detail_info['l'][i])        
        print(model_cent_freq)
        
        cent_freq = psd[np.round(psd.Freq,4) == np.round(model_cent_freq,4)]['Freq']
        lower_freq = psd[np.round(psd.Freq,3) == np.round(model_cent_freq-dnu-0.1,3)]['Freq']
        upper_freq = psd[np.round(psd.Freq,3) == np.round(model_cent_freq+dnu+0.1,3)]['Freq']
        lower_freq = np.abs(lower_freq - model_cent_freq)
        upper_freq = np.abs(upper_freq - model_cent_freq)
        # dif_value=[]
        # cand_freq=[]
        # for j in psd['Freq']:
        #     if j-model_cent_freq < 0.00001:        #the judgement should be modified cause the resolution of is different
        #         cand_freq.append(j)
        #         dif_value.append(abs(j-model_cent_freq))
        # cent_freq=psd[psd.Freq==cand_freq[dif_value.index(min(dif_value))]]['Freq'].tolist()

        
        # dif_value=[]
        # cand_freq=[]
        # for j in psd['Freq']:
        #     if j-(cent_freq[0]-dnu+0.1) < 0.00001:
        #         cand_freq.append(j)
        #         dif_value.append(abs(j-(cent_freq[0]-dnu+0.00002)))
        # lower_freq=psd[psd.Freq==cand_freq[dif_value.index(min(dif_value))]]['Freq'].tolist()

        
        # dif_value=[]
        # cand_freq=[]
        # for j in psd['Freq']:
        #     if j-(cent_freq[0]+dnu-0.1) < 0.00001:
        #         cand_freq.append(j)
        #         dif_value.append(abs(j-(cent_freq[0]+dnu-0.00002)))
        # upper_freq=psd[psd.Freq==cand_freq[dif_value.index(min(dif_value))]]['Freq'].tolist()
        
        # print(cent_freq)
        # print(lower_freq)
        # print(upper_freq)   
        lower_freq_index=lower_freq.argmin()
        # print(lower_freq_index)
        upper_freq_index=upper_freq.argmin()
     
        
        # print(int(upper_freq_index[0])-int(lower_freq_index[0]))
        
        ppm_generate_power = psd['Ppm'][lower_freq_index:upper_freq_index]
        # plt.plot(np.arange(len(ppm_generate_power)),ppm_generate_power)
        # plt.savefig('1.png')
        # plt.close()
        ppm_generate_power = pd.DataFrame(ppm_generate_power)
        if i <10:
            ppm_generate_power.to_csv(setPath+'%s/Spower_0%s.csv' %(name,i),header=True)
            same_length(setPath+'%s/Spower_0%s.csv' %(name,i))
            # plt.plot(np.arange(len(ppm_new)),ppm_new)
            # plt.savefig('2.png')
            # plt.close()
            np.savetxt(tempPath+'/power10000_%s_0%s.csv' %(name,i),ppm_new)
        else:
            ppm_generate_power.to_csv(setPath+'%s/Spower_%s.csv' %(name,i),header=True)
            same_length(setPath+'%s/Spower_%s.csv' %(name,i))
            np.savetxt(tempPath+'/power10000_%s_%s.csv' %(name,i),ppm_new)
        
        #print('Done')

    #print(len(ppm_generate_power))
#%%
#peaks_name,peaks_path=file_name(pkbgPath,'csv')
ps_name,ps_path=file_name(psdPath,'npy')
mkdir(setPath)
mkdir(psdPath)
mkdir(tempPath)

for i in tqdm(range(len(ps_path))):
    psd = np.load(ps_path[i])
    psd = pd.DataFrame(psd, columns=['Freq','Ppm'])
    name=ps_name[i].split("_",2)[2]
    name=name[:-4]
    dir_name=ps_name[i].split("_",1)[1]
    dir_name=dir_name[:-8]
    star_detail_info=pd.read_csv(pkbgPath+'%s' %name,header=0)
    dnu=1.0
    
    power_cut(dir_name)
    
temp_name,temp_path=file_name(dataPath+'/temp/','csv')
#files_name.sort(key=lambda x:int(x[6:-4]))
#files_path.sort(key=lambda x:int(x[46:-4]))
sample_full=np.zeros((len(temp_name),reshape_num))  #the final sample set
#plt.plot(psd.iloc[:,0],psd.iloc[:,2]-psd.iloc[:,1])
#plt.show
for i in tqdm(range(len(temp_name))):
    temp=np.loadtxt(temp_path[i])
    name=temp_name[i].split("_",2)[2]
    name=name[:-7]
    star_detail_info=pd.read_csv(pkbgPath+'%s.csv' %name,header=0)
    j=temp_name[i].split("_",6)[6]
    j=int(j[:-4])
    l.append(star_detail_info['L'][j]) 
    
    if temp.shape[0] == reshape_num:
        sample_full[i,:]=temp[:]
    else:
        sample_full[i,0:temp.shape[0]]=temp[:]
        sample_full[i,temp.shape[0]:reshape_num-1]=0
    print('1')
np.save(dataPath+'/Simu_Sample_%s.npy' % reshape_num,sample_full)
    
np.save(dataPath+'/Simu_Sample_label_%s.npy' % reshape_num,np.array(l))
    

