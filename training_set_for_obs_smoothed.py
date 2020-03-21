# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:23:12 2018

@author: Administrator
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


#%%

dataPath='/Users/duminghao/Desktop/ML_Mode_Classify/data'
pkbgPath=dataPath+'/pkbg_results/'
setPath =dataPath+'/set/'
psdPath =dataPath+'/psd/'
basic_para=pd.read_csv(dataPath+'/final_subgiants1.csv',header=0)
reshape_num=10000   #interpolate to this number
l=[]               # the label set


#%%
def mkdir(path):  
  
    folder = os.path.exists(path)  
  
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹  
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径  
        print("---  new folder...  ---")  
        print("   ---  OK  ---"  )
  
    else:  
        print("---  There is this folder!  ---")

#%%
def power_cut(kic):
    
    mkdir(setPath+'%s' %kic)

    star_detail_info=pd.read_csv(pkbgPath+'%s_modeid_peakbagging_analysis.csv' %kic,header=0)
#    freq_num=star_detail_info.shape[0]
    dnu=basic_para[basic_para.kic==kic]['dnu'].tolist()
    dnu=dnu[0]

    psd=pd.read_csv(psdPath+'%s_wg1_smooth.power' % kic)
    psd.columns=['Freq','Ppm']
   
    
    for i in range(len(star_detail_info['freq'])):
        if star_detail_info['l'][i] == 3:
            continue                           #ignore the l=3 peak because of the so poor data
        model_cent_freq=star_detail_info['freq'][i]
        l.append(star_detail_info['l'][i])        
            
        dif_value=[]
        cand_freq=[]
        for j in psd['Freq']:
            if j-model_cent_freq < 1:
                cand_freq.append(j)
                dif_value.append(abs(j-model_cent_freq))
        cent_freq=psd[psd.Freq==cand_freq[dif_value.index(min(dif_value))]]['Freq'].tolist()

        
        dif_value=[]
        cand_freq=[]
        for j in psd['Freq']:
            if j-(cent_freq[0]-dnu+2) < 1:
                cand_freq.append(j)
                dif_value.append(abs(j-(cent_freq[0]-dnu+2)))
        lower_freq=psd[psd.Freq==cand_freq[dif_value.index(min(dif_value))]]['Freq'].tolist()

        
        dif_value=[]
        cand_freq=[]
        for j in psd['Freq']:
            if j-(cent_freq[0]+dnu-2) < 1:
                cand_freq.append(j)
                dif_value.append(abs(j-(cent_freq[0]+dnu-2)))
        upper_freq=psd[psd.Freq==cand_freq[dif_value.index(min(dif_value))]]['Freq'].tolist()
        
        lower_freq_index=psd[psd.Freq==lower_freq[0]].index.tolist()
        upper_freq_index=psd[psd.Freq==upper_freq[0]].index.tolist()
        
#        print(lower_freq_index)
#        print(upper_freq_index)
        
        ppm_generate_power=psd['Ppm'][lower_freq_index[0]:upper_freq_index[0]]
        ppm_generate_power=pd.DataFrame(ppm_generate_power)
        if i <10:
            ppm_generate_power.to_csv(setPath+'%s/Spower_0%s.csv' %(kic,i),header=True)
            same_length(setPath+'%s/Spower_0%s.csv' %(kic,i))
            np.savetxt(dataPath+'/temp/power10000_%s_0%s.csv' %(kic,i),ppm_new)
        else:
            ppm_generate_power.to_csv(setPath+'%s/Spower_%s.csv' %(kic,i),header=True)
            same_length(setPath+'%s/Spower_%s.csv' %(kic,i))
            np.savetxt(dataPath+'/temp/power10000_%s_%s.csv' %(kic,i),ppm_new)
        
        #print('Done')

    print(len(ppm_generate_power))
#%%    

def file_name(file_path,file_type):
    global files_name
    global files_path
    files_name=[]  
    files_path=[]
    for root, dirs, files in os.walk(file_path):  
        for file in files:              
            if os.path.splitext(file)[1] == '.%s' % file_type:  
                files_name.append(file)
                files_path.append(os.path.join(root,file))

                
#%%
                # to modify the data to the same lenghth through interpolation

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
    
    
#%% to get the kic star list that can be made into sample
kic_list=[]
kic_wg1file=[]
kic_time_del=[]
for i in basic_para['kic']:
    if os.path.exists(psdPath+'%s_wg1.power' % i):
        kic_wg1file.append(i)
        
kic_time_del=list(basic_para[basic_para.obs_time<60]['kic']) #obs_day=60 means a good quality of data
        
        
file_name(pkbgPath,'csv')
kic_peakfile=[]        
for i in files_name:
    name_split=i.split('_',1)
    kic_peakfile.append(name_split[0]) 
kic_peakfile=list(map(int,kic_peakfile))           
        
#kic_list=list(set(kic_wg1file).intersection(set(kic_peakfile))-set(kic_time_del))

kic_list=list(set(kic_wg1file).intersection(set(kic_peakfile))-set(kic_time_del))
kic_list.sort()    

#%%        
# main program
for kic in kic_list:
    power_cut(kic)
    

file_name(dataPath+'/temp/','csv')
files_name.sort(key=lambda x:int(x[6:-4]))
files_path.sort(key=lambda x:int(x[46:-4]))
sample_full=np.zeros((reshape_num,len(files_name)))  #the final sample set
#plt.plot(psd.iloc[:,0],psd.iloc[:,2]-psd.iloc[:,1])
#plt.show
for i in range(len(files_name)):
    temp=np.loadtxt(files_path[i])
    if temp.shape[0] == reshape_num:
        sample_full[:,i]=temp[:]
    else:
        sample_full[0:temp.shape[0]-1,i]=temp[:]
        sample_full[temp.shape[0]:9999,i]=temp[:]

np.savetxt(dataPath+'/Sample_interP10000_Smooth.csv',sample_full,delimiter=',')
np.savetxt(dataPath+'/Sample_label(l)_Smooth.csv',np.array(l))
 
#%%    

    
    
    
    
    
    
    
