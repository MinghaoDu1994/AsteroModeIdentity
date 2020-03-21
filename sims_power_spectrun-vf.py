
# coding: utf-8

# In[178]:


import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

path = os.getcwd()+'/paper_simu'
freq_path = path+'/freq_csv'
peak_path = path+'/peaks'
peakfig_path = path+'/peakfig'
points_num = 60000
ps_path=path+'/simups_%s'  %(points_num)                                                                                           
wn_c=10        #white noise constant.   #1=>2  10=>20
amp_c=10.     #amp ~= 400
# In[179]:

def mkdir(path): 
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

def file_name(file_path,file_type): #search files
    files_name=[]  
    files_path=[]
    for root, dirs, files in os.walk(file_path):  
        for file in files:              
            if os.path.splitext(file)[1] == '.%s' % file_type:  
                files_name.append(file)
                files_path.append(os.path.join(root,file))
    return files_name,files_path


def GuessProfile(sigma,miu,llim,ulim,count): #make guess profile
    #global Guess
    x=np.linspace(llim,ulim,count)
    guess_profile=(1/np.sqrt(2*np.pi)*sigma)*np.exp(-(x-miu)**2/(2*sigma**2))
    Guess=pd.DataFrame({'x':x,'y':guess_profile})
    return Guess

# def GuessProfile(amp,sigma,miu,llim,ulim,count): #make guess profile
#     #global Guess
#     x=np.linspace(llim,ulim,count)
#     guess_profile=amp*np.exp(-(x-miu)**2/(2*sigma**2))
#     Guess=pd.DataFrame({'x':x,'y':guess_profile})
#     return Guess


def CalAmp(series_freq,which_gauss): #calculate the amptitide
    Amp=np.zeros(len(series_freq))
    series_freq = np.array(series_freq)
    for i in range(len(series_freq)):
        Amp[i]=np.mean(which_gauss[np.round(which_gauss['x'],3)==np.round(series_freq[i],3)]['y'])
    return Amp


def LorentzProfile(Amp,x0,line_width,llim,ulim,count): #make lorentz profile
    x=np.linspace(llim,ulim,count)
    lorentz_profile=Amp/(((x-x0)/line_width)**2+1)
    return x,lorentz_profile


def smooth(x, window_len = 11, window = "hanning"):
    # stole from https://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = x #np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == "flat":
        w = np.ones(window_len,"d")
    else:
        w = eval("np."+window+"(window_len)") 
    
    y = np.convolve(w/w.sum(),s,mode="same")
    return y

def SmoothWrapper(x, y, period, windowtype, samplinginterval):
    #samplinginterval = np.median(x[1:-1] - x[0:-2])
    xp = np.arange(np.min(x),np.max(x),samplinginterval)
    yp = np.interp(xp, x, y)
    window_len = int(period/samplinginterval)
    if window_len % 2 == 0:
        window_len = window_len + 1
    ys = smooth(yp, window_len, window = windowtype)
    yf = np.interp(x, xp, ys)
    return yf

shutil.rmtree(peak_path)
shutil.rmtree(peakfig_path)
shutil.rmtree(ps_path)
mkdir(peak_path)
mkdir(peakfig_path)
mkdir(ps_path)
 
files_name,files_path=file_name(freq_path,'csv')

# To ensure the amptitude of each peak, we make three guess profile of different amptitude that follow the ratio of 1:.5:0.5(l=0,1,2)

sigma,miu,llim,ulim,count = 1.7,  6.75,  0.5,  13,  points_num
guess_l0=GuessProfile(sigma,miu,llim,ulim,count)
# guess_l1=GuessProfile(sigma*1.5,miu,llim,ulim,count)
# guess_l2=GuessProfile(sigma/2,miu,llim,ulim,count)

# amp,sigma,miu,llim,ulim,count = 10, 2, 7, 0, 14, points_num
# guess_l0=GuessProfile(amp,sigma,miu,llim,ulim,count)
# guess_l1=GuessProfile(amp*1.5,sigma,miu,llim,ulim,count)
# guess_l2=GuessProfile(amp/2,sigma,miu,llim,ulim,count)


#plt.plot(guess_l0['x'],guess_l0['y'])


# To simulate the line width of different effective temperature,we adopt the fitting fomular:
# 1.47*(Teff/5777)^(9.8+-2) (from li pkbg 2018)


#np.random.seed(1111)
freq_miuHz=np.linspace(llim,ulim,count)
#line_width=np.zeros((4,28))
Teff=[5200,5600,6000,6400] #6400
x = np.linspace(llim,ulim,points_num)
s_interval = np.median(x[1:-1]-x[0:-2])
for i in tqdm(range(len(Teff))):
    line_width=1.47*(Teff[i]/5777)**np.random.uniform(9.8,0.2,31) * ((ulim-llim)/(2*600))
    for j in tqdm(range(len(files_path))): #    for j in np.array([0]): #
        
        Freqs=pd.read_csv(files_path[j])
        peak_num=Freqs.shape[0]
        freq0=Freqs[Freqs['L']==0]['Freq']
        freq1=Freqs[Freqs['L']==1]['Freq']
        freq2=Freqs[Freqs['L']==2]['Freq']

        amp_c=10
        Amp0 = amp_c * CalAmp(freq0,guess_l0)
        print(Amp0)
        # Amp1=amp_c*CalAmp(freq1,guess_l1)
        # Amp2=amp_c*CalAmp(freq2,guess_l2)
        Amp0_ = np.insert(Amp0,5,Amp0[5])
        Amp1 = 1.7 * Amp0_ #(1.7 + (0.2*np.random.randn(11)-0.1)) * Amp0_
        Amp2 = 0.65 * Amp0 #(0.65 + (0.2*np.random.random_sample(10)-0.1)) * Amp0

        Amp=np.hstack((Amp1,Amp0,Amp2))
        #print(Amp)

        for k in range(peak_num): 
            x,amp=LorentzProfile(Amp[k],Freqs['Freq'][k],line_width[k],llim,ulim,count)
            #print(amp)
            peak = np.array(np.vstack((x,amp)))
            peak = peak.T
            np.save(peak_path+'/peak_%s.npy' %(k),peak)
            # peak=pd.DataFrame({'x':x,'amp':amp})
            # #print(peak)
            # peak.to_csv(peak_path+'/peak_%s.csv' %(k))
            # plt.plot(x,amp)
        
            # plt.savefig(peakfig_path+'/peak_%s_%s.eps' %(Freqs['Freq'][i],Teff[j]))
            #plt.show()
        print('Done')
        peak_name,peak_path_=file_name(peak_path,'npy')
        #print(peak_name)
        
        blank=np.zeros(points_num)
        white_noise=wn_c*np.random.chisquare(2,points_num)
        ps=blank        
        for l in peak_path_:
            temp=np.load(l)
            amp=temp[:,1]
            ps=ps+amp
            #plt.plot(x,ps)
            #plt.show()
        ps=ps*white_noise

        # no smooth and no normalize
        ps = SmoothWrapper(x,ps,(ulim-llim)/600,'bartlett',s_interval)
        #ps = ps / np.max(ps)
        # plt.plot(x,ps)
        # plt.show()
        result=np.array(np.vstack((freq_miuHz,ps)))
        result=result.T
        np.save(ps_path+'/ps_%s_%s.npy' %(Teff[i],files_name[j]),result)
        # result=pd.DataFrame({'freq':freq_miuHz,'psd':ps})
        # result.to_csv(ps_path+'/ps_%s_%s.power' %(Teff[i],files_name[j]))
        print('d')
#print(line_width)



# %%
