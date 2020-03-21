#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")

path=os.getcwd()
freqpath = path + '/freq/'
echellepath = path + '/echelle/'
delta_omega=1
n=10
gmode_num=1


# In[2]:


def produce_l0(bias0): # 0.25~0.3
    #global omega0
    xx=np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4])
    fx=1/(11-10*xx**2)
    omega0=bias0+np.arange(0,(n)*delta_omega,delta_omega)+fx
    return omega0
def produce_l2(bias2): # 0.13~0.17
    
    xx=np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4])
    fx=1/(11-10*xx**2)
    omega2=bias2+np.arange(0,(n)*delta_omega,delta_omega)+fx
    return omega2

def produce_l1(alpha=np.array([None]),gmode_index=np.array([5]),gmodeFeq=0.6,surfacel1=False):
    omega=1.75+np.arange(1,(n-gmode_index.shape[0]+2)*delta_omega,delta_omega)
    print(omega)
    omega_n=omega[gmode_index-1]+gmodeFeq
    new_omega=np.hstack((omega,omega_n))
    new_omega.sort()
    print(new_omega.shape)
    A=np.zeros((n+1,n+1))
    for i in range(gmode_index.shape[0]):
        A[gmode_index[i],:]=A[:,gmode_index[i]]=-alpha[i]

    for i in range(len(new_omega)):
        A[i,i]=new_omega[i]
    eigenfreq,eigenvector=np.linalg.eig(A)
    eigenfreq.sort()  
    #print(eigenfreq.shape)
    if surfacel1==True:
        #xx=np.concatenate((np.repeat(0,8),[0.1,0.2,0.3,0.4,0.5,0.6]))
        xx=np.array([-0.5,-0.4,-0.3,-0.2,-0.1,-0.1,0,0.1,0.2,0.3,0.4])
        fx=1/(11-10*xx**2)
        eigenfreq=eigenfreq+fx
    else:
        eigenfreq=eigenfreq
    return eigenfreq,omega,A


# In[3]:


# a,b,c = produce_l1(alpha=np.array([0.3]),gmode_index=np.array([5]),gmodeFeq=0.6,surfacel1=False)


# In[20]:


omega_n_set=np.array([0.1,0.2,0.3,0.4])
for alpha in np.linspace(0.2,0.4,10,endpoint=True):
    for omega_n in omega_n_set:
        for bias in np.linspace(2.2,2.2+0.04,4,endpoint=True):
            #for bias0 in np.linspace(2.12,2.12+0.04,4,endpoint=True):
            eigenfreq,omega,_=produce_l1([alpha],np.array([5]),omega_n,surfacel1=True)
            omega0=produce_l0(bias)
            omega2=produce_l2(bias-0.1)
            freq=np.hstack((eigenfreq,omega0,omega2))
            #print()
            #freq=pd.DataFrame(freq_temp,columns=['Freq'])
            '''
            freql=np.concatenate((np.repeat(1,n),np.repeat(0,n-gmode_num),np.repeat(2,n-gmode_num)),axis=0)
            result=np.concatenate((freq.reshape(40,1),freql.reshape(40,1)),axis=1)
            np.save(path+'/freq_csv_mod1/freqs_%s_%s_%s_%s.npy' %(alpha,omega_n,bias0,bias2),result)
            '''
            freq_temp=np.hstack((eigenfreq,omega0,omega2))
            freq=pd.DataFrame(freq_temp,columns=['Freq'])
            freq['L']=np.concatenate((np.repeat(1,n+1),np.repeat(0,n),np.repeat(2,n)),axis=0)
            
            freq.to_csv(freqpath+'freqs_%.2f_%.2f_%.2.csv' %(alpha,omega_n,bias))

            #plt.axis([0,1,0,15])
            colors = ['c','b','orange','y' ]
            markers = ['o','s','^','v']
            fig = plt.figure(figsize=(6,4))
            plt.rc('font',family='Times New Roman')

            ax = fig.add_subplot(111)
            #ax.title('alpha=%s_%s_%s' %(alpha,omega_n,bias))
            # ax.scatter((eigenfreq)%delta_omega,eigenfreq)
            # ax.scatter((omega0)%delta_omega,omega0)
            # ax.scatter((omega2)%delta_omega,omega2)
            ax.scatter((eigenfreq)%delta_omega,eigenfreq,c=colors[1],marker=markers[1])
            ax.scatter((omega0)%delta_omega,omega0,c=colors[0],marker=markers[0])
            ax.scatter((omega2)%delta_omega,omega2,c=colors[2],marker=markers[2])
            plt.savefig(echellepath+'echelle_%.2f_%.2f_%.2f.png' %(alpha,omega_n,bias)) #eps
            plt.close()


# In[ ]:


# def plott(surface=True):   
#     eigenfreq,omega,A=produce_l1([0.5],gmodeFeq=-0.1,surfacel1=surface)
#     omega0=produce_l0(0.15+2)
#     omega2=produce_l2(0.05+2)
#     #print(eigenfreq,omega0,omega2)
#     eigenfreq.shape
#     #print(A)
#     plt.axis([0,1,0,15])
#     plt.scatter((eigenfreq)%delta_omega,eigenfreq)
#     plt.scatter((omega0)%delta_omega,omega0)
#     plt.scatter((omega2)%delta_omega,omega2)
#     #plt.savefig('/Users/duminghao/Desktop/ML_Mode_Classify/simu_subgiants_freqs/temp_.eps')
# plott()


# # In[ ]:


# plott(False)


# # In[5]:


# fig = plt.figure(figsize=(6,4))
# plt.rc('font',family='Times New Roman')

# ax1 = fig.add_subplot(121)
# eigenfreq,omega,_ = produce_l1([0.05],np.array([np.int(5)]),0.0,surfacel1=False)
# ax1.scatter((eigenfreq-0.15)%delta_omega,eigenfreq,c='k')

# ax1.text(0.2,10,'$\\alpha$ = 0.05')
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.set_xlabel('$\omega^2 \ module\ \Delta\omega^2=1$')
# ax1.set_ylabel('$\omega^2$')
# ax1.set_xlim([0.1,1.1])

# ax2 = fig.add_subplot(122)
# eigenfreq,omega,_ = produce_l1([0.35],np.array([np.int(5)]),0.0,surfacel1=False)
# ax2.scatter((eigenfreq-0.15)%delta_omega,eigenfreq,c='k')

# ax2.text(0.2,10,'$\\alpha$ = 0.35')
# ax2.set_xticks([])
# ax2.set_yticks([])
# ax2.set_xlabel('$\omega^2 \ module\ \Delta\omega^2=1$')
# ax2.set_ylabel('$\omega^2$')
# ax2.set_xlim([0.1,1.1])

#plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
# plt.tight_layout()
# plt.savefig('./lecture_figures/avoid_crossing.eps', dpi=200)


# # In[16]:


# colors = ['c','b','orange','y' ]
# markers = ['o','s','^','v']
# omega_n_set=np.array([0.1,0.4])
# fig = plt.figure(figsize=(12,7))
# count = 0
# for alpha in np.linspace(0.2,0.4,2,endpoint=True):
#     for omega_n in omega_n_set:
#         for bias in np.linspace(2.2,2.2+0.04,2,endpoint=True):
#             eigenfreq,omega,_=produce_l1([alpha],np.array([5]),omega_n,surfacel1=False)
#             omega0=produce_l0(bias)
#             omega2=produce_l2(bias-0.1)
#             count += 1
#             ax = fig.add_subplot(2,4,count)
#             ax.scatter((eigenfreq)%delta_omega,eigenfreq,c=colors[1],marker=markers[1])
#             ax.scatter((omega0)%delta_omega,omega0,c=colors[0],marker=markers[0])
#             ax.scatter((omega2)%delta_omega,omega2,c=colors[2],marker=markers[2])
#             ax.text(0.35,10,'$\\alpha=%s$' %alpha)
#             ax.text(0.35,9,'$\\omega_n=+%s$' %omega_n)
#             ax.text(0.35,8,'$\\epsilon={:0.2f}$'.format(bias-2))
#             ax.set_xticks([])
#             ax.set_yticks([])
# plt.tight_layout()            
# plt.savefig('./lecture_figures/echelle.eps', dpi=200)


# In[ ]:


#color='', marker='o', edgecolors='g'

