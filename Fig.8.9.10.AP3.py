#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:59:33 2022

@author: mira
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:03:56 2021
Read the scalar output of the SLope cases of the Paper2 simulations.
Plots continental scale SLR. 
@author: mira
"""




import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.io import netcdf
import numpy as np
import matplotlib
from netCDF4 import Dataset
import pandas as pd
import matplotlib.colors as colors
import matplotlib.cm as mplcm
from matplotlib.pyplot import cm
import plotly.express as px
import seaborn as sns
from scipy import interpolate

plt.set_cmap('seismic')

ps = [0.15, 0.29, 0.4, 0.46, 0.51, 0.54, 0.6, 0.63, 0.64, 0.69, 0.7, 0.74, 0.76, 0.79, 0.79, 0.83, 0.85, 0.86, 0.88, 0.9, 0.92, 0.95, 0.96, 0.98, 1.0]
gammas = [2954923, 3886395, 3440211, 4205230, 3677928, 5175963, 1560081, 7280916, 3139194, 2760685, 5321878, 8654548, 4609682, 2200776, 2640377, 7593133,2450186, 4167483, 2098892,8939808, 5864477,6598244, 4849305,1710386 ,6285577]  
models = ['BCC-CSM2-MR','CAMS-CSM1-0','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','GFDL-CM4','GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','NESM3', 'Ctrl' ]

# conversions.
mm_equiv = 360.e12   # mass of ice corresponding to 1mm of eustatic sea level rise
kg_to_gt = 1.e-12    # conversion from kg to Gt

NumSamples = 25    # number of p and gamma combinations.
NumYears = 526
NumModels = 14
NUM_COLORS =  NumSamples

### PLOT A TIME SERIES OF SLR WITH NEW FIGURE FOR EACH MODEL ####

SLC = np.zeros((NumYears,NumSamples,NumModels))

expNum = 0
from scipy import interpolate

for model in models:
 
    for i in range(0,NumSamples):
        filename = (str(model) + '_p_' + str(ps[i]) + '_gamma_' + str(gammas[i]) + '.scalars.nc')
        g = Dataset('../../' + str(model) + '/' + filename ,'r')
        MAF = g.variables['imass_above_flotation'][:]
        

        slc = -(MAF[:] - MAF[0])/mm_equiv
        
        
        
        if len(slc)<NumYears:
            print('I am too short!')
            print('My model name is ', model)
            print('I should be 526 long but i am', len(slc))
            print('I am being extrapolated!')


    
            x = np.arange(0, len(slc))
            f = interpolate.interp1d(x, slc ,fill_value='extrapolate')
            xnew = np.arange(0,NumYears)
            ynew = f(xnew)   # use interpolation function returned by `interp1d`
            
            slc = ynew
            
        
        SLC[:,i,expNum] = slc
        
    expNum = expNum + 1



# Read MAF (Mass above flotation)
ctrl_data = np.genfromtxt('../../MAF/MAF_continent_Ctrl.txt', dtype = None, delimiter = ',')  # Open CTRL RUN MAF files.







###############################################################################
######################## PLOTS ################################################
###############################################################################
  



###### MAKE FIGURE APPENDIX FIG 2#######


############################ FIGURE APPENDIX 2A ########################################
## Try doing a mean time series plot for reach model, along with a vertical range for each model on the right. 
plt.figure(figsize = (5,5))
n = 14  # number of curves to plot. (iterating through unique colors)


color = plt.cm.tab20(np.linspace(0,1,n))
k=0
Ymean = np.zeros((NumYears,NumModels-1))
 
x = np.linspace(0,525, 526)
# go over NumModels -1 to get rid of control
for i in range(0,NumModels-1):
 
    Ymean[:,i] = np.mean(SLC[:,:,i], axis=1)
    plt.plot(x, Ymean[:,i], alpha=0.8, color = color[k], linewidth = 2.5)# make it a regular scale

    


    i = i+1
    k = k+1

plt.legend(['BCC-CSM2-MR','CAMS-CSM1-0','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','GFDL-CM4','GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','NESM3'], fontsize = 11, frameon= False)

#plt.title('Sea Level Contribution Range (mm)', fontsize = 15)
plt.xlabel('Year', fontsize = 18)
plt.ylabel('Model-Mean Sea Level Rise (mm)', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.text(x=350,y = 2000, s = '(a)', fontsize = 16)
plt.show()




###### MAKE FIGURE APPENDIX 2B #######
 

plt.figure(figsize = (5,5))
n = 14  # number of curves to plot. (iterating through unique colors)

color = plt.cm.tab20(np.linspace(0,1,n))
k=0
Ymean = np.zeros((NumYears,NumModels-1))
 
x = np.linspace(0,525, 526)
# go over NumModels -1 to get rid of control
for i in range(0,NumModels-1):
 
    Ymean[:,i] = np.mean(SLC[:,:,i], axis=1)

    # plot on log log scale
    plt.loglog(x, Ymean[:,i], alpha=0.8, color = color[k], linewidth = 2.5) # make it a log-log scale
    

    i = i+1
    k = k+1


#plt.title('Sea Level Contribution Range (mm)', fontsize = 15)
plt.xlabel('Year', fontsize = 18)
plt.ylabel('Model-Mean Sea Level Rise (mm)', fontsize = 18)
plt.text(x=1, y = 1000, s = '(b)', fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
#plt.text(x=0.001, y = 0.4, s = '(b)', fontsize = 15, transform = ax.transAxes)
plt.show()





###### MAKE FIGURE APPENDIX 2C #######
###############################################################################
# compute the gradient of the mean slr curves to get an idea of the rate of sea level rise. 
plt.figure(figsize = (5,5))
k=0
for i in range(0,13):
    plt.plot(np.gradient(Ymean[:,i]), alpha=0.8, color = color[k], linewidth = 2.5)
    k=k+1

plt.show()
# plt.legend(['BCC-CSM2-MR','CAMS-CSM1-0','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','GFDL-CM4','GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','NESM3'], loc = 'lower right')
plt.ylabel('Rate of Change in SLR (mm/yr)', fontsize = 18)
plt.xlabel('Year', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.text(x=1, y = 4.8, s = '(c)', fontsize = 16)

##############################################################################







### MAKE FIGURE 8 ##########
 
j = 0
fig, axs = plt.subplots(4,4, figsize=(15, 10), facecolor='w', edgecolor='k', sharey = True, sharex = True)
axs = axs.ravel()
models = ['BCC-CSM2-MR','CAMS-CSM1-0','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','GFDL-CM4','GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','NESM3']

for model in models:
    
    FINAL_SLC = np.zeros(len(ps))
    


    for i in range(0,NumSamples):
    
        filename = (str(model) + '_p_' + str(ps[i]) + '_gamma_' + str(gammas[i]) + '.scalars.nc')
        g = Dataset('../../' + str(model) + '/' + filename ,'r')
        MAF = g.variables['imass_above_flotation'][:]

        #plt.plot(MAF)

        slc = -(MAF[:] - MAF[0])/mm_equiv
        
        # if the run goes all the way to 526 time steps (final) then take the 511th time slice to make it 2500. There is one run that only made it to 508 so thats why i have the conditional to allow that one in still.
        if len(MAF) >= 508: 
            final_slc = slc[-1]
        else:
           final_slc = float("NaN")
        
        FINAL_SLC[i] = final_slc
        
    #c = FINAL_SLC # WITHOUT CTRL REMOVED
    
    # if you want to remove the control from the runs...
    if model == 'Ctrl':
        c = FINAL_SLC # don't remove control from itself.
    else:
        c = FINAL_SLC - (ctrl_data[:,4]/mm_equiv) # WITH CTRL REMOVED

    sct = axs[j].scatter(ps, gammas, c = c, cmap='coolwarm', s=150) 
    #axs[j].set_title(str(model))
    axs[j].set_title(str(model), loc='left', y=1.0, pad=-14)
    axs[j].set_xlabel('p', fontsize = 15)
    axs[j].set_ylabel('$\gamma _0$', fontsize = 15)
    axs[j].ticklabel_format(axis='y', style='sci', scilimits = (4,6))
    
    #plt.scatter(ps, gammas, c = FINAL_SLC, cmap='coolwarm', s=150, vmin=0, vmax=1500) 
    cbar = fig.colorbar(sct, ax=axs[j])
    #cbar.set_label('mm', labelpad = -20, rotation=90)
    cbar.ax.set_title('mm')
    #cbar.set_xlabel('mm', labelpad = -20, rotation=90)
    
    j=j+1
    
FINAL_SLC_CTRL = (-0.60473138, -1.42936516, -1.89360332, -0.93458492, -1.06896961,
       -1.83862782, -0.14660156, -1.78365219, -1.64315903, -0.34207028,
       -0.91625971, -1.21557117, -1.64315903, -0.6413818 , -2.64493632,
       -1.76532698, -1.0628612 , -0.83685052, -0.80630851, -1.25832999,
       -2.00966287, -1.03842771, -1.18502915, -0.98345208, -1.85084462)

sct = axs[13].scatter(ps, gammas, c = FINAL_SLC_CTRL, cmap='Reds_r', s=150, vmin = -3, vmax = 0) 
cbar = fig.colorbar(sct, ax = axs[13])
cbar.ax.set_title('mm')
axs[13].set_xlabel('p', fontsize = 15)
axs[13].set_ylabel('$\gamma _0$', fontsize = 15)


axs[12].set_title('',loc='left', y=1.0, pad=-24)
axs[13].set_title('', loc='left', y=1.0, pad=-24)
axs[12].text(0.15, 0.89, 'NESM3', horizontalalignment='center', verticalalignment='center', transform=axs[12].transAxes, fontsize = 12 )
axs[13].text(0.2, 0.89, 'CONTROL', horizontalalignment='center', verticalalignment='center', transform=axs[13].transAxes, fontsize = 12 )

axs.flat[-1].set_visible(False) # to remove last plot
axs.flat[-2].set_visible(False) # to remove last plot        
plt.tight_layout
plt.show()
#fig.suptitle('Sea level rise (mm) at 2500', fontsize=16)
    







########################### DO SOME QUICK FITS ##############################

import scipy
from scipy import stats
def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2, slope, std_err








########### MAKE FIGURE 7A #########
##### PLOT A VIOLIN PLOT OF THE FINAL DISTRIBUTION OF SLR FOR EACH MODEL

j = 0
# initialize arrays for R2 vlaues
NumModels = 14
FINAL_SLC = np.zeros([NumSamples, NumModels])

color = plt.cm.tab20(np.linspace(0,1,NumModels))
k=0

for model in models:

    for i in range(0,NumSamples):
    
        filename = (str(model) + '_p_' + str(ps[i]) + '_gamma_' + str(gammas[i]) + '.scalars.nc')
        g = Dataset('../../' + str(model) + '/' + filename ,'r')
        MAF = g.variables['imass_above_flotation'][:]

        #plt.plot(MAF)

        slc = -(MAF[:] - MAF[0])/mm_equiv
        
        # if the run goes all the way to 526 time steps (final) then take the 525tgh time slice to make it 2500. There is one run that only made it to 508 so thats why i have the conidtional to allow that one in still.
        if len(MAF) >= 508: 
            #final_slc = slc[-1]  # choose year 2500
            final_slc = slc[99] # choose year 2100
        else:
            final_slc = float("NaN")
        
        FINAL_SLC[i,k] = final_slc
        

        # FINAL_SLC = FINAL_SLC # WITHOUT CTRL REMOVED
    
        # if you want to remove the control from the runs...
        if model == 'Ctrl':
            c = FINAL_SLC # don't remove control from itself.
        else:
            c = FINAL_SLC[:,k] - (ctrl_data[:,4]/mm_equiv) # WITH CTRL REMOVED
    

   
    k=k+1

plt.figure(figsize=(10,10))



ax = sns.violinplot(data=FINAL_SLC[:,0:13],
                    scale="width", color="0.8", palette= cm.tab20(np.linspace(0,1,14)))
ax.set_xticklabels(['BCC-CSM2-MR','CAMS-CSM1-0','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','GFDL-CM4','GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','NESM3'], rotation = 90, fontsize = 11)


# for regular scale
plt.ylabel('Sea level rise (mm)', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.text(x =0 , y = 325 , s = 'Year 100',fontsize = 15)
plt.text(x =0, y = 355, s = '(a)', fontsize = 15)

plt.gcf().subplots_adjust(bottom=0.15)




########### MAKE FIGURE 7b #########
##### PLOT A VIOLIN PLOT OF THE FINAL DISTRIBUTION OF SLR FOR EACH MODEL

j = 0
# initialize arrays for R2 vlaues
NumModels = 14
FINAL_SLC = np.zeros([NumSamples, NumModels])


color = plt.cm.tab20(np.linspace(0,1,NumModels))
k=0

for model in models:

    for i in range(0,NumSamples):
    
        filename = (str(model) + '_p_' + str(ps[i]) + '_gamma_' + str(gammas[i]) + '.scalars.nc')
        g = Dataset('../../' + str(model) + '/' + filename ,'r')
        MAF = g.variables['imass_above_flotation'][:]

        #plt.plot(MAF)

        slc = -(MAF[:] - MAF[0])/mm_equiv
        
        # if the run goes all the way to 526 time steps (final) then take the 525tgh time slice to make it 2500. There is one run that only made it to 508 so thats why i have the conidtional to allow that one in still.
        if len(MAF) >= 508: 
            final_slc = slc[-1]  # choose year 2500
            #final_slc = slc[99] # choose year 2100
        else:
            final_slc = float("NaN")
        
        FINAL_SLC[i,k] = final_slc
        

        # FINAL_SLC = FINAL_SLC # WITHOUT CTRL REMOVED
    
        # if you want to remove the control from the runs...
        if model == 'Ctrl':
            c = FINAL_SLC # don't remove control from itself.
        else:
            c = FINAL_SLC[:,k] - (ctrl_data[:,4]/mm_equiv) # WITH CTRL REMOVED
    

   
    k=k+1
   

plt.figure(figsize = (10,10))

ax = sns.violinplot(data=FINAL_SLC[:,0:13],
                    scale="width", color="0.8", palette= cm.tab20(np.linspace(0,1,14)))
ax.set_xticklabels(['BCC-CSM2-MR','CAMS-CSM1-0','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','GFDL-CM4','GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','NESM3'], rotation = 90, fontsize = 11)

# for regular scale
plt.ylabel('Sea level rise (mm)', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
#plt.text(x =0 , y = 350 , s = 'Year 2100',fontsize = 15)
plt.text(x =0 , y = 3300 , s = 'Year 500',fontsize = 15)
plt.text(x =0, y = 3550, s = '(b)', fontsize = 15)

plt.gcf().subplots_adjust(bottom=0.15)  # change the space at hte bottom so that all the labels show up and arent cut off along the bottom.




##############3 FIG 9 COMBINED A AND B ###############


   
## MAKE FIGURE 9A ########
########################## PLOT GAMMA VS SLR #################################
j = 0
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='w', edgecolor='k')

n=14 #models
color = plt.cm.tab20(np.linspace(0,1,n))
k=0
for model in models:
    
    FINAL_SLC = np.zeros(len(ps))

    for i in range(0,NumSamples):
    
        filename = (str(model) + '_p_' + str(ps[i]) + '_gamma_' + str(gammas[i]) + '.scalars.nc')
        g = Dataset('../../' + str(model) + '/' + filename ,'r')
        MAF = g.variables['imass_above_flotation'][:]

        #plt.plot(MAF)

        slc = -(MAF[:] - MAF[0])/mm_equiv
        
        # if the run goes all the way to 526 time steps (final) then take the 525tgh time slice to make it 2500. There is one run that only made it to 508 so thats why i have the conidtional to allow that one in still.
        if len(MAF) >= 508: 
            final_slc = slc[-1]
        else:
            final_slc = float("NaN")
        
        FINAL_SLC[i] = final_slc
        

    #FINAL_SLC = FINAL_SLC # WITHOUT CTRL REMOVED
    
    # if you want to remove the control from the runs...
    if model == 'Ctrl':
        c = FINAL_SLC # don't remove control from itself.
    else:
        c = FINAL_SLC - (ctrl_data[:,4]/mm_equiv) # WITH CTRL REMOVED
    
    #col = next(color)

    sct = ax1.scatter(gammas, FINAL_SLC, color = color[k])
    #plt.title('SLR (mm )
    ax1.set_xlabel('$\gamma _0$', fontsize = 15)
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    matplotlib.rcParams['font.size']=12

    ax1.set_ylabel('Sea level rise (mm)', fontsize = 15)
    # plt.ticklabel_format(fontsize = 13)

    
    # convert p, gamma to arrays instead of lists for ease of other computations.
    gammas = np.array(gammas)
    
    # fit a best fit line
    idx = np.isnan(FINAL_SLC); #find nans
    m, b = np.polyfit(gammas[~idx], FINAL_SLC[~idx], 1) # dont include nans in the fit
    #ps = pd.Series(ps)
    ax1.plot(gammas, m*gammas + b, color = color[k])

    
    #plt.scatter(ps, gammas, c = FINAL_SLC, cmap='coolwarm', s=150, vmin=0, vmax=1500) 
    #fig.colorbar(sct, ax=axs[j])
    plt.show()
    #fig.suptitle('Multi-Model Sea level rise (mm) at 2500 as a function of gamma', fontsize=16)

    k=k+1
    j=j+1
        
plt.tight_layout(rect=[0,0,0.75,1])
ax1.text(x =1500000, y = 3000, s = '(a)', fontsize = 15)




## MAKE FIGURE 9B ########
########################## PLOT P VS SLR ##############
j = 0

n=14 #models
color = plt.cm.tab20(np.linspace(0,1,n))
k=0


for model in models:

    FINAL_SLC = np.zeros(len(ps))

    for i in range(0,NumSamples):

        filename = (str(model) + '_p_' + str(ps[i]) + '_gamma_' + str(gammas[i]) + '.scalars.nc')
        g = Dataset('../../' + str(model) + '/' + filename ,'r')
        MAF = g.variables['imass_above_flotation'][:]

        #plt.plot(MAF)

        slc = -(MAF[:] - MAF[0])/mm_equiv

        # if the run goes all the way to 526 time steps (final) then take the 525tgh time slice to make it 2500. There is one run that only made it to 508 so thats why i have the conidtional to allow that one in still.
        if len(MAF) >= 508: 
            final_slc = slc[-1]
        else:
            final_slc = float("NaN")

        FINAL_SLC[i] = final_slc


    #FINAL_SLC = FINAL_SLC # WITHOUT CTRL REMOVED

    # if you want to remove the control from the runs...
    if model == 'Ctrl':
        c = FINAL_SLC # don't remove control from itself.
    else:
        c = FINAL_SLC - (ctrl_data[:,4]/mm_equiv) # WITH CTRL REMOVED

    #col = next(color)
    sct = ax2.scatter(ps, FINAL_SLC, color = color[k])
    #plt.title('SLR (mm )
    ax2.set_xlabel('p', fontsize = 15)
    ax2.set_ylabel('Sea level rise (mm)', fontsize = 15)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))


    # convert p, gamma to arrays instead of lists for ease of other computations.
    ps = np.array(ps)

    # fit a best fit line
    idx = np.isnan(FINAL_SLC); #find nans
    m, b = np.polyfit(ps[~idx], FINAL_SLC[~idx], 1) # dont include nans in the fit
    #ps = pd.Series(ps)

    ax2.plot(ps, m*ps + b, color = color[k])


    #plt.scatter(ps, gammas, c = FINAL_SLC, cmap='coolwarm', s=150, vmin=0, vmax=1500) 
    #fig.colorbar(sct, ax=axs[j])
    plt.show()
    #fig.suptitle('Multi-Model Sea level rise (mm) at 2500 as a function of P', fontsize=16)


    j=j+1
    k=k+1

plt.tight_layout(rect=[0,0,0.75,1])
# plt.legend( ['BCC-CSM2-MR, $R^2$=0.06 ','CAMS-CSM1-0, $R^2$=0.08','CESM2, $R^2$=0.23','CNRM-CM6-1, $R^2$=0.15','CNRM-ESM2-1, $R^2$=0.18','CanESM5, $R^2$=0.19','EC-Earth3, R$R^2$=0.23','EC-Earth3-Veg, $R^2$=0.21','GFDL-CM4, $R^2$=0.04','GFDL-ESM4, $R^2$=0.16','IPSL-CM6A-LR, $R^2$=0.21','MPI-ESM1-2-HR, $R^2$=0.17','NESM3, $R^2$=0.06', 'Ctrl' ], loc=(1.04,0))
plt.text(x =0.14, y = 3000, s = '(b)', fontsize = 15)



plt.subplots_adjust(bottom=0.1)

