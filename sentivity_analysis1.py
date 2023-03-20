# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:07:23 2019

@author: HP
"""


import math
import matplotlib.pyplot as plt
from SALib.analyze import sobol
import seaborn as sn
import numpy as np
from scipy import stats
import pandas as pd
from statistics import mean, stdev
import matplotlib.ticker as ticker

def output_uncert(x):
    y = math.sqrt(x[3])*100/x[2]
    return y

data5all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\5, all\NNdata5all.xlsx')
data5kinetic = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\5, kinetic\NNdata5kinetic.xlsx')
data5reserv = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\5, reserv\NNdata5reserv.xlsx')
data5op = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\5, op\NNdata5op.xlsx')
data10all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\10, all\NNdata10all.xlsx')
data10kinetic = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\10, kinetic\NNdata10kinetic.xlsx')
data10reserv = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\10, reserv\NNdata10reserv.xlsx')
data10op = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\10, op\NNdata10op.xlsx')
data15all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\15, all\NNdata15all.xlsx')
data15kinetic = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\15, kinetic\NNdata15kinetic.xlsx')
data15reserv = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\15, reserv\NNdata15reserv.xlsx')
data15op = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\15, op\NNdata15op.xlsx')
data20all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\20, all\NNdata20all.xlsx')
data20kinetic = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\20, kinetic\NNdata20kinetic.xlsx')
data20reserv = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\20, reserv\NNdata20reserv.xlsx')
data20op = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\20, op\NNdata20op.xlsx')
data25all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\25, all\NNdata25all.xlsx')
data25kinetic = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\25, kinetic\NNdata25kinetic.xlsx')
data25reserv = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\25, reserv\NNdata25reserv.xlsx')
data25op = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\25, op\NNdata25op.xlsx')
datacase1 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase\NNdata.xlsx')
datacase2 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase2\NNdata.xlsx')
datacase3 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase3\NNdata.xlsx')
inputuncert = [0,5,10,15,20,25]
# uncertainty, P90,P10,P50

result5all = stats.describe(data5all['recovery'], ddof=1, bias=False) #nobs, minmax, mean, variance, skewnwess, kurtosis
result5kinetic = stats.describe(data5kinetic['recovery'], ddof=1, bias=False)
result5res = stats.describe(data5reserv['recovery'], ddof=1, bias=False)
result5op = stats.describe(data5op['recovery'], ddof=1, bias=False)
result10all = stats.describe(data10all['recovery'], ddof=1, bias=False)
result10kinetic = stats.describe(data10kinetic['recovery'], ddof=1, bias=False)
result10res = stats.describe(data10reserv['recovery'], ddof=1, bias=False)
result10op = stats.describe(data10op['recovery'], ddof=1, bias=False)
result15all = stats.describe(data15all['recovery'], ddof=1, bias=False)
result15kinetic = stats.describe(data15kinetic['recovery'], ddof=1, bias=False)
result15res = stats.describe(data15reserv['recovery'], ddof=1, bias=False)
result15op = stats.describe(data15op['recovery'], ddof=1, bias=False)
result20all = stats.describe(data20all['recovery'], ddof=1, bias=False)
result20kinetic = stats.describe(data20kinetic['recovery'], ddof=1, bias=False)
result20res = stats.describe(data20reserv['recovery'], ddof=1, bias=False)
result20op = stats.describe(data20op['recovery'], ddof=1, bias=False)
result25all = stats.describe(data25all['recovery'], ddof=1, bias=False)
result25kinetic = stats.describe(data25kinetic['recovery'], ddof=1, bias=False)
result25res = stats.describe(data25reserv['recovery'], ddof=1, bias=False)
result25op = stats.describe(data25op['recovery'], ddof=1, bias=False)
resultcase1 = stats.describe(datacase1['recovery'], ddof=1, bias=False)
resultcase2 = stats.describe(datacase2['recovery'], ddof=1, bias=False)
resultcase3 = stats.describe(datacase3['recovery'], ddof=1, bias=False)

quantile5all = np.quantile(data5all['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile5kinetic = np.quantile(data5kinetic['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile5reserv = np.quantile(data5reserv['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile5op = np.quantile(data5op['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile10all = np.quantile(data10all['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile10kinetic = np.quantile(data10kinetic['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile10reserv = np.quantile(data10reserv['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile10op = np.quantile(data10op['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile15all = np.quantile(data15all['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile15kinetic = np.quantile(data15kinetic['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile15reserv = np.quantile(data15reserv['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile15op = np.quantile(data15op['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile20all = np.quantile(data20all['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile20kinetic = np.quantile(data20kinetic['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile20reserv = np.quantile(data20reserv['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile20op = np.quantile(data25all['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile25all = np.quantile(data25all['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile25kinetic = np.quantile(data25kinetic['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile25reserv = np.quantile(data25reserv['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantile25op = np.quantile(data25all['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase1 = np.quantile(datacase1['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase2 = np.quantile(datacase2['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase3 = np.quantile(datacase3['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])

#%%
# Figure 2

fig, axs = plt.subplots(2,2, figsize= (8,8), dpi = 1000)
#plt.title('Probability Distribution of Oil Recovery (varying all parameters)')
#bins10all = list(np.histogram(data10all['recovery'],bins = 200, density = 'false'))
#bins20all = list(np.histogram(data20all['recovery'],bins = 200, density = 'false'))
#bins30all = list(np.histogram(data30all['recovery'],bins = 200, density = 'false'))
#bins40all = list(np.histogram(data40all['recovery'],bins = 200, density = 'false'))
#sigma10all = math.sqrt(result10all[3])
#sigma20all = math.sqrt(result20all[3])
#sigma30all = math.sqrt(result30all[3])
#sigma40all = math.sqrt(result40all[3])
#plt.plot(bins10all[1], 1/(sigma10all * np.sqrt(2 * np.pi)) * np.exp( - (bins10all[1] - result10all[2])**2 / (2 * sigma10all**2) ),'r+-',linewidth=2)
#plt.plot(bins20all[1], 1/(sigma20all * np.sqrt(2 * np.pi)) * np.exp( - (bins20all[1] - result20all[2])**2 / (2 * sigma20all**2) ),'go--',linewidth=2)
#plt.plot(bins30all[1], 1/(sigma30all * np.sqrt(2 * np.pi)) * np.exp( - (bins30all[1] - result30all[2])**2 / (2 * sigma30all**2) ),'b^-.',linewidth=2)
#plt.plot(bins40all[1], 1/(sigma40all * np.sqrt(2 * np.pi)) * np.exp( - (bins40all[1] - result40all[2])**2 / (2 * sigma40all**2) ),'k*:',linewidth=2)
sns.distplot(data5all['recovery'], bins = 200, hist = False, color = 'blue', kde_kws={'linestyle':'--'}, ax = axs[0,0])
sns.distplot(data10all['recovery'], bins = 200, hist = False, color = 'red', kde_kws={'linestyle':'-.'}, ax = axs[0,0])
sns.distplot(data15all['recovery'], bins = 200, hist = False, color = 'green', kde_kws={'linestyle':':'}, ax = axs[0,0])
sns.distplot(data20all['recovery'], bins = 200, hist = False, color = 'orange', kde_kws={'linestyle':'-'}, ax = axs[0,0])
sns.distplot(data25all['recovery'], bins = 200, hist = False, color = 'violet', kde_kws={'linestyle':'-','marker':'.'}, ax = axs[0,0])
axs[0,0].set_title('a) Probability Distribution of Oil Recovery, % \n (All Parameters)', y = -0.35, fontsize = 12)
axs[0,0].set_xlabel('Oil Recovery, %', fontsize = 12)
axs[0,0].set_ylabel('Probability, %', fontsize = 12)
#axs[0,0].legend(labels=['η = 5%', 'η = 10%', 'η = 15%', 'η = 20%', 'η = 25%'])
axs[0,0].set_xlim([0,30])
axs[0,0].set_ylim([0,0.25])
#axs[0,0].set_xticks(fontsize = 14, rotation = 0)
#axs[0,0].set_yticks(fontsize = 14, rotation = 0)
axs[0,0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals = 0))

#plt.title('Probability Distribution of Oil Recovery (varying only micronial kinetic parameters)')
sns.distplot(data5kinetic['recovery'], bins = 200, hist = False, color = 'blue', kde_kws={'linestyle':'--'}, ax = axs[0,1])
sns.distplot(data10kinetic['recovery'], bins = 200, hist = False, color = 'red', kde_kws={'linestyle':'-.'}, ax = axs[0,1])
sns.distplot(data15kinetic['recovery'], bins = 200, hist = False, color = 'green', kde_kws={'linestyle':':'}, ax = axs[0,1])
sns.distplot(data20kinetic['recovery'], bins = 200, hist = False, color = 'orange', kde_kws={'linestyle':'-'}, ax = axs[0,1])
sns.distplot(data25kinetic['recovery'], bins = 200, hist = False, color = 'violet', kde_kws={'linestyle':'-','marker':'.'}, ax = axs[0,1])
axs[0,1].set_title('b) Probability Distribution of Oil Recovery, % \n (Microbial Kinetic Parameters)', y = -0.35, fontsize = 12)
axs[0,1].set_xlabel('Oil Recovery, %', fontsize = 12)
axs[0,1].set_ylabel('Probability, %', fontsize = 12)
axs[0,1].set_xlim([0,30])
axs[0,1].set_ylim([0,0.35])
#axs[0,1].set_xticks(fontsize = 14, rotation = 0)
#axs[0,0].set_yticks(fontsize = 14, rotation = 0)
axs[0,1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals = 0))

#plt.title('Probability Distribution of Oil Recovery (varying only reservoir parameters)')
sns.distplot(data5reserv['recovery'], bins = 200, hist = False, color = 'blue', kde_kws={'linestyle':'--'}, ax = axs[1,1])
sns.distplot(data10reserv['recovery'], bins = 200, hist = False, color = 'red', kde_kws={'linestyle':'-.'}, ax = axs[1,1])
sns.distplot(data15reserv['recovery'], bins = 200, hist = False, color = 'green', kde_kws={'linestyle':':'}, ax = axs[1,1])
sns.distplot(data20reserv['recovery'], bins = 200, hist = False, color = 'orange', kde_kws={'linestyle':'-'}, ax = axs[1,1])
sns.distplot(data25reserv['recovery'], bins = 200, hist = False, color = 'violet', kde_kws={'linestyle':'-','marker':'.'}, ax = axs[1,1])
axs[1,1].set_title('c) Probability Distribution of Oil Recovery, % \n (Reservoir Parameters)', y = -0.35, fontsize = 12)
axs[1,1].set_xlabel('Oil Recovery, %', fontsize = 12)
axs[1,1].set_ylabel('Probability, %', fontsize = 12)
#axs[1,0].legend(labels=['η = 5%', 'η = 10%', 'η = 15%', 'η = 20%', 'η = 25%'])
axs[1,1].set_xlim([0,20])
axs[1,1].set_ylim([0,0.65])
#axs[1,0].set_xticks(fontsize = 14, rotation = 0)
#axs[0,0].set_yticks(fontsize = 14, rotation = 0)
axs[1,1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals = 0))

#plt.title('Probability Distribution of %Recovery (varying only operational parameters)')
sns.distplot(data5op['recovery'], bins = 200, hist = False, color = 'blue', kde_kws={'linestyle':'--'}, ax = axs[1,0])
sns.distplot(data10op['recovery'], bins = 200, hist = False, color = 'red', kde_kws={'linestyle':'-.'}, ax = axs[1,0])
sns.distplot(data15op['recovery'], bins = 200, hist = False, color = 'green', kde_kws={'linestyle':':'}, ax = axs[1,0])
sns.distplot(data20op['recovery'], bins = 200, hist = False, color = 'orange', kde_kws={'linestyle':'-'}, ax = axs[1,0])
sns.distplot(data25op['recovery'], bins = 200, hist = False, color = 'violet', kde_kws={'linestyle':'-','marker':'.'}, ax = axs[1,0])
axs[1,0].set_title('d) Probability Distribution of Oil Recovery, % \n (Operational Parameters)', y = -0.35, fontsize = 12)
axs[1,0].set_xlabel('Oil Recovery, %', fontsize = 12)
axs[1,0].set_ylabel('Probability, %', fontsize = 12)
#axs[1,1].legend(labels=['η = 5%', 'η = 10%', 'η = 15%', 'η = 20%', 'η = 25%'])
axs[1,0].set_xlim([0,30])
axs[1,0].set_ylim([0,0.35])
#axs[1,1].set_xticks(fontsize = 14, rotation = 0)
#axs[0,0].set_yticks(fontsize = 14, rotation = 0)
axs[1,0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals = 0))
fig.legend(labels=['η = 5%', 'η = 10%', 'η = 15%', 'η = 20%', 'η = 25%'], bbox_to_anchor = (0.9,0), ncol = 5)    
fig.tight_layout()
#%%
# Figure 4

fig1, axs =  plt.subplots(2,2, figsize= (8,8), dpi = 1000)
axs[0,0].set_title('a) Oil Recovery, % vs Input Uncertainty \n (varying all parameters)', y = -0.35, fontsize = 12)
axs[0,0].plot(inputuncert,[11.0498085,quantile5all[0],quantile10all[0],quantile15all[0],quantile20all[0],quantile25all[0]],c='blue',marker='.')
axs[0,0].plot(inputuncert,[11.0498085,quantile5all[2],quantile10all[2],quantile15all[2],quantile20all[2],quantile25all[2]],c='green',marker='s')
axs[0,0].plot(inputuncert,[11.0498085,quantile5all[4],quantile10all[4],quantile15all[4],quantile20all[4],quantile25all[4]],c='red',marker='^')
axs[0,0].set_xlabel('Input Uncertainty, %', fontsize = 12)
axs[0,0].set_ylabel('Oil Recovery, %', fontsize = 12)
axs[0,0].set_xlim([0, 30])
axs[0,0].set_ylim([0, 30])



axs[0,1].set_title('b) Oil Recovery, % vs Input Uncertainty \n (varying only Microbial Kinetic parameters)', y = -0.35, fontsize = 12)
axs[0,1].plot(inputuncert,[11.0498085,quantile5kinetic[0],quantile10kinetic[0],quantile15kinetic[0],quantile20kinetic[0],quantile25kinetic[0]],c='blue',marker ='.')
axs[0,1].plot(inputuncert,[11.0498085,quantile5kinetic[2],quantile10kinetic[2],quantile15kinetic[2],quantile20kinetic[2],quantile25kinetic[2]],c='green',marker='s')
axs[0,1].plot(inputuncert,[11.0498085,quantile5kinetic[4],quantile10kinetic[4],quantile15kinetic[4],quantile20kinetic[4],quantile25kinetic[4]],c='red',marker='^')
axs[0,1].set_xlabel('Input Uncertainty, %', fontsize = 12)
axs[0,1].set_ylabel('Oil Recovery, %', fontsize = 12)
axs[0,1].set_xlim([0, 30])
axs[0,1].set_ylim([0, 30])


axs[1,0].set_title('c) Oil Recovery, % vs Input Uncertainty \n (varying only reservoir parameters)', y = -0.35, fontsize = 12)
axs[1,0].plot(inputuncert,[11.0498085,quantile5reserv[0],quantile10reserv[0],quantile15reserv[0],quantile20reserv[0],quantile25reserv[0]],c='blue', marker='.')
axs[1,0].plot(inputuncert,[11.0498085,quantile5reserv[2],quantile10reserv[2],quantile15reserv[2],quantile20reserv[2],quantile25reserv[2]],c='green',marker='s')
axs[1,0].plot(inputuncert,[11.0498085,quantile5reserv[4],quantile10reserv[4],quantile15reserv[4],quantile20reserv[4],quantile25reserv[4]],c='red',marker='^')
axs[1,0].set_xlabel('Input Uncertainty, %', fontsize = 12)
axs[1,0].set_ylabel('Oil Recovery, %', fontsize = 12)
axs[1,0].set_xlim([0, 30])
axs[1,0].set_ylim([0, 30])



axs[1,1].set_title('d) Oil Recovery, % vs Input Uncertainty \n (varying only operational parameters)', y = -0.35, fontsize = 12)
axs[1,1].plot(inputuncert,[11.0498085,quantile5op[0],quantile10op[0],quantile15op[0],quantile20op[0],quantile25op[0]],c='blue',marker ='.')
axs[1,1].plot(inputuncert,[11.0498085,quantile5op[2],quantile10op[2],quantile15op[2],quantile20op[2],quantile25op[2]],c='green',marker='s')
axs[1,1].plot(inputuncert,[11.0498085,quantile5op[4],quantile10op[4],quantile15op[4],quantile20op[4],quantile25op[4]],c='red',marker='^')
axs[1,1].set_xlabel('Input Uncertainty, %', fontsize = 12)
axs[1,1].set_ylabel('Oil Recovery, %', fontsize = 12)
axs[1,1].set_xlim([0, 30])
axs[1,1].set_ylim([0, 30])

fig1.legend(labels=['P90','P50','P10'], bbox_to_anchor = (0.7,0), ncol = 3)
fig1.tight_layout()
#%%
# Figure 3

fig2, axs = plt.subplots(1,2, figsize= (8,4), dpi = 1000)
inun = [*[0], *inputuncert]
axs[0].set_title('a) Input Uncertainty vs Output Uncertainty \n (varying all parameters)', y = -0.35, fontsize = 12)
axs[0].plot(inun,[0, output_uncert(result5all),output_uncert(result10all),output_uncert(result15all),output_uncert(result20all),output_uncert(result25all)], marker = 'o')
axs[0].plot(inun,[0,output_uncert(result5kinetic)
                         ,output_uncert(result10kinetic)
                         ,output_uncert(result15kinetic)
                         ,output_uncert(result20kinetic)
                         ,output_uncert(result25kinetic)],c='blue', marker = '.')
axs[0].plot(inun,[0,output_uncert(result5res)
                         ,output_uncert(result10res)
                         ,output_uncert(result15res)
                         ,output_uncert(result20res)
                         ,output_uncert(result25res)],c='green',marker='s')
axs[0].plot(inun,[0,output_uncert(result5op)
                         ,output_uncert(result10op)
                         ,output_uncert(result15op)
                         ,output_uncert(result20op)
                         ,output_uncert(result25op)],c='red',marker='^')
axs[0].legend(labels=['All parameters','Microbial Kinetic Parameters','Reservoir Parameters','Operational Parameters'], loc = 0, fontsize = 'small')
axs[0].set_xlabel('Input Uncertainty, %', fontsize = 12)
axs[0].set_ylabel('Output Uncertainty, %', fontsize = 12)
axs[0].set_xlim([0, 85])
axs[0].set_ylim([0, 85])

axs[1].set_title('b) 1-Cummulative Distribution Function of Oil \nRecovery, % (varying all parameters)', y = -0.35, fontsize = 12)
values,base = np.histogram(data5all['recovery'],bins=200)
values1,base1 = np.histogram(data10all['recovery'],bins=200)
values2,base2 = np.histogram(data15all['recovery'],bins=200)
values3,base3 = np.histogram(data20all['recovery'],bins=200)
values4,base4 = np.histogram(data25all['recovery'],bins=200)
cumulative = np.cumsum(values)
cumulative1 = np.cumsum(values1)
cumulative2 = np.cumsum(values2)
cumulative3 = np.cumsum(values3)
cumulative4 = np.cumsum(values4)
axs[1].set_xlabel('Oil Recovery, %', fontsize = 12)
axs[1].set_ylabel('1-Cummulative Probability Density', fontsize = 12) 
axs[1].plot(base[:-1], 1-(cumulative/len(data5all['recovery'])), c='blue', linestyle='--')
axs[1].plot(base1[:-1],1-(cumulative1/len(data10all['recovery'])), c='green', linestyle='-.')
axs[1].plot(base2[:-1],1-(cumulative2/len(data15all['recovery'])), c='red', linestyle=':')
axs[1].plot(base3[:-1],1-(cumulative3/len(data20all['recovery'])), c='orange', linestyle='-')
axs[1].plot(base4[:-1],1-(cumulative4/len(data25all['recovery'])), c='violet', linestyle='-', marker = '.')
axs[1].legend(labels=['η = 5%','η = 10%','η = 15%','η = 20%','η = 25%'])
axs[1].set_xlim([0, 60])
axs[1].set_ylim([0, 1])
fig2.tight_layout()

#%%
# Sobol graphs, Figure 5

Si5 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\May graphs\sobol sens\5.xlsx")
Si10 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\May graphs\sobol sens\10.xlsx")
Si15 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\May graphs\sobol sens\15.xlsx")
Si20 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\May graphs\sobol sens\20.xlsx")
Si25 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\May graphs\sobol sens\25.xlsx")
dfst = pd.read_excel(r"D:\Project\MC Simulation\MEOR\May graphs\sobol sens\Variable ranking.xlsx")

fig3, axs = plt.subplots(3,2, figsize= (10,20), dpi = 1000)
mylabels = ['Yxs','Yps','Kxs','Umax', 'Xi','Si','Ai','v','Uw','IFTmax','Swir','Sori']
# 5
axs[0,0].set_title('a)Sensitivity indices S1(green) and \n ST(green+white) for  η= 5% ', y = -0.25, fontsize = 12)
#plt.subplot2grid(shape = (5,2), loc = (0,0), colspan = 1, rowspan = 3)
p1=axs[0,0].bar(mylabels, Si5['ST'], color ='white', edgecolor = 'black')
p2=axs[0,0].bar(mylabels, Si5['S1'], color='#7eb54e', edgecolor = 'black')
ΣST = sum(Si5['ST'])
ΣS1 = sum(Si5['S1'])
axs[0,0].text(7, 0.25, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), style='italic', fontsize=12,
     bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs[0,0].set_xticklabels(mylabels, rotation = 45)
axs[0,0].legend((p1[0], p2[0]), ('ST', 'S1'))
axs[0,0].set_xlabel("Parameters")
axs[0,0].set_ylabel("Sensitivity Index")
# 10
axs[0,1].set_title('b)Sensitivity indices S1(green) and \n ST(green+white) for  η= 10% ', y = -0.25, fontsize = 12)
p3=axs[0,1].bar(mylabels, Si10['ST'], color ='white', edgecolor = 'black')
p4=axs[0,1].bar(mylabels, Si10['S1'], color='#7eb54e', edgecolor = 'black')
ΣST = sum(Si10['ST'])
ΣS1 = sum(Si10['S1'])
axs[0,1].text(7, 0.25, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), style='italic', fontsize=12,
     bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs[0,1].set_xticklabels(mylabels, rotation = 45)
axs[0,1].legend((p3[0], p4[0]), ('ST', 'S1'))
axs[0,1].set_xlabel("Parameters")
axs[0,1].set_ylabel("Sensitivity Index")
# 15
axs[1,0].set_title('c)Sensitivity indices S1(green) and \n ST(green+white) for  η= 15% ', y = -0.25, fontsize = 12)
p5=axs[1,0].bar(mylabels, Si15['ST'], color ='white', edgecolor = 'black')
p6=axs[1,0].bar(mylabels, Si15['S1'], color='#7eb54e', edgecolor = 'black')
ΣST = sum(Si15['ST'])
ΣS1 = sum(Si15['S1'])
axs[1,0].text(7, 0.25, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), style='italic', fontsize=12,
     bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs[1,0].set_xticklabels(mylabels, rotation = 45)
axs[1,0].legend((p5[0], p6[0]), ('ST', 'S1'))
axs[1,0].set_xlabel("Parameters")
axs[1,0].set_ylabel("Sensitivity Index")
# 20
axs[1,1].set_title('d)Sensitivity indices S1(green) and \n ST(green+white) for  η= 20% ', y = -0.25, fontsize = 12)
p7=axs[1,1].bar(mylabels, Si20['ST'], color ='white', edgecolor = 'black')
p8=axs[1,1].bar(mylabels, Si20['S1'], color='#7eb54e', edgecolor = 'black')
ΣST = sum(Si20['ST'])
ΣS1 = sum(Si20['S1'])
axs[1,1].text(7, 0.25, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), style='italic', fontsize=12,
     bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs[1,1].set_xticklabels(mylabels, rotation = 45)
axs[1,1].legend((p7[0], p8[0]), ('ST', 'S1'))
axs[1,1].set_xlabel("Parameters")
axs[1,1].set_ylabel("Sensitivity Index")
# 25
axs[2,0].set_title('e)Sensitivity indices S1(green) and \n ST(green+white) for  η= 25% ', y = -0.25, fontsize = 12)
p9=axs[2,0].bar(mylabels, Si25['ST'], color ='white', edgecolor = 'black')
p10=axs[2,0].bar(mylabels, Si25['S1'], color='#7eb54e', edgecolor = 'black')
ΣST = sum(Si25['ST'])
ΣS1 = sum(Si25['S1'])
axs[2,0].text(7, 0.25, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), style='italic', fontsize=12,
     bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs[2,0].set_xticklabels(mylabels, rotation = 45)
axs[2,0].legend((p9[0], p10[0]), ('ST', 'S1'))
axs[2,0].set_xlabel("Parameters")
axs[2,0].set_ylabel("Sensitivity Index")

uncert = np.array([5,10,15,20,25])
axs[2,1].set_title('f) Parameters Ranking based on ST ', y = -0.20, fontsize = 12)
axs[2,1].step(uncert, dfst.loc[:,'Yxs'],color='g',marker='.',where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Yps'],color='c',marker= 'o', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Kxs (g/l)'],color= 'r',marker='^', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Umax (h-1)'],color= 'b',marker='v', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Xi (g/l)'],color= 'm',marker='1', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Si (g/l)'],color= 'y',marker='3', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Ai (g/l)'],color= 'k',marker='>', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Flow Velocity (m/s)'],color= 'y',marker='<', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Viscosity of injection fluid (Nsm-2)'],color= 'g',marker='v', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Initial IFT (mN/m)'],color= 'c',marker='*', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Swir'],color= 'r',marker='s', where='pre')
axs[2,1].step(uncert, dfst.loc[:,'Sori'],color= 'b',marker='P', where='pre')
axs[2,1].set_xticks(uncert)
axs[2,1].set_yticks( [1,2,3,4,5,6,7,8,9,10,11,12])
#plt.legend(labels =['Yxs','Yps','Kxs','Umax', 'Xi','Si','Ai','v','Uw','IFTmax','Swir','Sori'])
axs[2,1].set_ylim(13,0)
axs[2,1].set_xlabel('Input Uncertainty(%)')
axs[2,1].set_ylabel('Rank')
#pos = axs[2,1].get_position()
#axs[2,1].set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
axs[2,1].legend(labels =['Yxs','Yps','Kxs','Umax', 'Xi','Si','Ai','v','Uw','IFTmax','Swir','Sori'], loc = 0)
fig3.tight_layout()




#%%
plt.figure(9)
#plt.title('Input Uncertainty vs Output Uncertainty')
inun = [*[0], *inputuncert]
plt.plot(inun,[0,output_uncert(result5kinetic)
                         ,output_uncert(result10kinetic)
                         ,output_uncert(result15kinetic)
                         ,output_uncert(result20kinetic)
                         ,output_uncert(result25kinetic)],c='blue', marker = '.')
plt.plot(inun,[0,output_uncert(result5res)
                         ,output_uncert(result10res)
                         ,output_uncert(result15res)
                         ,output_uncert(result20res)
                         ,output_uncert(result25res)],c='green',marker='s')
plt.plot(inun,[0,output_uncert(result5op)
                         ,output_uncert(result10op)
                         ,output_uncert(result15op)
                         ,output_uncert(result20op)
                         ,output_uncert(result25op)],c='red',marker='^')
plt.legend(labels=['Microbial Kinetic Parameters','Reservoir Parameters','Operational Parameters'])
plt.xlabel('Input Uncertainty, %')
plt.ylabel('Output Uncertainty')
plt.xlim([0, 30])
plt.ylim([0, 80])
plt.show()



figure, axe = plt.subplots()
sns.distplot(data25all['recovery'], bins = 200, hist = False, color = 'violet', kde_kws={'linestyle':'-.'})
sns.distplot(datacase1['recovery'], bins = 200, hist = False, color = 'orange', kde_kws={'linestyle':'-'})
sns.distplot(datacase2['recovery'], bins = 200, hist = False, color = 'red', kde_kws={'linestyle':'-'})
sns.distplot(datacase3['recovery'], bins = 200, hist = False, color = 'blue', kde_kws={'linestyle':'-'})
plt.ylabel('Probability')
plt.xlabel('Oil Recovery, %')
plt.legend(labels=['25% Input Uncertainty','Case1','Case2','Case3'])
#axe.text(0, 0.5, "two functions", bbox=dict(facecolor='red', alpha=0.5))
plt.xlim([0,50])
plt.ylim([0,0.17])
plt.show()


plt.figure(12)

sns.distplot(data25all['recovery'], bins = 200, hist = False, color = 'violet', kde_kws={'linestyle':'-.'})
plt.ylabel('Probability')
plt.xlabel('Oil Recovery, %')
plt.legend(labels=['Yxs, Yps, Ai: η = 5%\n v, Uw, Sori: η = 5%; \n Others: η = 25%','All: η = 25%'])
plt.xlim([0,50])
plt.ylim([0,0.15])
plt.show()







'''
plt.figure(12)
x = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
y10 =[33.646, 33.378, 35.34, 33.74, 33.484, 34.188, 33.614, 33.99, 34, 33.6, 33.7, 33.8, 33.857, 33.85, 33.976, 33.81, 33.964, 34, 34.009, 33.973, 34.031]
y20 = [72.02626122, 67.34911589, 64.24847223, 62.11200848, 63.69619858, 62.61598608, 63.07462157, 63.19923747, 63.20431296, 63.85125768, 63.33546785, 63.22134977, 62.89083067, 63.14484995, 63.02483491, 63.00309393, 63.15928554, 63.1824847, 63.12894246, 63.07714995, 63.26627304]
y30 = [319.24, 91.269, 101.227, 101.23, 92.6, 98.7, 94.244, 95.508, 104.843, 95.635, 97.581, 94.396, 102.871, 96.707, 97.401, 98.478, 98.638, 97.056, 98.66, 97.96 ,100.563]
y40 = [121.249, 126.3, 171.854, 131.134, 136.145, 135.607, 140.09, 165.622, 166.005, 142.509, 183.171, 139.63, 151.311, 148.486, 153.36, 150.4, 148.7, 144.478, 151.603, 146.818, 149.784]

plt.plot(x, y10, c='blue', linestyle='--')
plt.plot(x, y20, c='green', linestyle='-.')
plt.plot(x, y30, c='red', linestyle=':')
plt.plot(x, y40, c='orange', linestyle='-')
plt.xlabel('Number of Realisations')
plt.ylabel('Output Uncertainty(%)')
plt.legend(labels=['η = 10','η = 20','η = 30','η = 40'])
plt.xlim([0, 100000])
plt.ylim([0,200])
plt.show()
'''




