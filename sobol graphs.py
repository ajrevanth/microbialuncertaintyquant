import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from SALib.analyze import sobol
from SALib.sample import saltelli
import seaborn as sns
from scipy import stats
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF
from sklearn.metrics import explained_variance_score
from numpy.random import seed
import random as python_random
python_random.seed(7)
np.random.seed(7)
seed(7)
import tensorflow
tensorflow.random.set_seed(7)

###
data = pd.read_excel(r'D:\Project\MC Simulation\MEOR\UALHSK.xlsx',sheet_name ='Sheet1') #change directory for excel file
y = data["% Oil recovery"]
features = [
    "Yxs",
    "Yps",
    "Kxs (g/l)",
    "Umax (h-1)",
    "Xi (g/l)",
    "Si (g/l)",
    "Ai (g/l)",
    "Resident Time (h)",
    "Flow Velocity (m/s)",
    "Viscosity of injection fluid (Nsm-2)",
    "Initial IFT (mN/m)",
    "Swir",
    "Sori",
]
X = data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=7,test_size = 0.3)

# Keras NN
def nn_model(learn_rate=0.01, momentum=0.9):
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(6, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal'))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    # Compile model
    
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
modelkerasnn = KerasRegressor(build_fn=nn_model, batch_size = 512,epochs = 1000 ,verbose=0)
pipe_kerasnn = make_pipeline(StandardScaler() ,modelkerasnn)
pipe_kerasnn.fit(train_X, train_y)

# pred_y_kerasnn = pipe_kerasnn.predict(val_X)
# mse = mean_squared_error(pred_y_kerasnn, val_y, squared=False)
# rsqe = r2_score(pred_y_kerasnn, val_y)
# ex_var = explained_variance_score(pred_y_kerasnn, val_y)
# print("r2 score= " + str(rsqe) + " rmse= "+ str(mse) + " Explained Variance = " + str(ex_var))


#%%
Si5 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\Sobols paper\5.xlsx")
Si10 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\Sobols paper\10.xlsx")
Si15 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\Sobols paper\15.xlsx")
Si20 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\Sobols paper\20.xlsx")
Si25 = pd.read_excel(r"D:\Project\MC Simulation\MEOR\Sobols paper\25.xlsx")
dfst = pd.read_excel(r"D:\Project\MC Simulation\MEOR\Sobols paper\Variable ranking.xlsx")
#%%

#FIGURE 3
#fig1, axs =  plt.subplots(3,2, figsize= (7.48,7.48), dpi = 1000)
fig1 = plt.figure(figsize=(15, 10), constrained_layout=True, dpi =1000)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.7)
gs = fig1.add_gridspec(3, 3)

axs0 = fig1.add_subplot(gs[0, 0])
axs1 = fig1.add_subplot(gs[0, 1])
axs2 = fig1.add_subplot(gs[1, 0])
axs3 = fig1.add_subplot(gs[1, 1])
axs4 = fig1.add_subplot(gs[2, 0])
mylabels = ['Yxs','Yps','Kxs','Umax', 'Xi','Si','Ai','v','Uw','IFTmax','Swir','Sori']

# 5
#axs[0,0].set_title('a)Sensitivity indices S1(green) and \n ST(green+white) for  η= 5% ', y = -0.25, fontsize = 12)
#plt.subplot2grid(shape = (5,2), loc = (0,0), colspan = 1, rowspan = 3)
axs0.set_title('(a)', y = 1, fontsize = 10)
p1=axs0.bar(mylabels, Si5['ST'], color ='white', edgecolor = 'black', yerr = 2*Si5['ST_conf'],capsize=3)
p2=axs0.bar(mylabels, Si5['S1'], color='#7eb54e', edgecolor = 'black', yerr = 2*Si5['S1_conf'],capsize=3)
ΣST = sum(Si5['ST'])
ΣS1 = sum(Si5['S1'])
axs0.text(4, 0.27, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), fontsize=10)
#    bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs0.set_xticklabels(mylabels, rotation = 45, fontsize = 9)
axs0.legend((p1[0], p2[0]), ('ST', 'S1'), fontsize = 10)
axs0.set_xlabel("Parameters", fontsize = 11)
axs0.set_ylabel("Sensitivity Index", fontsize = 11)

# 10
axs1.set_title('(b)', y = 1, fontsize = 10)
#axs[0,1].set_title('b)Sensitivity indices S1(green) and \n ST(green+white) for  η= 10% ', y = -0.25, fontsize = 12)
p3=axs1.bar(mylabels, Si10['ST'], color ='white', edgecolor = 'black', yerr = 2*Si10['ST_conf'],capsize=3)
p4=axs1.bar(mylabels, Si10['S1'], color='#7eb54e', edgecolor = 'black', yerr = 2*Si10['S1_conf'],capsize=3)
ΣST = sum(Si10['ST'])
ΣS1 = sum(Si10['S1'])
axs1.text(4, 0.27, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), fontsize=10)
#axs[0,1].text(7, 0.25, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), style='italic', fontsize=12,
#     bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs1.set_xticklabels(mylabels, rotation = 45, fontsize = 9)
axs1.legend((p3[0], p4[0]), ('ST', 'S1'), fontsize = 10)
axs1.set_xlabel("Parameters", fontsize = 11)
axs1.set_ylabel("Sensitivity Index", fontsize = 11)

# 15
axs2.set_title('(c)', y = 1, fontsize = 10)
#axs[1,0].set_title('c)Sensitivity indices S1(green) and \n ST(green+white) for  η= 15% ', y = -0.25, fontsize = 12)
p5=axs2.bar(mylabels, Si15['ST'], color ='white', edgecolor = 'black', yerr = 2*Si15['ST_conf'],capsize=3)
p6=axs2.bar(mylabels, Si15['S1'], color='#7eb54e', edgecolor = 'black', yerr = 2*Si15['S1_conf'],capsize=3)
ΣST = sum(Si15['ST'])
ΣS1 = sum(Si15['S1'])
axs2.text(4, 0.27, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), fontsize=10)
#axs[1,0].text(7, 0.25, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), style='italic', fontsize=12,
#     bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs2.set_xticklabels(mylabels, rotation = 45, fontsize = 9)
axs2.legend((p5[0], p6[0]), ('ST', 'S1'), fontsize = 10)
axs2.set_xlabel("Parameters", fontsize = 11)
axs2.set_ylabel("Sensitivity Index", fontsize = 11)

# 20
axs3.set_title('(d)', y = 1, fontsize = 10)
#axs[1,1].set_title('d)Sensitivity indices S1(green) and \n ST(green+white) for  η= 20% ', y = -0.25, fontsize = 12)
p7=axs3.bar(mylabels, Si20['ST'], color ='white', edgecolor = 'black', yerr = 2*Si20['ST_conf'],capsize=3)
p8=axs3.bar(mylabels, Si20['S1'], color='#7eb54e', edgecolor = 'black', yerr = 2*Si20['S1_conf'],capsize=3)
ΣST = sum(Si20['ST'])
ΣS1 = sum(Si20['S1'])
axs3.text(4, 0.27, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), fontsize=10)
#axs[1,1].text(7, 0.25, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), style='italic', fontsize=12,
#     bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs3.set_xticklabels(mylabels, rotation = 45, fontsize = 9)
axs3.legend((p7[0], p8[0]), ('ST', 'S1'), fontsize = 10)
axs3.set_xlabel("Parameters", fontsize = 11)
axs3.set_ylabel("Sensitivity Index", fontsize = 11)

# 25
axs4.set_title('(e)', y = 1, fontsize = 10)
#axs[2,0].set_title('e)Sensitivity indices S1(green) and \n ST(green+white) for  η= 25% ', y = -0.25, fontsize = 12)
p9=axs4.bar(mylabels, Si25['ST'], color ='white', edgecolor = 'black', yerr = 2*Si25['ST_conf'],capsize=3)
p10=axs4.bar(mylabels, Si25['S1'], color='#7eb54e', edgecolor = 'black', yerr = 2*Si25['S1_conf'],capsize=3)
ΣST = sum(Si25['ST'])
ΣS1 = sum(Si25['S1'])
axs4.text(4, 0.27, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), fontsize=10)
#axs[2,0].text(7, 0.25, ' ΣST=%0.2f\n ΣS1=%0.2f' % (ΣST,ΣS1), style='italic', fontsize=12,
#     bbox={'facecolor': 'white', 'alpha': 0, 'pad': 0})
axs4.set_xticklabels(mylabels, rotation = 45, fontsize = 9)
axs4.legend((p9[0], p10[0]), ('ST', 'S1'), fontsize = 10)
axs4.set_xlabel("Parameters", fontsize = 11)
axs4.set_ylabel("Sensitivity Index", fontsize = 11)
#fig1.tight_layout(pad = 0.5)
fig1.savefig('fig3_s.jpeg', dpi = 1000)
#%%

#figure 5

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from math import sqrt
from scipy.stats import pearsonr

#uncertainty = int (input("Type uncertainty(%):"))
#u = uncertainty
#param = input("what do u want to vary ?\n 1) All parameters \n 2) Microbial kinetic parameters \n 3) Reservoir parameters \n 4) Operational parameters")

u1 =5
u =25
real = 30000
    
yxs = 0.1843
#yxsdev = 0.0287 #0.0052 
yps = 0.078 #0.0988  
#ypsdev = 0.1743 #0.0168 
ks = 6.86 #14.17  
#ksdev = 0.1288 #1.3397
umax = 0.053
#umaxdev = 0.0248 #0.0048
Ka = 0.428
Yxa = 0.889

#reservoir param
somean = 0.4
swirmean = 0.2
iftmaxmean = 51.6

#operational param
ximean =  0.152167
simean = 19.234
aimean = 3
uwmean = 0.001
Tmean = 132
vmean = 0.0004

Yxs = [yxs]*real
Yps = [yps]*real
Kxs = [ks]*real
Umax = [umax]*real

v =[vmean]*real
uw = [uwmean]*real
T = [Tmean]*real
ai = [aimean]*real
si = [simean]*real
xi = [ximean]*real

so = [somean]*real
swir = [swirmean]*real
iftmax = [iftmaxmean]*real
'''
def PosNormal(mean, sigma, size):
    import numpy as np
    w = [mean]*size
    for i in range(size):
        while 1:
            w[i] = np.random.normal(loc = mean,scale = sigma)
            if w[i] > 0:
                break
            else:
                continue      
    return w
xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
sampling = LHS(xlimits=xlimits)
num = 50
x = sampling(num)
'''




l_bounds =[vmean-(u*sqrt(3)*vmean/100),uwmean-(u*sqrt(3)*uwmean/100), 
            aimean-(u*sqrt(3)*aimean/100), simean-(u*sqrt(3)*simean/100), ximean-(u*sqrt(3)*ximean/100),
            somean-(u*sqrt(3)*somean/100), swirmean-(u*sqrt(3)*swirmean/100), iftmaxmean-(u*sqrt(3)*iftmaxmean/100),
            yxs-(u*sqrt(3)*yxs/100), yps-(u*sqrt(3)*yps/100), ks-(u*sqrt(3)*ks/100), umax-(u*sqrt(3)*umax/100)]
u_bounds = [vmean+(u*sqrt(3)*vmean/100), uwmean+(u*sqrt(3)*uwmean/100), 
             aimean+(u*sqrt(3)*aimean/100), simean+(u*sqrt(3)*simean/100), ximean+(u*sqrt(3)*ximean/100),
             somean+(u*sqrt(3)*somean/100), swirmean+(u*sqrt(3)*swirmean/100), iftmaxmean+(u*sqrt(3)*iftmaxmean/100),
             yxs+(u*sqrt(3)*yxs/100), yps+(u*sqrt(3)*yps/100), ks+(u*sqrt(3)*ks/100), umax+(u*sqrt(3)*umax/100)]
sampler = qmc.LatinHypercube(d=12, optimization = "random-cd")
sample = sampler.random(n=real)
sample = qmc.scale(sample, l_bounds, u_bounds)
v, uw, ai, si, xi, so, swir, iftmax, Yxs, Yps, Kxs, Umax = [sample[:,i] for i in range(0,12)]
 
'''
'Yxs', 'Yps', 'Kxs (g/l)', 'Umax (h-1)','Xi (g/l)','Si (g/l)','Ai (g/l)',
'Resident Time (h)','Flow Velocity (m/s)','Viscosity of injection fluid (Nsm-2)',
'Initial IFT (mN/m)','Swir','Sori'
'''
data = {'Yxs':Yxs,'Yps':Yps,'Kxs (g/l)':Kxs,'Umax (h-1)':Umax, 'Xi (g/l)': xi,'Si (g/l)': si,
        'Ai (g/l)':ai, 'Resident Time (h)':T,'Flow Velocity (m/s)': v,
        'Viscosity of injection fluid (Nsm-2)':uw,'Initial IFT (mN/m)':iftmax,
        'Swir':swir,'Sori':so}
data1 = {'Yxs':Yxs,'Yps':Yps,'Kxs (g/l)':[ks]*real,'Umax (h-1)':Umax, 'Xi (g/l)': [ximean]*real,'Si (g/l)': si,
        'Ai (g/l)':ai, 'Resident Time (h)':T,'Flow Velocity (m/s)': v,
        'Viscosity of injection fluid (Nsm-2)':uw,'Initial IFT (mN/m)':[iftmaxmean]*real,
        'Swir':swir,'Sori':so}
data2 = {'Yxs':[yxs]*real,'Yps':[yps]*real,'Kxs (g/l)':Kxs,'Umax (h-1)':[umax]*real, 'Xi (g/l)': xi,'Si (g/l)': [simean]*real,
        'Ai (g/l)':[aimean]*real, 'Resident Time (h)':T,'Flow Velocity (m/s)': [vmean]*real,
        'Viscosity of injection fluid (Nsm-2)':[uwmean]*real,'Initial IFT (mN/m)':iftmax,
        'Swir':[swirmean]*real,'Sori':[somean]*real}
df1 = pd.DataFrame(data)
df2 = pd.DataFrame(data1)
df3 = pd.DataFrame(data2)
Y = pipe_kerasnn.predict(df1)
Y1 = pipe_kerasnn.predict(df2)
Y2 = pipe_kerasnn.predict(df3)
corr,_ = pearsonr(Y, Y1)
corr1,_ = pearsonr(Y, Y2)
#df1['recovery'] = Y
#df1.to_excel("NNdata.xlsx",) 

fig1 = plt.figure(figsize=(11, 11), constrained_layout=True, dpi =1000)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.6)
gs = fig1.add_gridspec(3, 3)

axs0 = fig1.add_subplot(gs[0, 0])
axs1 = fig1.add_subplot(gs[0, 1])
axs2 = fig1.add_subplot(gs[1, 0])
axs3 = fig1.add_subplot(gs[1, 1])

uncert = np.array([5,10,15,20,25])
#axs[2,1].set_title('f) Parameters Ranking based on ST ', y = -0.20, fontsize = 12)
axs0.step(uncert, dfst.loc[:,'Yxs'],color='g',marker='.',where='pre')
axs0.step(uncert, dfst.loc[:,'Yps'],color='c',marker= 'o', where='pre')
axs0.step(uncert, dfst.loc[:,'Kxs (g/l)'],color= 'r',marker='^', where='pre')
axs0.step(uncert, dfst.loc[:,'Umax (h-1)'],color= 'b',marker='v', where='pre')
axs0.step(uncert, dfst.loc[:,'Xi (g/l)'],color= 'm',marker='1', where='pre')
axs0.step(uncert, dfst.loc[:,'Si (g/l)'],color= 'y',marker='3', where='pre')
axs0.step(uncert, dfst.loc[:,'Ai (g/l)'],color= 'k',marker='>', where='pre')
axs0.step(uncert, dfst.loc[:,'Flow Velocity (m/s)'],color= 'y',marker='<', where='pre')
axs0.step(uncert, dfst.loc[:,'Viscosity of injection fluid (Nsm-2)'],color= 'g',marker='v', where='pre')
axs0.step(uncert, dfst.loc[:,'Initial IFT (mN/m)'],color= 'c',marker='*', where='pre')
axs0.step(uncert, dfst.loc[:,'Swir'],color= 'r',marker='s', where='pre')
axs0.step(uncert, dfst.loc[:,'Sori'],color= 'b',marker='P', where='pre')
axs0.set_xticks(uncert)
axs0.set_yticks( [1,2,3,4,5,6,7,8,9,10,11,12])
#plt.legend(labels =['Yxs','Yps','Kxs','Umax', 'Xi','Si','Ai','v','Uw','IFTmax','Swir','Sori'])
axs0.set_ylim(13,0)
axs0.set_xlabel('Input Uncertainty(%)')
axs0.set_ylabel('Rank')
pos = axs0.get_position()
axs0.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
axs0.legend(labels =['Yxs','Yps','Kxs','Umax', 'Xi','Si','Ai','v','Uw','IFTmax','Swir','Sori'],fontsize = 9, loc = 1)
axs0.set_title('(a)')

axs2.set_title('(b)')
axs2.scatter(Y, Y1, marker = '^')
j = np.linspace(0,60,20000)
axs2.plot(j,j, color = 'red')
axs2.set_xlabel('Set 1')
axs2.set_ylabel('Set 2')
axs2.set_xlim([0,60])
axs2.set_ylim([0,60])
axs2.text(1,55,'Pearson Correlation Coeffcient = '+ corr, style='italic', fontsize=9,
    bbox={'facecolor': 'white', 'alpha': 0, 'pad': 8})
#plt.legend(labels =['Data points','Y=X'])


axs3.set_title('(c)')
axs3.scatter(Y, Y2,color = 'g', marker = 'o')
j = np.linspace(0,60,20000)
axs3.plot(j,j, color = 'red')
axs3.set_xlabel('Set 1')
axs3.set_ylabel('Set 3')
axs3.set_xlim([0,60])
axs3.set_ylim([0,60])
axs3.text(1,55,'Pearson Correlation Coeffcient = '+ corr1, style='italic', fontsize=9,
    bbox={'facecolor': 'white', 'alpha': 0, 'pad': 8})
#plt.legend(labels =['Data points','Y=X'])

axs1.axis('off')
values = [ ['Rank','Parameters'],
    ['1','Yps'],
          ['2','Yxs'],
          ['3','Ai'],
          ['4','Sori'],
          ['5','Uw'],
          ['6','v'],
          ['7','Si'],
          ['8','Umax'],
          ['9','Swir'],
          ['10','Xi'],
          ['11','IFTmax'],
          ['12','Kxs']]

table = axs1.table(cellText=values,
                  rowLoc='center',
                  colLoc='center',
                  cellLoc = 'center',
                  #colWidths = [0.4, 0.5, 0.6, 0.5],
                  loc='center')
#cell = table._cells
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.15,1)
table.auto_set_column_width(col=list(range(4)))
#mergecells(table,(0,0),(1,0))
#mergecells(table,(0,1),(0,2))
#mergecellsN(table,[(0,1),(0,2),(0,3)])

fig1.tight_layout()
fig1.savefig('fig1test.jpeg', dpi = 1000)
#%%

#S2 graphs

#from randomnumbergenerater import PosNormal
random_state = 0
yxs = 0.1843
#yxsdev = 0.0287 #0.0052 
yps = 0.078 #0.0988  
#ypsdev = 0.1743 #0.0168 
ks = 6.86 #14.17  
#ksdev = 0.1288 #1.3397
umax = 0.053
#umaxdev = 0.0248 #0.0048
Ka = 0.428
Yxa = 0.889

#reservoir param
somean = 0.4
swirmean = 0.2
iftmaxmean = 51.6

#operational param
ximean =  0.152167
simean = 19.234
aimean = 3
uwmean = 0.001
Tmean = 132
vmean = 0.0004
u = 25

problem={ 
        'num_vars': 12,
        'names':['Yxs','Yps','Kxs (g/l)','Umax (h-1)', 'Xi (g/l)','Si (g/l)','Ai (g/l)',
             'Flow Velocity (m/s)','Viscosity of injection fluid (Nsm-2)',
             'Initial IFT (mN/m)','Swir','Sori'],
        'bounds': [[ yxs-(u*sqrt(3)*yxs/100),yxs+(u*sqrt(3)*yxs/100)], [yps-(u*sqrt(3)*yps/100),yps+(u*sqrt(3)*yps/100)], 
               [ks-(u*sqrt(3)*ks/100), ks+(u*sqrt(3)*ks/100)], [umax-(u*sqrt(3)*umax/100),umax+(u*sqrt(3)*umax/100)],
               [ximean-(u*sqrt(3)*ximean/100),ximean+(u*sqrt(3)*ximean/100)], [simean-(u*sqrt(3)*simean/100), simean+(u*sqrt(3)*simean/100)],
               [aimean-(u*sqrt(3)*aimean/100),aimean+(u*sqrt(3)*aimean/100)],
               [vmean-(u*sqrt(3)*vmean/100),vmean+(u*sqrt(3)*vmean/100)], [uwmean-(u*sqrt(3)*uwmean/100), uwmean+(u*sqrt(3)*uwmean/100)],
               [iftmaxmean-(u*sqrt(3)*iftmaxmean/100),iftmaxmean+(u*sqrt(3)*iftmaxmean/100)], [swirmean-(u*sqrt(3)*swirmean/100),swirmean+(u*sqrt(3)*swirmean/100)],
               [somean-(u*sqrt(3)*somean/100), somean+(u*sqrt(3)*somean/100)]],
        'dists': ['unif','unif','unif','unif','unif','unif','unif','unif','unif','unif','unif',
              'unif']
        }


input_data = saltelli.sample(problem, 8196, calc_second_order=True)

    # Finding Output
Y = np.zeros([input_data.shape[0]])

T = np.full(len(input_data[:,]),Tmean)
data1 = {'Yxs':input_data[:,0],'Yps':input_data[:,1],'Kxs (g/l)':input_data[:,2],'Umax (h-1)':input_data[:,3], 'Xi (g/l)': input_data[:,4],'Si (g/l)': input_data[:,5],
    'Ai (g/l)':input_data[:,6], 'Resident Time (h)':T,'Flow Velocity (m/s)': input_data[:,7],
    'Viscosity of injection fluid (Nsm-2)':input_data[:,8],'Initial IFT (mN/m)':input_data[:,9],
    'Swir':input_data[:,10],'Sori':input_data[:,11]}
df = pd.DataFrame(data1)
Y = pipe_kerasnn.predict(df) 

    
Si = sobol.analyze(problem, Y, calc_second_order = True)

def S2_to_dict(matrix, problem):
    result = {}
    names = list(problem["names"])
    
    for i in range(problem["num_vars"]):
        for j in range(i+1, problem["num_vars"]):
            if names[i] not in result:
                result[names[i]] = {}
            if names[j] not in result:
                result[names[j]] = {}
                
            result[names[i]][names[j]] = result[names[j]][names[i]] = float(matrix[i][j])
            
    return result

result = {} #create dictionary to store new
result['S1']={k : float(v) for k, v in zip(problem["names"], Si["S1"])}
result['S1_conf']={k : float(v) for k, v in zip(problem["names"], Si["S1_conf"])}
result['S2'] = S2_to_dict(Si['S2'], problem)
result['S2_conf'] = S2_to_dict(Si['S2_conf'], problem)
result['ST']={k : float(v) for k, v in zip(problem["names"], Si["ST"])}
result['ST_conf']={k : float(v) for k, v in zip(problem["names"], Si["ST_conf"])}

import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt

# Load Sensitivity Analysis results as dictionary
SAresults = result
#SAresults = np.load('SAresults.npy').item()
# Get list of parameters
parameters = list(SAresults['S1'].keys())
# Set min index value, for the effects to be considered significant
index_significance_value = 0.01

'''
Define some general layout settings.
'''
node_size_min = 15 # Max and min node size
node_size_max = 30
border_size_min = 1 # Max and min node border thickness
border_size_max = 8
edge_width_min = 1 # Max and min edge thickness
edge_width_max = 10
edge_distance_min = 0.1 # Max and min distance of the edge from the center of the circle
edge_distance_max = 0.6 # Only applicable to the curved edges

'''
Set up some variables and functions that will facilitate drawing circles and 
moving items around.
'''
# Define circle center and radius
center = [0.0,0.0] 
radius = 1.0
# Create an array with all angles in a circle (i.e. from 0 to 2pi)
step = 0.001
radi = np.arange(0,2*np.pi,step) 

# Function to get distance between two points
def distance(p1,p2):
    return np.sqrt(((p1-p2)**2).sum())

# Function to get middle point between two points
def middle(p1,p2):
    return (p1+p2)/2

# Function to get the vertex of a curve between two points
def vertex(p1,p2,c):
    m = middle(p1,p2)
    curve_direction = c-m
    return m+curve_direction*(edge_distance_min+edge_distance_max*(1-distance(m,c)/distance(c,p1)))

# Function to get the angle of the node from the center of the circle
def angle(p,c):
    # Get x and y distance of point from center
    [dx,dy] = p-c 
    if dx == 0: # If point on vertical axis (same x as center)
        if dy>0: # If point is on positive vertical axis
            return np.pi/2.
        else: # If point is on negative vertical axis
            return np.pi*3./2.
    elif dx>0: # If point in the right quadrants
        if dy>=0: # If point in the top right quadrant
            return np.arctan(dy/dx)
        else: # If point in the bottom right quadrant
            return 2*np.pi+np.arctan(dy/dx)
    elif dx<0: # If point in the left quadrants
        return np.pi+np.arctan(dy/dx)

'''
First, set up graph with all parameters as nodes and draw all second order (S2)
indices as edges in the network. For every S2 index, we need a Source parameter,
a Target parameter, and the Weight of the line, given by the S2 index itself. 
'''
combs = [list(c) for c in list(itertools.combinations(parameters, 2))]

Sources = list(list(zip(*combs))[0])
Targets = list(list(zip(*combs))[1])
# Sometimes computing errors produce negative Sobol indices. The following reads
# in all the indices and also ensures they are between 0 and 1.
Weights = [max(min(x, 1), 0) for x in [SAresults['S2'][Sources[i]][Targets[i]] for i in range(len(Sources))]]
Weights = [0 if x<index_significance_value else x for x in Weights]

# Set up graph
G = nx.Graph()
# Draw edges with appropriate weight
for s,t,weight in zip(Sources, Targets, Weights):
    G.add_edges_from([(s,t)], w=weight)

# Generate dictionary of node postions in a circular layout
Pos = nx.circular_layout(G)

'''
Normalize node size according to first order (S1) index. First, read in S1 indices,
ensure they're between 0 and 1 and normalize them within the max and min range
of node sizes.
Then, normalize edge thickness according to S2. 
'''
# Node size
first_order = [max(min(x, 1), 0) for x in [SAresults['S1'][key] for key in SAresults['S1']]]
first_order = [0 if x<index_significance_value else x for x in first_order]
node_size = [node_size_min*(1 + (node_size_max-node_size_min)*k/max(first_order)) for k in first_order]
# Node border thickness
total_order = [max(min(x, 1), 0) for x in [SAresults['ST'][key] for key in SAresults['ST']]]
total_order = [0 if x<index_significance_value else x for x in total_order]
border_size = [border_size_min*(1 + (border_size_max-border_size_min)*k/max(total_order)) for k in total_order]
# Edge thickness
edge_width = [edge_width_min*((edge_width_max-edge_width_min)*k/max(Weights)) for k in Weights]

'''
Draw network. This will draw the graph with straight lines along the edges and 
across the circle. 
'''    
nx.draw_networkx_nodes(G, Pos, node_size=node_size, node_color='#98B5E2', 
                       edgecolors='#1A3F7A', linewidths = border_size)
nx.draw_networkx_edges(G, Pos, width=edge_width, edge_color='#2E5591', alpha=0.7)
names = nx.draw_networkx_labels(G, Pos, font_size=12, font_color='#0B2D61', font_family='sans-serif')
for node, text in names.items():
    position = (radius*1.1*np.cos(angle(Pos[node],center)), radius*1.1*np.sin(angle(Pos[node],center)))
    text.set_position(position)
    text.set_clip_on(False)
plt.gcf().set_size_inches(9,9) # Make figure a square
plt.axis('off')

'''
 We can now draw the network with curved lines along the edges and across the circle.
 Calculate all distances between 1 node and all the others (all distances are 
 the same since they're in a circle). We'll need this to identify the curves 
 we'll be drawing along the perimeter (i.e. those that are next to each other).
 '''
min_distance = round(min([distance(Pos[list(G.nodes())[0]],Pos[n]) for n in list(G.nodes())[1:]]), 1)

# Figure to generate the curved edges between two points
def xy_edge(p1,p2): # Point 1, Point 2
    m = middle(p1,p2) # Get middle point between the two nodes
    # If the middle of the two points falls very close to the center, then the 
    # line between the two points is simply straight
    if distance(m,center)<1e-6:
        xpr = np.linspace(p1[0],p2[0],10)
        ypr = np.linspace(p1[1],p2[1],10)
    # If the distance between the two points is the minimum (i.e. they are next
    # to each other), draw the edge along the perimeter     
    elif distance(p1,p2)<=min_distance:
        # Get angles of two points
        p1_angle = angle(p1,center)
        p2_angle = angle(p2,center)
        # Check if the points are more than a hemisphere apart
        if max(p1_angle,p2_angle)-min(p1_angle,p2_angle) > np.pi:
            radi = np.linspace(max(p1_angle,p2_angle)-2*np.pi,min(p1_angle,p2_angle))
        else :
            radi = np.linspace(min(p1_angle,p2_angle),max(p1_angle,p2_angle))
        xpr = radius*np.cos(radi)+center[0]
        ypr = radius*np.sin(radi)+center[1]  
    # Otherwise, draw curve (parabola)
    else: 
        edge_vertex = vertex(p1,p2,center)
        a = distance(edge_vertex, m)/((distance(p1,p2)/2)**2)
        yp = np.linspace(-distance(p1,p2)/2, distance(p1,p2)/2,100)
        xp = a*(yp**2)
        xp += distance(center,edge_vertex)
        theta_m = angle(middle(p1,p2),center)
        xpr = np.cos(theta_m)*xp - np.sin(theta_m)*yp
        ypr = np.sin(theta_m)*xp + np.cos(theta_m)*yp
        xpr += center[0]
        ypr += center[1]
    return xpr,ypr

'''
Draw network. This will draw the graph with curved lines along the edges and 
across the circle. 
'''
plt.figure(figsize=(9,9))
for i, e in enumerate(G.edges()):
    x,y=xy_edge(Pos[e[0]],Pos[e[1]])
    plt.plot(x,y,'-',c='#2E5591',lw=edge_width[i],alpha=0.7)
for i, n in enumerate(G.nodes()):
    plt.plot(Pos[n][0],Pos[n][1], 'o', c='#98B5E2', markersize=node_size[i]/5, markeredgecolor = '#1A3F7A', markeredgewidth = border_size[i]*1.15)

for i, text in enumerate(G.nodes()):
    if node_size[i]<100:
        position = (radius*1.05*np.cos(angle(Pos[text],center)), radius*1.05*np.sin(angle(Pos[text],center)))
    else:
        position = (radius*1.01*np.cos(angle(Pos[text],center)), radius*1.01*np.sin(angle(Pos[text],center)))
    plt.annotate(text, position, fontsize = 12, color='#0B2D61', family='sans-serif')          
plt.axis('off')
plt.tight_layout()
plt.show()


def drawgraphs(SAresults):
    # Get list of parameters
    parameters = list(SAresults['S1'].keys())
    # Set min index value, for the effects to be considered significant
    index_significance_value = 0.01

    '''
    Define some general layout settings.
    '''
    node_size_min = 15 # Max and min node size
    node_size_max = 30
    border_size_min = 1 # Max and min node border thickness
    border_size_max = 8
    edge_width_min = 1 # Max and min edge thickness
    edge_width_max = 10
    edge_distance_min = 0.1 # Max and min distance of the edge from the center of the circle
    edge_distance_max = 0.6 # Only applicable to the curved edges

    '''
    Set up some variables and functions that will facilitate drawing circles and 
    moving items around.
    '''
    # Define circle center and radius
    center = [0.0,0.0] 
    radius = 1.0

    # Function to get distance between two points
    def distance(p1,p2):
        return np.sqrt(((p1-p2)**2).sum())

    # Function to get middle point between two points
    def middle(p1,p2):
        return (p1+p2)/2

    # Function to get the vertex of a curve between two points
    def vertex(p1,p2,c):
        m = middle(p1,p2)
        curve_direction = c-m
        return m+curve_direction*(edge_distance_min+edge_distance_max*(1-distance(m,c)/distance(c,p1)))

    # Function to get the angle of the node from the center of the circle
    def angle(p,c):
        # Get x and y distance of point from center
        [dx,dy] = p-c 
        if dx == 0: # If point on vertical axis (same x as center)
            if dy>0: # If point is on positive vertical axis
                return np.pi/2.
            else: # If point is on negative vertical axis
                return np.pi*3./2.
        elif dx>0: # If point in the right quadrants
            if dy>=0: # If point in the top right quadrant
                return np.arctan(dy/dx)
            else: # If point in the bottom right quadrant
                return 2*np.pi+np.arctan(dy/dx)
        elif dx<0: # If point in the left quadrants
            return np.pi+np.arctan(dy/dx)

    '''
    First, set up graph with all parameters as nodes and draw all second order (S2)
    indices as edges in the network. For every S2 index, we need a Source parameter,
    a Target parameter, and the Weight of the line, given by the S2 index itself. 
    '''
    combs = [list(c) for c in list(itertools.combinations(parameters, 2))]

    Sources = list(list(zip(*combs))[0])
    Targets = list(list(zip(*combs))[1])
    # Sometimes computing errors produce negative Sobol indices. The following reads
    # in all the indices and also ensures they are between 0 and 1.
    Weights = [max(min(x, 1), 0) for x in [SAresults['S2'][Sources[i]][Targets[i]] for i in range(len(Sources))]]
    Weights = [0 if x<index_significance_value else x for x in Weights]

    # Set up graph
    G = nx.Graph()
    # Draw edges with appropriate weight
    for s,t,weight in zip(Sources, Targets, Weights):
        G.add_edges_from([(s,t)], w=weight)

    # Generate dictionary of node postions in a circular layout
    Pos = nx.circular_layout(G)

    '''
    Normalize node size according to first order (S1) index. First, read in S1 indices,
    ensure they're between 0 and 1 and normalize them within the max and min range
    of node sizes.
    Then, normalize edge thickness according to S2. 
    '''
    # Node size
    first_order = [max(min(x, 1), 0) for x in [SAresults['S1'][key] for key in SAresults['S1']]]
    first_order = [0 if x<index_significance_value else x for x in first_order]
    node_size = [node_size_min*(1 + (node_size_max-node_size_min)*k/max(first_order)) for k in first_order]
    # Node border thickness
    total_order = [max(min(x, 1), 0) for x in [SAresults['ST'][key] for key in SAresults['ST']]]
    total_order = [0 if x<index_significance_value else x for x in total_order]
    border_size = [border_size_min*(1 + (border_size_max-border_size_min)*k/max(total_order)) for k in total_order]
    # Edge thickness
    edge_width = [edge_width_min*((edge_width_max-edge_width_min)*k/max(Weights)) for k in Weights]

    '''
    Draw network. This will draw the graph with straight lines along the edges and 
    across the circle. 
    '''    
    nx.draw_networkx_nodes(G, Pos, node_size=node_size, node_color='#98B5E2', 
                           edgecolors='#1A3F7A', linewidths = border_size)
    nx.draw_networkx_edges(G, Pos, width=edge_width, edge_color='#2E5591', alpha=0.7)
    names = nx.draw_networkx_labels(G, Pos, font_size=12, font_color='#0B2D61', font_family='sans-serif')
    for node, text in names.items():
        position = (radius*1.3*np.cos(angle(Pos[node],center)), radius*1.3*np.sin(angle(Pos[node],center)))
        text.set_position(position)
        text.set_clip_on(False)
    plt.gcf().set_size_inches(9,9) # Make figure a square
    plt.axis('off')

    '''
     We can now draw the network with curved lines along the edges and across the circle.
     Calculate all distances between 1 node and all the others (all distances are 
     the same since they're in a circle). We'll need this to identify the curves 
     we'll be drawing along the perimeter (i.e. those that are next to each other).
     '''
    min_distance = round(min([distance(Pos[list(G.nodes())[0]],Pos[n]) for n in list(G.nodes())[1:]]), 1)

    # Figure to generate the curved edges between two points
    def xy_edge(p1,p2): # Point 1, Point 2
        m = middle(p1,p2) # Get middle point between the two nodes
        # If the middle of the two points falls very close to the center, then the 
        # line between the two points is simply straight
        if distance(m,center)<1e-6:
            xpr = np.linspace(p1[0],p2[0],10)
            ypr = np.linspace(p1[1],p2[1],10)
        # If the distance between the two points is the minimum (i.e. they are next
        # to each other), draw the edge along the perimeter     
        elif distance(p1,p2)<=min_distance:
            # Get angles of two points
            p1_angle = angle(p1,center)
            p2_angle = angle(p2,center)
            # Check if the points are more than a hemisphere apart
            if max(p1_angle,p2_angle)-min(p1_angle,p2_angle) > np.pi:
                radi = np.linspace(max(p1_angle,p2_angle)-2*np.pi,min(p1_angle,p2_angle))
            else:
                radi = np.linspace(min(p1_angle,p2_angle),max(p1_angle,p2_angle))
            xpr = radius*np.cos(radi)+center[0]
            ypr = radius*np.sin(radi)+center[1]  
        # Otherwise, draw curve (parabola)
        else: 
            edge_vertex = vertex(p1,p2,center)
            a = distance(edge_vertex, m)/((distance(p1,p2)/2)**2)
            yp = np.linspace(-distance(p1,p2)/2, distance(p1,p2)/2,100)
            xp = a*(yp**2)
            xp += distance(center,edge_vertex)
            theta_m = angle(middle(p1,p2),center)
            xpr = np.cos(theta_m)*xp - np.sin(theta_m)*yp
            ypr = np.sin(theta_m)*xp + np.cos(theta_m)*yp
            xpr += center[0]
            ypr += center[1]
        return xpr,ypr

    '''
    Draw network. This will draw the graph with curved lines along the edges and 
    across the circle. 
    '''
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(1,1,1)
    for i, e in enumerate(G.edges()):
        x,y=xy_edge(Pos[e[0]],Pos[e[1]])
        ax.plot(x,y,'-',c='#2E5591',lw=edge_width[i],alpha=0.7)
    for i, n in enumerate(G.nodes()):
        ax.plot(Pos[n][0],Pos[n][1], 'o', c='#98B5E2', markersize=node_size[i]/5, markeredgecolor = '#1A3F7A', markeredgewidth = border_size[i]*1.15)

    for i, text in enumerate(G.nodes()):
        if node_size[i]<100:
            position = (radius*1.05*np.cos(angle(Pos[text],center)), radius*1.05*np.sin(angle(Pos[text],center)))
        else:
            position = (radius*1.01*np.cos(angle(Pos[text],center)), radius*1.01*np.sin(angle(Pos[text],center)))
        plt.annotate(text, position, fontsize = 12, color='#0B2D61', family='sans-serif')          
    ax.axis('off')
    fig.tight_layout()
    plt.show() 


#%%


#Figure 2
data5all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\5, all\NNdata5all.xlsx')
data10all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\10, all\NNdata10all.xlsx')
data15all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\15, all\NNdata15all.xlsx')
data20all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\20, all\NNdata20all.xlsx')
data25all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\25, all\NNdata25all.xlsx')
def major_formatter(x, pos):
    return int(x*100)
#%%
fig1 = plt.figure(figsize=(11,5), constrained_layout=True, dpi =1000)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.6)
gs = fig1.add_gridspec(1,2)

axs = fig1.add_subplot(gs[0, 0])
axs2 = fig1.add_subplot(gs[0, 1])

sns.distplot(data5all['recovery'], bins = 100, hist = True, color = 'blue', kde_kws={'marker':'*','markevery': 10,'markersize':7}, ax = axs)
sns.distplot(data10all['recovery'], bins = 100, hist = True, color = 'red', kde_kws={'linestyle':'--'}, ax = axs)
sns.distplot(data15all['recovery'], bins = 100, hist = True, color = 'green', kde_kws={'marker':'^','markevery': 10,'markersize':5}, ax = axs)
sns.distplot(data20all['recovery'], bins = 100, hist = True, color = 'orange', kde_kws={'linestyle':'-'}, ax = axs)
sns.distplot(data25all['recovery'], bins = 100, hist = True, color = 'violet', kde_kws={'marker':'.','markevery': 10,'markersize':7}, ax = axs)
#axs.set_title('(a)', y = 0.35, fontsize = 12)
axs.set_xlabel('Oil recovery, %', fontsize = 12)
axs.set_ylabel('Probability, %', fontsize = 12)
axs.set_ylim([0, 0.25])
axs.set_xlim([0, 55])
#axs[0,0].legend(labels=['η = 5%', 'η = 10%', 'η = 15%', 'η = 20%', 'η = 25%'])
axs.set_xlim([0,50])
#axs.set_ylim([0,0.25])
plt.xticks(fontsize = 9, rotation = 0)
plt.yticks(fontsize = 9, rotation = 0)
axs.yaxis.set_major_formatter(major_formatter)
# axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals = 0))
fig1.legend(labels=['η = 5%', 'η = 10%', 'η = 15%', 'η = 20%', 'η = 25%'], bbox_to_anchor = (0.85,1.05), ncol = 5, fontsize= 10)    

#fig.savefig('fig2.jpeg', dpi = 1000,bbox_inches='tight')


#axs2.set_title('b) 1-Cummulative Distribution Function of Oil \nRecovery, % (varying all parameters)', y = -0.35, fontsize = 12)
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
axs2.set_xlabel('Oil Recovery, %', fontsize = 12)
axs2.set_ylabel('1-Cummulative Probability Density', fontsize = 12) 
axs2.plot(base[:-1], 1-(cumulative/len(data5all['recovery'])), c='blue', marker ='*', markevery = 10)
axs2.plot(base1[:-1],1-(cumulative1/len(data10all['recovery'])), c='red', linestyle='--')
axs2.plot(base2[:-1],1-(cumulative2/len(data15all['recovery'])), c='green', marker = '^', markevery = 10)
axs2.plot(base3[:-1],1-(cumulative3/len(data20all['recovery'])), c='orange', linestyle='-')
axs2.plot(base4[:-1],1-(cumulative4/len(data25all['recovery'])), c='violet', marker = '.', markevery = 10)
#axs2.legend(labels=['η = 5%','η = 10%','η = 15%','η = 20%','η = 25%'])
axs2.set_xlim([0, 50])
axs2.set_ylim([0, 1])
fig1.tight_layout()
#%%

#Figure 6
def output_uncert(x):
    y = math.sqrt(x[3])*100/x[2]
    return y
def major_formatter(x, pos):
    return int(x*100)
datacase1 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase\NNdata.xlsx')
datacase2 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase2\NNdata.xlsx')
datacase3 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase3\NNdata.xlsx')
datacase4 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase4\NNdata.xlsx')
data25all = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\25, all\NNdata25all.xlsx')

resultcase1 = stats.describe(datacase1['recovery'], ddof=1, bias=False)
resultcase2 = stats.describe(datacase2['recovery'], ddof=1, bias=False)
resultcase3 = stats.describe(datacase3['recovery'], ddof=1, bias=False)
resultcase4 = stats.describe(datacase4['recovery'], ddof=1, bias=False)
result25all = stats.describe(data25all['recovery'], ddof=1, bias=False)

quantilecase1 = np.quantile(datacase1['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase2 = np.quantile(datacase2['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase3 = np.quantile(datacase3['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase4 = np.quantile(datacase4['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])


#%%
fig1 = plt.figure(figsize=(11,5), constrained_layout=True, dpi =800)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.6)
gs = fig1.add_gridspec(1,2)

axs = fig1.add_subplot(gs[0, 0])
axs3 = fig1.add_subplot(gs[0, 1])
#axs3 = fig1.add_subplot(gs[1,0])

axs.set_title('(a)',fontsize = 11)
sns.distplot(data25all['recovery'], bins = 100, hist = False, color = 'blue', kde_kws={'marker':'*','markevery': 10,'markersize':7}, ax = axs)
sns.distplot(datacase1['recovery'], bins = 100, hist = False, color = 'red', kde_kws={'linestyle':'--'}, ax = axs)
sns.distplot(datacase2['recovery'], bins = 100, hist = False, color = 'green', kde_kws={'marker':'^','markevery': 10,'markersize':5}, ax = axs)
sns.distplot(datacase3['recovery'], bins = 100, hist = False, color = 'orange', kde_kws={'linestyle':'-'}, ax = axs)
sns.distplot(datacase4['recovery'], bins = 100, hist = False, color = 'violet', kde_kws={'marker':'.','markevery': 10,'markersize':7}, ax = axs)
#axs.set_title('(a)', y = 0.35, fontsize = 12)
axs.set_xlabel('Oil recovery, %', fontsize = 12)
axs.set_ylabel('Probability, %', fontsize = 12)
axs.set_ylim([0, 0.25])
axs.set_xlim([0, 55])
axs.legend(labels=['η = 25%', 'Case1 ', 'Case 2', 'Case 3', 'Case 4'])
#axs.set_xlim([0,50])
#axs.set_ylim([0,0.25])
#axs.set_xticks(fontsize = 9, rotation = 0)
#axs.set_yticks(fontsize = 9, rotation = 0)
axs.yaxis.set_major_formatter(major_formatter)
# axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals = 0))
#fig1.legend(labels=['η = 5%', 'η = 10%', 'η = 15%', 'η = 20%', 'η = 25%'], bbox_to_anchor = (0.85,1.05), ncol = 5, fontsize= 10)    

#fig.savefig('fig2.jpeg', dpi = 1000,bbox_inches='tight')


# axs2.set_title('b',fontsize =11)
# #1-Cummulative Distribution Function of Oil \nRecovery, % (varying all parameters)', y = -0.35, fontsize = 12)
# values,base = np.histogram(data25all['recovery'],bins=200)
# values1,base1 = np.histogram(datacase1['recovery'],bins=200)
# values2,base2 = np.histogram(datacase2['recovery'],bins=200)
# values3,base3 = np.histogram(datacase3['recovery'],bins=200)
# values4,base4 = np.histogram(datacase4['recovery'],bins=200)
# cumulative = np.cumsum(values)
# cumulative1 = np.cumsum(values1)
# cumulative2 = np.cumsum(values2)
# cumulative3 = np.cumsum(values3)
# cumulative4 = np.cumsum(values4)
# axs2.set_xlabel('Oil Recovery, %', fontsize = 12)
# axs2.set_ylabel('1-Cummulative Probability Density', fontsize = 12) 
# axs2.plot(base[:-1], 1-(cumulative/len(data25all['recovery'])), c='blue', marker ='*', markevery = 10)
# axs2.plot(base1[:-1],1-(cumulative1/len(datacase1['recovery'])), c='red', linestyle='--')
# axs2.plot(base2[:-1],1-(cumulative2/len(datacase2['recovery'])), c='green', marker = '^', markevery = 10)
# axs2.plot(base3[:-1],1-(cumulative3/len(datacase3['recovery'])), c='orange', linestyle='-')
# axs2.plot(base4[:-1],1-(cumulative4/len(datacase4['recovery'])), c='violet', marker = '.', markevery = 10)
# #axs2.legend(labels=['η = 5%','η = 10%','η = 15%','η = 20%','η = 25%'])
# axs2.set_xlim([0, 50])
# axs2.set_ylim([0, 1])

axs3.plot([0,1,2,3,4], [output_uncert(result25all), output_uncert(resultcase1), output_uncert(resultcase2), output_uncert(resultcase3), output_uncert(resultcase4)],c='blue',marker ='o')
axs3.plot([0,1,2,3,4], [0, 
                        (output_uncert(result25all)-output_uncert(resultcase1))*100/output_uncert(result25all), 
                        (output_uncert(result25all)-output_uncert(resultcase2))*100/output_uncert(result25all), 
                        (output_uncert(result25all)-output_uncert(resultcase3))*100/output_uncert(result25all), 
                        (output_uncert(result25all)-output_uncert(resultcase4))*100/output_uncert(result25all)],
          c='green',marker='s')
axs3.tick_params(axis='both', which='major', labelsize=9)
axs3.tick_params(axis='both', which='minor', labelsize=9)
axs3.set_xlabel('Case No.', fontsize = 12)
axs3.set_ylabel('Percentage', fontsize = 12)
axs3.set_xticks([0, 1, 2, 3, 4])

axs3.set_ylim([0,90])
axs3.set_title('(b)',fontsize = 11)
axs3.legend(labels=['Uncertainty of (output)\n oil recovery', '%, Decrese in Uncertainty of \n(output) oil recovery'], fontsize = 9)
fig1.tight_layout()
#%%
#plt.subplot2grid(shape=(2,8), loc=(1, 4), colspan=4,rowspan=1)
axs[1,1].plot([1,2,3,4,5],[quantile25all[0],quantilecase1[0],quantilecase2[0],quantilecase3[0],quantilecase4[0]],c='blue',marker ='o')
axs[1,1].plot([1,2,3,4,5],[quantile25all[2],quantilecase1[2],quantilecase2[2],quantilecase3[2],quantilecase4[2]],c='green',marker='s')
axs[1,1].plot([1,2,3,4,5],[quantile25all[4],quantilecase1[4],quantilecase2[4],quantilecase3[4],quantilecase4[4]],c='red',marker='^')
plt.xlabel('Case No.', fontsize = 12)
plt.ylabel('Uncertainty in (output) \n oil recovery, %', fontsize = 12)
plt.title('(c)', y = 1, fontsize = 11)
plt.xlim([1, 5])
axs[1,1].set_ylim([0,20])
#plt.lim([0, 30])
plt.tick_params(axis='both', which='major', labelsize=9)
plt.tick_params(axis='both', which='minor', labelsize=9)
plt.legend(labels=['P90 oil recovery','P50 oil recovery','P10 oil recovery'], fontsize = 10, bbox_to_anchor=(0.7, 0.5))























