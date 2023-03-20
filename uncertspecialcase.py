import pandas as pd
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

#%%

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from math import sqrt


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
#%%
#0 % INPUT UNCERTAINTY
data = {'Yxs':[yxs],'Yps':[yps],'Kxs (g/l)':[ks],'Umax (h-1)':[umax], 'Xi (g/l)': [ximean],'Si (g/l)': [simean],
        'Ai (g/l)':[aimean], 'Resident Time (h)':[Tmean],'Flow Velocity (m/s)': [vmean],
        'Viscosity of injection fluid (Nsm-2)':[uwmean],'Initial IFT (mN/m)':[iftmaxmean],
        'Swir':[swirmean],'Sori':[somean]}
df1 = pd.DataFrame(data)
Y = pipe_kerasnn.predict(df1)

#%%
# Yxs,Yps, Ai, Sori, Uw, Si, v, Umax
random_state = 0
l_bounds =[vmean-(u1*sqrt(3)*vmean/100),uwmean-(u1*sqrt(3)*uwmean/100), 
               aimean-(u1*sqrt(3)*aimean/100), simean-(u1*sqrt(3)*simean/100),
               somean-(u1*sqrt(3)*somean/100), swirmean-(u1*sqrt(3)*swirmean/100),
               yxs-(u1*sqrt(3)*yxs/100), yps-(u1*sqrt(3)*yps/100), umax-(u1*sqrt(3)*umax/100)]
u_bounds = [vmean+(u1*sqrt(3)*vmean/100), uwmean+(u1*sqrt(3)*uwmean/100), 
                aimean+(u1*sqrt(3)*aimean/100), simean+(u1*sqrt(3)*simean/100),
                somean+(u1*sqrt(3)*somean/100), swirmean+(u1*sqrt(3)*swirmean/100),
                yxs+(u1*sqrt(3)*yxs/100), yps+(u1*sqrt(3)*yps/100), umax+(u1*sqrt(3)*umax/100)]
sampler = qmc.LatinHypercube(d=9, optimization = "random-cd")
sample = sampler.random(n=real)
sample = qmc.scale(sample, l_bounds, u_bounds)
v, uw, ai, si, so, swir, Yxs, Yps, Umax = [sample[:,i] for i in range(0,9)]
 
'''
'Yxs', 'Yps', 'Kxs (g/l)', 'Umax (h-1)','Xi (g/l)','Si (g/l)','Ai (g/l)',
'Resident Time (h)','Flow Velocity (m/s)','Viscosity of injection fluid (Nsm-2)',
'Initial IFT (mN/m)','Swir','Sori'
'''
data = {'Yxs':Yxs,'Yps':Yps,'Kxs (g/l)':Kxs,'Umax (h-1)':Umax, 'Xi (g/l)': xi,'Si (g/l)': si,
        'Ai (g/l)':ai, 'Resident Time (h)':T,'Flow Velocity (m/s)': v,
        'Viscosity of injection fluid (Nsm-2)':uw,'Initial IFT (mN/m)':iftmax,
        'Swir':swir,'Sori':so}
df1 = pd.DataFrame(data)
Y = pipe_kerasnn.predict(df1)
df1['recovery'] = Y
#with pd.ExcelWriter("D:\Project\MC Simulation\MEOR\May graphs\Specialcase3\NNdata.xlsx") as writer:
#    df1.to_excel(writer) 
df1.to_excel(r"D:\Project\MC Simulation\MEOR\May graphs\Specialcase4\NNdata.xlsx",sheet_name='Sheet1',engine = 'xlsxwriter')
#%%
from scipy import stats
from statistics import mean, stdev
import seaborn as sns
import matplotlib.ticker as ticker

datacase1 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase\NNdata.xlsx')
datacase2 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase2\NNdata.xlsx')
datacase3 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase3\NNdata.xlsx')
datacase4 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\Specialcase4\NNdata.xlsx')

datacase25 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\specialspecialcase\5.xlsx')
datacase24 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\specialspecialcase\4.xlsx')
datacase23 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\specialspecialcase\3.xlsx')
datacase22 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\specialspecialcase\2.xlsx')
datacase21 = pd.read_excel(r'D:\Project\MC Simulation\MEOR\May graphs\specialspecialcase\1.xlsx') #25all

resultcase25 = stats.describe(datacase25['recovery'], ddof=1, bias=False)#nobs, minmax, mean, variance, skewnwess, kurtosis
resultcase24 = stats.describe(datacase24['recovery'], ddof=1, bias=False)
resultcase23 = stats.describe(datacase23['recovery'], ddof=1, bias=False)
resultcase22 = stats.describe(datacase22['recovery'], ddof=1, bias=False)
resultcase21 = stats.describe(datacase21['recovery'], ddof=1, bias=False)

resultcase1 = stats.describe(datacase1['recovery'], ddof=1, bias=False)
resultcase2 = stats.describe(datacase2['recovery'], ddof=1, bias=False)
resultcase3 = stats.describe(datacase3['recovery'], ddof=1, bias=False)
resultcase4 = stats.describe(datacase4['recovery'], ddof=1, bias=False)

quantilecase1 = np.quantile(datacase1['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase2 = np.quantile(datacase2['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase3 = np.quantile(datacase3['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase4 = np.quantile(datacase4['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])

quantilecase25 = np.quantile(datacase25['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase24 = np.quantile(datacase24['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase23 = np.quantile(datacase23['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase22 = np.quantile(datacase22['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
quantilecase21 = np.quantile(datacase21['recovery'],[0.1, 0.25, 0.5, 0.75, 0.9])
#%%
figure, axs = plt.subplots(1,1, figsize= (4,4), dpi = 1000)
sns.distplot(datacase21['recovery'], bins = 200, hist = False, color = 'violet', kde_kws={'linestyle':'-.'}, ax = axs)
sns.distplot(datacase1['recovery'], bins = 200, hist = False, color = 'orange', kde_kws={'linestyle':'-'}, ax = axs)
sns.distplot(datacase2['recovery'], bins = 200, hist = False, color = 'red', kde_kws={'linestyle':'-'}, ax = axs)
sns.distplot(datacase3['recovery'], bins = 200, hist = False, color = 'blue', kde_kws={'linestyle':'-'}, ax = axs)
sns.distplot(datacase4['recovery'], bins = 200, hist = False, color = 'green', kde_kws={'linestyle':'-'}, ax = axs)
axs.set_ylabel('Probability, %', fontsize = 12)
axs.set_xlabel('Oil Recovery, %', fontsize = 12)
axs.legend(labels=['25% Input Uncertainty','Case1','Case2','Case3', 'Case4'])
#axe.text(0, 0.5, "two functions", bbox=dict(facecolor='red', alpha=0.5))
#axs.set_title('a) Probability Distribution of Oil Recovery for Cases 1,2,3,4', y = -0.3, fontsize = 12)
axs.set_xlim([0,35])
axs.set_ylim([0,0.2])
#axs[0,0].set_title('Case Study 1', y = -0.35, fontsize = 12)
axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals = 0))
#%%
figure, axs = plt.subplots(1,1, figsize= (4,4), dpi = 1000)
sns.distplot(datacase21['recovery'], bins = 200, hist = False, color = 'violet', kde_kws={'linestyle':'-.'}, ax = axs)
sns.distplot(datacase22['recovery'], bins = 200, hist = False, color = 'orange', kde_kws={'linestyle':'-'}, ax = axs)
sns.distplot(datacase23['recovery'], bins = 200, hist = False, color = 'red', kde_kws={'linestyle':'-'}, ax = axs)
sns.distplot(datacase24['recovery'], bins = 200, hist = False, color = 'blue', kde_kws={'linestyle':'-'}, ax = axs)
sns.distplot(datacase25['recovery'], bins = 200, hist = False, color = 'green', kde_kws={'linestyle':'-'}, ax = axs)
axs.set_ylabel('Probability, %', fontsize = 12)
axs.set_xlabel('Oil Recovery, %', fontsize = 12)
axs.legend(labels=['Case1','Case2','Case3', 'Case4','Case5'])
#axe.text(0, 0.5, "two functions", bbox=dict(facecolor='red', alpha=0.5))
#axs.set_title('a) Probability Distribution of Oil Recovery for Cases 1,2,3,4', y = -0.3, fontsize = 12)
axs.set_xlim([0,35])
axs.set_ylim([0,0.12])
axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals = 0))


#figure.tight_layout()
#%%
values,base = np.histogram(data25all['recovery'],bins=200)
values1,base1 = np.histogram(datacase1['recovery'],bins=200)
values2,base2 = np.histogram(datacase2['recovery'],bins=200)
values3,base3 = np.histogram(datacase3['recovery'],bins=200)
values4,base4 = np.histogram(datacase4['recovery'],bins=200)
cumulative = np.cumsum(values)
cumulative1 = np.cumsum(values1)
cumulative2 = np.cumsum(values2)
cumulative3 = np.cumsum(values3)
cumulative4 = np.cumsum(values4)
plt.xlabel('Oil Recovery, %')
plt.ylabel('1-Cummulative Probability Density') 
plt.plot(base[:-1], 1-(cumulative/len(data25all['recovery'])), c='blue', linestyle='--')
plt.plot(base1[:-1],1-(cumulative1/len(datacase1['recovery'])), c='green', linestyle='-.')
plt.plot(base2[:-1],1-(cumulative2/len(datacase2['recovery'])), c='red', linestyle=':')
plt.plot(base3[:-1],1-(cumulative3/len(datacase3['recovery'])), c='orange', linestyle='-')
plt.plot(base4[:-1],1-(cumulative4/len(datacase4['recovery'])), c='violet', linestyle='-.')
plt.legend(labels=['η = 25%','Case 1','Case 2','Case 3','Case 4'])
plt.xlim([0, 50])
plt.ylim([0, 1])
plt.show()
