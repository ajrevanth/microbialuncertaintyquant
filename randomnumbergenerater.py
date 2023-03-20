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
data = pd.read_excel(r'UALHSK.xlsx',sheet_name ='Sheet1') #change directory for excel file
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


uncertainty = int (input("Type uncertainty(%):"))
u = uncertainty
param = input("what do u want to vary ?\n 1) All parameters \n 2) Microbial kinetic parameters \n 3) Reservoir parameters \n 4) Operational parameters")

if u == 5 or 10 or 15:
    real = 20000
if u == 20 or 25:
    real = 30000
else:
    print('wrong uncertainty')
    
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



if param == '4':
    l_bounds = [vmean-(u*sqrt(3)*vmean/100),uwmean-(u*sqrt(3)*uwmean/100),
                  aimean-(u*sqrt(3)*aimean/100), simean-(u*sqrt(3)*simean/100), ximean-(u*sqrt(3)*ximean/100)]
    u_bounds = [vmean+(u*sqrt(3)*vmean/100), uwmean+(u*sqrt(3)*uwmean/100),
                    aimean+(u*sqrt(3)*aimean/100), simean+(u*sqrt(3)*simean/100), ximean+(u*sqrt(3)*ximean/100)]
    sampler = qmc.LatinHypercube(d=5, optimization = "random-cd")
    sample = sampler.random(n=real)
    sample = qmc.scale(sample, l_bounds, u_bounds)
    v, uw, ai, si, xi =  [sample[:,i] for i in range(0,5)]
if param == '3':
    l_bounds = [somean-(u*sqrt(3)*somean/100), swirmean-(u*sqrt(3)*swirmean/100), iftmaxmean-(u*sqrt(3)*iftmaxmean/100)]
    u_bounds = [somean+(u*sqrt(3)*somean/100), swirmean+(u*sqrt(3)*swirmean/100), iftmaxmean+(u*sqrt(3)*iftmaxmean/100)]
    sampler = qmc.LatinHypercube(d=3, optimization = "random-cd")
    sample = sampler.random(n=real)
    sample = qmc.scale(sample, l_bounds, u_bounds)
    so, swir, iftmax =  [sample[:,i] for i in range(0,3)]
if param == '2':
    l_bounds = [yxs-(u*sqrt(3)*yxs/100), yps-(u*sqrt(3)*yps/100), ks-(u*sqrt(3)*ks/100), umax-(u*sqrt(3)*umax/100)]
    u_bounds = [yxs+(u*sqrt(3)*yxs/100), yps+(u*sqrt(3)*yps/100), ks+(u*sqrt(3)*ks/100), umax+(u*sqrt(3)*umax/100)]
    sampler = qmc.LatinHypercube(d=4, optimization = "random-cd")
    sample = sampler.random(n=real)
    sample = qmc.scale(sample, l_bounds, u_bounds)
    Yxs, Yps, Kxs, Umax =  [sample[:,i] for i in range(0,4)]
if param == '1':
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
df1 = pd.DataFrame(data)
Y = pipe_kerasnn.predict(df1)
df1['recovery'] = Y
df1.to_excel("NNdata.xlsx",)