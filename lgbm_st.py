import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time
plt.rcParams["text.usetex"]=True
plt.rcParams["text.latex.preamble"]=r"\usepackage{amsmath}\boldmath"
url="https://raw.githubusercontent.com/cakirogl/soil_shear/main/shearstrengthofsoil_new.csv";
df=pd.read_csv(url)
x, y = df.iloc[:, :-1], df.iloc[:, -1]
scaler=MinMaxScaler()
input_container = st.container()
output_container = st.container()
ic1,ic2=input_container.columns(2)
#oc=output_container.columns(1)
with ic1:
    W=st.number_input("**Water content [%]:**",min_value=15.0,max_value=250.0,step=5.0,value=33.82)
    LL=st.number_input("**Liquid limit [%]:**",min_value=30.0,max_value=70.0,step=5.0,value=54.53)
    PL=st.number_input("**Plastic limit [%]:**",min_value=2.0,max_value=25.0,step=1.0,value=20.21)
with ic2:
    PI=st.number_input("**Plasticity index [%]:**", min_value=15.0, max_value=55.0, step=5.0, value=34.32)
    LI=st.number_input("**Liquidity index [%]:**", min_value=0.0, max_value=7.0, step=0.01, value=0.0)

new_sample=np.array([[W, LL, PL, PI, LI]],dtype=object)
x=scaler.fit_transform(x)
lgbm_settings={'n_estimators': 14, 'num_leaves': 240, 'min_child_samples': 6, 
               'learning_rate': 0.40900945122677634, 'log_max_bin': 10, 
               'colsample_bytree': 0.7726323753260981, 'reg_alpha': 0.021112250160766365, 
               'reg_lambda': 0.0014243465097419357, "verbose":-1}
model=LGBMRegressor(**lgbm_settings)
model.fit(x,y)
train_color="teal";test_color="fuchsia";eqn="-0.076+x"
timeStart = time()
timeEnd=time()

fig, ax=plt.subplots()
with ic2:
    #st.write(f"W={new_sample[0][0]}, LL={new_sample[0][1]}, PL={new_sample[0][2]}, PI={new_sample[0][3]}, LI={new_sample[0][4]}")
    st.write(f"**Shear strength = **{model.predict(new_sample)[0]:.3f}** MPa**")
    
#with oc:
#    st.write(f"**Shear strength = **{model.predict(new_sample)[0]:.2f}** MPa**")
#ax.scatter(yhat_train, y_train, color='teal', label=r'$\mathbf{LightGBM\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='fuchsia', label=r'$\mathbf{LightGBM\text{ }test}$')

#ax.set_xlabel(r'$\mathbf{CL_{predicted}\text{ }[kWh]}$', fontsize=14)
#ax.set_ylabel(r'$\mathbf{CL_{test}\text{ }[kWh]}$', fontsize=14)
#xmax=300;ymax=300;
#xk=[0,xmax];yk=[0,ymax];ykPlus10Perc=[0,ymax*1.1];ykMinus10Perc=[0,ymax*0.9];
#ax.tick_params(axis='x',labelsize=14)
#ax.tick_params(axis='y',labelsize=14)
#ax.plot(xk,yk, color='black')
#ax.plot(xk,ykPlus10Perc, dashes=[2,2], color='black')
#ax.plot(xk,ykMinus10Perc,dashes=[2,2], color='black')
#ax.grid(True)
#ratio=1.0
#xmin,xmax=ax.get_xlim()
#ymin,ymax=ax.get_ylim()
#ax.set_aspect(ratio*np.abs((xmax-xmin)/(ymax-ymin)));

def linearRegr(x,a0,a1):
    return a0+a1 * np.array(x)

#coeffs, covmat=curve_fit(f=linearRegr, xdata=np.concatenate((yhat_train,yhat_test)).flatten(),ydata=np.concatenate((y_train,y_test)).flatten())
#print(f"a0={coeffs[0]}, a1={coeffs[1]}")
#regr=linearRegr(xk,coeffs[0], coeffs[1])
#ax.plot(xk,regr, label=r"$\mathbf{y=0.125+0.999x}$")
#plt.legend(loc='upper left',fontsize=12)
#plt.show()