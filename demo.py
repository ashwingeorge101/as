from numpy import split
import streamlit as st
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
import matplotlib.pyplot as plt  # data-visualization
#%matplotlib inline
import seaborn as sns  # built on top of matplotlib
sns.set()
import pandas as pd  # working with data frames
import plotly.express as px
import numpy as np  # scientific computing
import missingno as msno  # analysing missing data
#import tensorflow as tf  # used to train deep neural network architectures
import tensorflow as tf  # used to train deep neural network architectures
from tensorflow.python.keras import layers
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title='demo',page_icon=':bar_chart:',layout='wide')
st.title("POWERTELL")
data=st.file_uploader('upload a file')
df=pd.read_csv(data)
df["Date-Hour(NMT)"] = df["Date-Hour(NMT)"].astype(np.datetime64)  # set the data type of the datetime column to np.datetime64
df.set_index("Date-Hour(NMT)", inplace=True)  # set the datetime columns to be the index
df.index.name = "datetime"  # change the name of the index
#d=st.number_input("Enter the datatime column number")
#j=st.number_input("Enter the solar production column number")
# name=[]
# name.append(df.columns[d])
# df["Datetime"]=df.name.copy
#df["Datetime"].astype(np.datetime64)
# df.set_index("Datetime", inplace=True)
#name=df.columns[d]
#solar=df.columns[j]
#df[name]=df[name].astype(np.datetime64)
#df.set_index(name, inplace=True)  # set the datetime columns to be the index
#df.index.name = "datetime"  # change the name of the index
correlation=df.corr(method ='pearson').values
len1 =len(correlation)
len2 = len(correlation[0])
for i in range(0,len1):
  for j in range(0,len2):
    if (correlation[i][j] == 1) | (correlation[i][j]< 0.4):
      correlation[i][j] = 0
array = []
for i in range(0,len1):
    for j in range(0,len2):
      if correlation[i][j] > 0:
        array.append(i)
num=set(array)
len3=len(num)
ab=[]
for i in num:
  ab.append(df.columns[i])
df=df[ab]
a=round(len(df.index)*0.2)
b = a
test_data=df.SystemProduction[-a:].copy()
train_df,test_df=df[:-a],df[-a:]
st.success('Success message')
train = train_df
scalers={}
for i in train_df.columns:
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s
test = test_df
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s
def split_series(series, n_past, n_future):
  
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)
n_past = 2
n_future =1
n_features = len3
X_train, y_train = split_series(train_df.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
X_test, y_test = split_series(test_df.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
# E1D1

encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)

encoder_states1 = encoder_outputs1[1:]

decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)
model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)

model_e1d1.summary()
# E2D2

encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]

decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])

decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)

model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)

model_e2d2.summary()
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
history_e1d1=model_e1d1.fit(X_train,y_train,epochs=25,validation_data=(X_test,y_test),batch_size=32,verbose=0,callbacks=[reduce_lr])
model_e2d2.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
history_e2d2=model_e2d2.fit(X_train,y_train,epochs=25,validation_data=(X_test,y_test),batch_size=32,verbose=0,callbacks=[reduce_lr])
pred_e1d1=model_e1d1.predict(X_test)
pred_e2d2=model_e2d2.predict(X_test)
for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]
    pred_e1d1[:,:,index]=scaler.inverse_transform(pred_e1d1[:,:,index])
    pred_e2d2[:,:,index]=scaler.inverse_transform(pred_e2d2[:,:,index])
    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])
from sklearn.metrics import mean_absolute_error
for index,i in enumerate(train_df.columns):
  print(i)
  for j in range(1,2):
    print("Hour ",j,":")
    print("MAE-E1D1 : ",mean_absolute_error(y_test[:,j-1,index],pred_e1d1[:,j-1,index]))
  print()
from sklearn.metrics import mean_absolute_error
for index,i in enumerate(train_df.columns):
  print(i)
  for j in range(1,2):
    print("Hour ",j,":")
    print("MAE-E2D2 : ",mean_absolute_error(y_test[:,j-1,index],pred_e2d2[:,j-1,index]))
  print()
x = len(pred_e1d1)
c=[]
#for i in pred_e1d1:
  #for j in range(0,1):
    #a=pred_e1d1[0][j][6]
  #c.append(a)
for i in range(0,x):
 for j in range(0,len3):
   a=pred_e1d1[i][0][len3-1]
 c.append(a)  
c = [0 if ele<0 else ele for ele in c]
st.success('Success message')
c.insert(0,0)
c.insert(0,1)
sns.set_context("poster")
fig,(ax1) = plt.subplots(1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
sns.lineplot(x = test_data.index, y=test_data, color="green", ax=ax1, label="original") # get the ground-truth validation data
sns.lineplot(x = test_data.index, y=c, color="red", dashes=True, ax=ax1, label="Forecast", alpha=0.5)  # get the forecast
# set the axis labels and title
ax1.set_xlabel("Date")
ax1.set_ylabel("Solar Generation (MW)")
ax1.set_title("Time Series Data"); 
st.pyplot(fig)

