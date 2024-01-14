#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow import keras
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import warnings
import glob as glob
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

train_df = pd.read_csv("train_7/5days/train-3.csv")
test_df = pd.read_csv("test_7/5days/test-3.csv")
r_df = pd.read_csv("train_7/5days/train-2.csv")
e_df = pd.read_csv("test_7/5days/test-2.csv")
#train_df= train_df.sample(frac=1, random_state=0)
print(f"Total images for training: {len(train_df)}")
print(f"Total images for testing: {len(test_df)}")
#print(train_df)

y_train = pd.get_dummies(r_df["event"])
y_test = pd.get_dummies(e_df["event"])

train_dir = "train_7/5days/"
test_dir = "test_7/5days/"

# In[58]:
x_train = []
x_test = []

image_paths = train_df["image_name"].values.tolist()
for idx, path in enumerate(image_paths):
  img_data = tf.io.read_file(os.path.join(train_dir, path))
  img_data = tf.io.decode_jpeg(img_data)
  img_data = tf.image.resize(img_data,[299, 299])

  x_train.append(img_data)
  
x_train = np.array(x_train)/255.0

image_paths = test_df["image_name"].values.tolist()
for idx, path in enumerate(image_paths):
  img_data = tf.io.read_file(os.path.join(test_dir, path))
  img_data = tf.io.decode_jpeg(img_data)
  img_data = tf.image.resize(img_data,[299, 299])

  x_test.append(img_data)

x_test = np.array(x_test)/255.0

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print("abc")
print(x_train.shape[2])
# In[60]:


from tensorflow.keras import datasets, layers, models
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

model = tf.keras.applications.inception_v3.InceptionV3()
model_sequential = tf.keras.Sequential([
    model,
    #tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

#x = model.output
#x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
#x = tf.keras.layers.Dense(2, activation="softmax")(x)
#model_functional = tf.keras.Model(inputs=model.input, outputs = x)

#model_functional.summary()
model_sequential.summary()


#model_functional.compile(#optimizer='adam',
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss="categorical_crossentropy", optimizer=SGD(learning_rate=0.01, momentum=0.8),
              #metrics=['accuracy'])

model_sequential.compile(#optimizer='adam',
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss="categorical_crossentropy", optimizer=SGD(),
              metrics=['accuracy'])

#model_sequential.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#history = model.fit(x_train, y_train, epochs=10)
#history = model_functional.fit(x_train, y_train, epochs=1, validation_split=0.2,)
history = model_sequential.fit(x_train, y_train, epochs=120, validation_split=0.2,)

# In[ ]:


#model_functional.evaluate(x_test, y_test)
#model_sequential.evaluate(x_test, y_test)
_, accuracy = model_sequential.evaluate(x_test, y_test)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

x = model_sequential.predict(x_train)
a = model_sequential.predict(x_test)

print(x)
#x = np.where(x < 0.5, 0, 1)
#a = np.where(a < 0.5, 0, 1)
print(a)


print(x)
import numpy as np

def std_to_np(df):
    df_list = []
    df = np.array(df)
    for i in range(0, len(df) - 4, 1):
        df_s = df[i:i+5]
        df_list.append(df_s)
    return np.array(df_list)

X_train_np_array = std_to_np(x)
X_test_np_array = std_to_np(a)


y_train_new = y_train[4::]
y_test_new = y_test[4::]

print(a)
print(X_train_np_array.shape)
print(y_train_new)

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import LSTM
#from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
'''model = Sequential()
#model.add(LSTM(50, return_sequences=True, dropout=0.8, input_shape=(X_train_np_array.shape[1], X_train_np_array.shape[2])))
model.add(LSTM(50, dropout=0.2, input_shape=(X_train_np_array.shape[1], X_train_np_array.shape[2])))
#model.add(SimpleRNN(50, activation='relu', input_shape=(X_train_np_array.shape[1], X_train_np_array.shape[2])))
#model.add(LSTM(2000, return_sequences=True, dropout=0.6))
#model.add(LSTM(50, dropout=0.3))
#model.add(GRU(50,))

#model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

print(X_train_np_array.shape)
print(X_train_np_array.shape[2])
print(model.summary())'''

'''model.compile(#optimizer='adam',
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss="categorical_crossentropy", 
              optimizer=SGD(),
              metrics=['accuracy'])'''

'''model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#history = model.fit(x_train, y_train, epochs=10)
history = model.fit(X_train_np_array, y_train_new, epochs=500, validation_split=0.2,)
#history = model.fit([X_train_np_array[0], X_train_np_array[1]], y_train_new, epochs=100,)

print(X_test_np_array)
y_test_1 = model.predict(X_test_np_array)

# 予測結果の2値化
print(y_test_1)
print("onehot")
y_test_1 = np.where(y_test_1 < 0.5, 0, 1)
print(y_test_1)


_, accuracy = model.evaluate(X_test_np_array, y_test_new)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")'''


from sklearn.metrics import accuracy_score
# 時系列分割のためTimeSeriesSplitのインポート
from sklearn.model_selection import TimeSeriesSplit
from keras.layers import Dropout

tscv = TimeSeriesSplit(n_splits=4)
valid_scores = []

for fold, (train_indices, valid_indices) in enumerate(tscv.split(X_train_np_array)):
    model = Sequential()
    model.add(LSTM(10, batch_input_shape=(None, X_train_np_array.shape[1], X_train_np_array.shape[2])))
    model.add(Dropout(0.2))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer=SGD(), metrics=['accuracy'])

    #history = model.fit(X_train_np_array, y_train_new, epochs=100, validation_split=0.2, batch_size=64)
    history = model.fit(X_train_np_array, y_train_new, epochs=200, validation_split=0.2)
    # 予測
    y_test_pred = model.predict(X_test_np_array)

    # 予測結果の2値化
    y_test_pred = np.where(y_test_pred < 0.5, 0, 1)

    # 予測精度の算出と表示
    score = accuracy_score(y_test_new, y_test_pred)
    print(f'fold {fold} accracy: {score}')

    # 予測精度スコアをリストに格納
    valid_scores.append(score)
    print(y_test_pred)
print(f'valid_scores: {valid_scores}')
cv_score = np.mean(valid_scores)
print(f'CV score: {cv_score}')


# In[ ]:




