'''import os
import pandas as pd
import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt

data, sampling_rate = librosa.load('D:\Sachin Workspace\Datasets\dev-clean\LibriSpeech\dev-clean\84\\121123\\84-121123-0001.flac')

plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)
plt.show()
'''



import IPython.display as ipd

# % pylab inline
import os
import pandas as pd
import librosa
import glob
import librosa.display
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

import os

data = pd.read_csv('D:/Sachin Workspace/Datasets/train/train.csv')
print(data)
#Choose a random audio file and listen
i = random.choice(data.ID)
i=132
ipd.Audio('D:/Sachin Workspace/Datasets/train/Train/'+str(i)+'.wav')


def extract_features(files):
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(os.path.abspath('D:/Sachin Workspace/Datasets/train/Train') + '/' + str(files.ID) + '.wav')

    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)

    # We add also the classes of each file as a label at the end
    label = files.Class

    return mfccs, chroma, mel, contrast, tonnetz, label

from numpy import genfromtxt
#my_data = genfromtxt('my_file.csv', delimiter=',')

# Applying the function to the train data by accessing each row of the dataframe
features_labels = data.apply(extract_features, axis=1)
#features_labels = genfromtxt('C:/Users/sujan/PycharmProjects/Voice_Recog/features_df.csv', delimiter=',')
#features_labels = pd.read_csv('C:/Users/sujan/PycharmProjects/Voice_Recog/features_df.csv')
print(features_labels)

features_df = pd.DataFrame(features_labels)
features_df.to_csv('features_df.csv', index=False)

# We create an empty list where we will concatenate all the features into one long feature
# for each file to feed into our neural network
features = []
for i in range(0, len(features_labels)):
    features.append(np.concatenate((features_labels[i][0], features_labels[i][1],
                features_labels[i][2], features_labels[i][3],
                features_labels[i][4]), axis=0))


print(len(features))
# Similarly, we create a list where we will store all the labels

labels = []
for i in range(0, len(features_labels)):
    labels.append(features_labels[i][5])

print(len(labels))
# to let me know when it's done
os.system('say -v Juan ya acabé');
np.unique(labels, return_counts=True)


# Setting our X as a numpy array to feed into the neural network
X = np.array(features)
y = np.array(labels)
print(y)
lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))
print(y)

# Checking our shapes
print(X.shape)
print(y.shape)


# Choosing the first 3435 files to be our train data
# Choosing the next 1000 files to be our validation data
# Choosing the next 1000 files to be our test never before seen data
# This is analogous to a train test split but we add a validation split and we are making
# we do not shuffle anything since we are dealing with several time series

X_train = X[:3435]
y_train = y[:3435]

X_val = X[3435:4435]
y_val = y[3435:4435]

X_test = X[4435:]
y_test = y[4435:]

print(data[:3435]['Class'].value_counts(normalize=True))
print(data[3435:4435]['Class'].value_counts(normalize=True))
print(data[4435:]['Class'].value_counts(normalize=True))

print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)
# Scaling
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)

from datetime import datetime
startTime = datetime.now()


def model_func(layer_one_neurons=128, layer_one_dropout=.5, layer_two_neurons=512,
               layer_two_dropout=.5, layer_three_neurons=128, layer_three_dropout=.5):
    model = Sequential()

    model.add(Dense(layer_one_neurons,
                    input_shape=(193,),
                    activation='relu'))
    model.add(Dropout(layer_one_dropout))
    model.add(Dense(layer_two_neurons,
                    activation='relu'))
    model.add(Dropout(layer_two_dropout))
    model.add(Dense(layer_three_neurons,
                    activation='relu'))
    model.add(Dropout(layer_three_dropout))
    model.add(Dense(10,
                    activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model


nn = KerasRegressor(model_func, batch_size=512, verbose=2)

params = {
    'epochs': [100],
    'layer_one_neurons': [193],
    'layer_two_neurons': [64, 128],
    'layer_three_neurons': [128, 256],
    'layer_one_dropout': [.25, .5],
    'layer_two_dropout': [.5, .75],
    'layer_three_dropout': [.5, .75]
}
gs = GridSearchCV(nn, param_grid=params, cv=3)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


print(datetime.now() - startTime)
os.system('say -v Juan ya acabé');


# build a simple dense model with early stopping with softmax for categorical classification
model = Sequential()
model.add(Dense(193, input_shape=(193,), activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

# fitting the model with the train data and validation with the validation data we used early stop with patience 15
history = model.fit(X_train, y_train, batch_size=256, epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop])

# Checking how our model looks like and how many parameters it has
model.summary()

# Check out our train accuracy and validation accuracy over epochs.
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

import matplotlib.pyplot as plt
# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')

# Set title
plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)
plt.xticks(range(0,100,5), range(0,100,5))
plt.legend(fontsize = 18);

# Using our model to get the predictions for our test data
preds = model.predict_classes(X_test)
# Looking at our test data as a dataframe to be able to compare prediction values
test = data[4435:]
# Setting our predictions column
test['preds'] = preds
# Changing the prediction values to their actual labels
test['preds'] = test['preds'].map({8:'siren', 9:'street_music', 7:'jackhammer',
                   4:'drilling', 3:'dog_bark', 2:'children_playing',
                   6:'gun_shot', 5:'engine_idling', 0:'air_conditioner',
                   1:'car_horn'})

# Looking at how accurate our model is just by looking at it
print(test)

# Slicing our dataframe into the files we got wrong from our predictions
print(test[test['Class']!=test['preds']])
# Calculating the actual test accuracy
print(round((1-len(test[test['Class']!=test['preds']])/len(test)),2))
predic_nn = model.predict_proba(X_test)
predic_nn = pd.DataFrame(predic_nn)
predic_nn.to_csv('predict_nn.csv', index=False)

