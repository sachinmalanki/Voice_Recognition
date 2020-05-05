import os
import pandas as pd
from glob import glob
import numpy as np
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
# from path import Path

import IPython.display as ipd
# % pylab inline
import os
import pandas as pd
import librosa
import glob
import librosa.display
import random

import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
import os

data = pd.read_csv('D:/Sachin Workspace/Datasets/train/train.csv')
filename= 'D:/Sachin Workspace/Datasets/train/Train/0.wav'

i = random.choice(data.ID)
ipd.Audio('D:/Sachin Workspace/Datasets/train/Train/'+str(i)+'.wav')

X, sample_rate = librosa.load(filename, sr=None, res_type='kaiser_fast')
S = librosa.feature.melspectrogram(y=X, sr=sample_rate)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='linear')
plt.show()


# Although this function was modified and many parameteres were explored with, most of it
# came from Source 18 (sources in the READ.ME)

def images(file):
    # We define the audiofile from the rows of the dataframe when we iterate through
    # every row of our dataframe for train, val and test
    audiofile = os.path.join(os.path.abspath('D:/Sachin Workspace/Datasets/train/Train') + '/' + str(file.ID) + '.wav')

    # Loading the image with no sample rate to use the original sample rate and
    # kaiser_fast to make the speed faster according to a blog post about it (on references)
    X, sample_rate = librosa.load(audiofile, sr=None, res_type='kaiser_fast')

    # Setting the size of the image
    fig = plt.figure(figsize=[1, 1])

    # This is to get rid of the axes and only get the picture
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    # This is the melspectrogram from the decibels with a linear relationship
    S = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='linear')

    # Here we choose the path and the name to save the file, we will change the path when
    # using the function for train, val and test to make the function easy to use and output
    # the images in different folders to use later with a generator
    name = file.ID
    if not os.path.exists('D:/Sachin Workspace/Datasets/train/val_images/'):
        os.mkdir('D:/Sachin Workspace/Datasets/train/val_images/')
    file = 'D:/Sachin Workspace/Datasets/train/val_images/' + str(name) + '.png'

    # Here we finally save the image file choosing the resolution
    plt.savefig(file, dpi=500, bbox_inches='tight', pad_inches=0)

    # Here we close the image because otherwise we get a warning saying that the image stays
    # open and consumes memory
    plt.close()

data = pd.read_csv('D:/Sachin Workspace/Datasets/train/train.csv')
train = data[:3435]
val = data[3435:4435]
test = data[4435:]

train.apply(images, axis=1);
test.apply(images, axis=1);
val.apply(images, axis=1);
plt.close('all')

os.system('say -v Juan ya acabé');

data = pd.read_csv('D:/Sachin Workspace/Datasets/train/train.csv',dtype=str)

train = data[:3435]
val = data[3435:4435]
test = data[4435:]

# Function to change the file names to the image names to use them later
def make_jpg(files):
    return str(files)+'.jpg'

train['ID'] = train["ID"].apply(make_jpg)
val['ID'] = val["ID"].apply(make_jpg)
test['ID'] = test["ID"].apply(make_jpg)

# Rescaling the images as usual to feed into the CNN
datagen=ImageDataGenerator(rescale=1./255.)
train_generator=datagen.flow_from_dataframe(
    dataframe=train,
    directory="train_images",
    x_col="ID",
    y_col="Class",
    batch_size=32,
    shuffle=False,
    class_mode="categorical",
    target_size=(64,64))

val_generator=datagen.flow_from_dataframe(
    dataframe=val,
    directory="val_images",
    x_col="ID",
    y_col="Class",
    batch_size=32,
    shuffle=False,
    class_mode="categorical",
    target_size=(64,64))

test_generator=datagen.flow_from_dataframe(
    dataframe=test,
    directory="test_images",
    x_col="ID",
    y_col="Class",
    batch_size=32,
    shuffle=False,
    class_mode="categorical",
    target_size=(64,64))
plt.close('all')
os.system('say -v Juan ya acabé');


# Building our model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# Compiling using adam and categorical crossentropy
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# Fitting our CNN with 250 epochs and setting the results to history for visuals
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=108,
                    validation_data=val_generator,
                    validation_steps=32,
                    epochs=250)



# Check out our train accuracy and validation accuracy over epochs.
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

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

plt.legend(fontsize = 18)
plt.show();

# Generating a new test dataframe that includes the labels for comparison and
# checking the accuracy of our model with this never before seen data
test = data[4435:]

# Generating predictions on our never seen data with the model we built
preds = model.predict_generator(test_generator)

# Creating an empty list to store the values where the predictions are the maximum out
# of all the 10 possible values
p = []
for i in range(0, len(preds)):
    p.append(np.where(preds[i] == max(preds[i])))

# Creating an empty list to store the values in a clean list
predictions = []
for i in range(0, len(preds)):
    predictions.append(p[i][0][0])

# Adding those predictions to our test dataframe
test['predictions'] = predictions

# Changing the numeric values to their corresponding labels
test['predictions'] = test['predictions'].map({8: 'siren', 9: 'street_music', 7: 'jackhammer',
                                               4: 'drilling', 3: 'dog_bark', 2: 'children_playing',
                                               6: 'gun_shot', 5: 'engine_idling', 0: 'air_conditioner',
                                               1: 'car_horn'})

# Checking the percentage of correct predictions
print(round(len(test[test['Class'] == test['predictions']]) / len(test), 2))
# Saving the dataframe to use with our Dense NN to use as a voting classifier
predic_cnn = pd.DataFrame(preds)

predic_cnn.to_csv('predict_cnn.csv', index=False)
os.system('say -v Juan ya acabé');
