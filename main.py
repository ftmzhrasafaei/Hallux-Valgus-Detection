import cv2
import matplotlib.pyplot as plt
import os
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import copy
import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout,MaxPooling2D
from tensorflow.keras.models import Model
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
from tensorflow.keras.utils import to_categorical

import seaborn as sns
from sklearn.metrics import confusion_matrix
import copy

def Confusion_Graph(cm, tit= False):

    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    if tit:
        plt.title(tit)
    ## Display the visualization of the Confusion Matrix.
    plt.show()





def GetImage(path,val=False):
    list_files = os.listdir(path)
    nos = []
    scale = 0
    for image_name in list_files:
        #counter += 1
        image = cv2.imread(f'{path}/{image_name}',0)
       # print(image)

        binary_image = cv2.threshold(image,0, 255 , cv2.THRESH_OTSU)[1]
        #print(binary_image)
        #filtered = cv2.medianBlur(binary_image, 35)
        filtered = binary_image
        cropped = cv2.boundingRect(filtered)


        cropped_image = filtered[cropped[1]:cropped[3]+cropped[1],cropped[0]:cropped[2]+ cropped[0]]
        scale1 = cropped_image.shape[0]/cropped_image.shape[1]
        if scale1 > scale:
            scale = scale1
        rimg = cv2.resize(cropped_image, (100,int(100*scale1)))
        #rimg = rimg.tolist()
        nos.append(rimg)
        if not val:
            nos = nos + DataGen(rimg , 1)
    scale = math.ceil(scale * 100)
    return nos, scale

def SameSize(nos,scale):
    finalno = []
    black = cv2.imread('dataset/black.png',0)

    for im in nos:
        if im.shape[0] < scale:
            howmuch = int((scale - im.shape[0]))
            black = cv2.resize(black, (100,howmuch))
            im = cv2.vconcat([im, black])
        finalno.append(im[25:100][0:100])


    return finalno

def Create_DF(pathyes, pathno):


    yes,scaley = GetImage(pathyes)
    no,scalen = GetImage(pathno,True)
    scale = max([scalen,scaley])
    n = SameSize(no,scale)
    nl = [0 for i in range(len(n))]
    y = SameSize(yes,scale)
    yl = [1 for i in range(len(y))]
    images = n + y
    labels = nl + yl
    c = list(zip(images, labels))

    random.shuffle(c)
    random.shuffle(c)
    random.shuffle(c)
    images, labels = zip(*c)

    return images, labels,scale

def Split(a,b ,frc):
    c = list(zip(a, b))
    x = []


    lentest = int(len(a)*frc)

    for k in range(lentest):
        x.append(c.pop(random.randint(0, len(c)-1)))

    a,b = zip(*c)
    a1, b1 = zip(*x)
    return a,b,a1,b1


def DataGen(tobeaug , iteration):
    datagen = ImageDataGenerator(
            horizontal_flip=True,
            fill_mode='nearest')

    test_img = tobeaug
    img = image.img_to_array(test_img)
    img = img.reshape((1,) + img.shape)

    i = 0
    imglist = []
    for batch in datagen.flow(img, save_prefix='test', save_format='png'):  # this loops runs forever until we break, saving images to current directory with specified prefix

        cv2.imwrite('augmentning.png', image.img_to_array(batch[0]*255))
        imagek = cv2.imread('augmentning.png',0)
        binary_image = cv2.threshold(imagek,0, 255 , cv2.THRESH_OTSU)[1]
        filtered= cv2.medianBlur(binary_image, 35)
        imglist.append(filtered)
        i += 1
        if i > iteration:
            break
    return imglist

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
    class_names = [1,0]
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)


def show_image(img, labe, gues):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    if labe == 0:
        label = 'yes'
    else:
        label = 'no'

    if gues == 0:
        guess = 'yes'
    else:
        guess = 'no'

    plt.title("Excpected: " + label)
    plt.xlabel("Guess: " + guess)
    print("Excpected: " + label+ " and Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 9:
                return int(num)
        else:
            print("Try again...")






def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))











pyes = "dataset/pics/yes"
pno = "dataset/pics/no"
d, l ,s= Create_DF(pyes , pno)

a,b,a1,b1 = Split(d,l , 0.1)
train_set = np.array(a)
train_label = np.array(b)
train_set = train_set / 255.0
s=75

test_images = np.array(a1)
test_labels = np.array(b1)
test_images = test_images / 255.0

num_filters = 8
filter_size = 4
pool_size = 2

model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(s, 100,1)),
  Dropout(.2),
  Conv2D(3,8, activation='relu'),
  Dropout(.2),
  #MaxPooling2D(pool_size=pool_size),
  Conv2D(3,16, activation='relu'),
  Dropout(.2),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(.2),
  Dense(32, activation='relu'),
  Dense(2, activation='softmax')
])

#sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy',f1])
history = model.fit(train_set, train_label, epochs=20 ,validation_split=0.1, shuffle=True)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss of classifier')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy of classifier')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['f1'])
plt.plot(history.history['val_f1'])
plt.title('model F-score of classifier')
plt.ylabel('F Score')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

test_loss, test_acc, test_f1 = model.evaluate(test_images,  test_labels)



for num in range(5):
    image = train_set[num]
    label = train_label[num]
    predict(model, image, label)


predictions = model.predict(test_images)
p = [np.argmax(item) for item in predictions]
cm = confusion_matrix(test_labels ,p )
Confusion_Graph(cm,'On Test DataSet')

predictions = model.predict(train_set)
p = [np.argmax(item) for item in predictions]
cm = confusion_matrix(train_label ,p )
Confusion_Graph(cm,'On Train DataSet')
