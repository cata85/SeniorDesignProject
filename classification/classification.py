import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import keras
import random

from sklearn import svm, datasets
from skimage.transform import resize
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, AveragePooling2D
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import keras_applications
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications import InceptionResNetV2
from keras.applications import Xception
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D

import functools
from keras.utils import multi_gpu_model
import multiprocessing as mp


# GLOBALS
DATADIR = r"C:\Users\carte\Desktop\BeeMachine\Bombus_images\Bumble_iNat_BugGuide_BBW"
CATEGORIES = ["Bombus_affinis", "Bombus_appositus", "Bombus_auricomus", "Bombus_bifarius", "Bombus_bimaculatus", "Bombus_borealis", 
              "Bombus_caliginosus", "Bombus_centralis", "Bombus_citrinus", "Bombus_crotchii", "Bombus_cryptarum", "Bombus_fernaldae_flavidus",
              "Bombus_fervidus", "Bombus_flavifrons", "Bombus_fraternus", "Bombus_frigidus", "Bombus_griseocollis", "Bombus_huntii", 
              "Bombus_impatiens", "Bombus_insularis", "Bombus_melanopygus", "Bombus_mixtus", "Bombus_morrisoni", "Bombus_nevadensis", 
              "Bombus_occidentalis", "Bombus_pensylvanicus_sonorus", "Bombus_perplexus", "Bombus_rufocinctus", "Bombus_sandersoni", 
              "Bombus_sitkensis", "Bombus_sylvicola", "Bombus_ternarius", "Bombus_terricola", "Bombus_vagans", "Bombus_vandykei", "Bombus_vosnesenskii"]
# CATEGORIES = ["Bombus Affinis", "Bombus Appositus", "Bombus Auricomus", "Bombus Bifarius", "Bombus Bimaculatus", "Bombus Borealis", 
#               "Bombus Caliginosus", "Bombus Centralis", "Bombus Citrinus", "Bombus Crotchii", "Bombus Cryptarum", "Bombus Fernaldae",
#               "Bombus Fervidus", "Bombus Flavifrons", "Bombus Fraternus", "Bombus Frigidus", "Bombus Griseocollis", "Bombus Huntii", 
#               "Bombus Impatiens", "Bombus Insularis", "Bombus Melanopygus", "Bombus Mixtus", "Bombus Morrisoni", "Bombus Nevadensis", 
#               "Bombus Occidentalis", "Bombus Pensylvanicus", "Bombus Perplexus", "Bombus Rufocinctus", "Bombus Sandersoni", 
#               "Bombus Sitkensis", "Bombus Sylvicola", "Bombus Ternarius", "Bombus Terricola", "Bombus Vagans", "Bombus Vandykei", "Bombus Vosnesenskii"]
NUM_CLASSES = len(CATEGORIES) #Number of classes (e.g., species)
IMG_SIZE = 299                #length and width of input images
TRAINING_DATA = []
BATCH_SIZE = 16


def main(op):
    if op == 'train':
        create_training_data()
        X, y = resize_img()
        X_train, X_test, y_train, y_test = split_data(X, y)
        weights_dict = weights(y_train)
        train_datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')
        model = create_model()
        mcp_save, reduce_lr_loss = setup_early_stopping()
        model = compile_model()
        history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                steps_per_epoch = X_train.shape[0] // BATCH_SIZE,
                                epochs=30,
                                validation_data=(X_test, y_test),  
                                verbose=1,
                                class_weight=weights_dict,
                                callbacks=[mcp_save, reduce_lr_loss],
                                use_multiprocessing=False,
                                workers=4)
        save_model(model)
    if op == 'load':
        model = load_model()
        path = "C:/Users/carte/Desktop/BeeMachine/Bombus_images/Bumble_iNat_BugGuide_BBW/Bombus_affinis/0K9KLKWKIKT0UQA09QZSEQLSBQHS6QD0KKPKQKNKHKT00KPKLKO05QNK6QHSVQOKKKUKRK6K5Q10HK2K0KV00KA0SKNK.jpg" # Affinis
        path2 = "C:/Users/carte/Desktop/BeeMachine/Bombus_images/Bumble_iNat_BugGuide_BBW/Bombus_citrinus/1RLQWRJK1RHQARRQUR80JQX0YQI0CQ20Q0XQR090FQU000P0K020K0E0L0SQJR20JRLQTQ40000Q3RW0.jpg"           # Citrinus
        path3 = "C:/Users/carte/Desktop/BeeMachine/Bombus_images/Bumble_iNat_BugGuide_BBW/Bombus_huntii/1HMHDHXHRRRLVZXLVZ5L9Z8LUZRLVHIHTHIHJH5LWZNHNZ6HBZ8LPZRL9ZHLPZZL6Z4LUZ7L9Z2H1Z9HDHKL1ZMLBZ.jpg"   # Huntii
        img_array = plt.imread(path3)
        img_array = resize(img_array, (IMG_SIZE, IMG_SIZE,3))
        img_array = np.array([img_array,])
        model_out = model.predict(img_array)[0,:]
        probabilities = np.argsort(model_out)
        print(f'1: {CATEGORIES[probabilities[-1]]}\n2: {CATEGORIES[probabilities[-2]]}\n3: {CATEGORIES[probabilities[-3]]}')
    pass


def create_training_data():
    for category in CATEGORIES:
      path=os.path.join(DATADIR,category) #path to genus category
      class_num = CATEGORIES.index(category)
      for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            new_array = cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB)
            TRAINING_DATA.append([new_array, class_num])
        except Exception as e:
            pass


def resize_img():
    X = []
    y = []
    for features, label in TRAINING_DATA:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,3)
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2) #split into training and test data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, X_test, y_train, y_test


def weights(y_train):
    weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    weights_dict = dict(enumerate(weights))
    return weights_dict


def create_model():
    base_model = InceptionV3(include_top=False, pooling ='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.06)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.06)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    return model


def setup_early_stopping():
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min')
    return mcp_save, reduce_lr_loss


def compile_model(model):
    top3_acc = functools.partial(keras.metrics.sparse_top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'

    model.compile(loss="sparse_categorical_crossentropy", 
                optimizer="sgd", 
                metrics=['accuracy'])
    return model


def save_model(model):
    model.save('model.h5')


def load_model():
    model = tf.keras.models.load_model('C:/Users/carte/Desktop/BeeMachine/model_InceptionV3_04-12-2020.h5')
    return model


if __name__ == '__main__':
    main('load')
    



# NOTES
# from skimage.transform import resize
# path = "C:/Users/carte/Desktop/BeeMachine/Bombus_images/Bumble_iNat_BugGuide_BBW/Bombus_affinis/0K9KLKWKIKT0UQA09QZSEQLSBQHS6QD0KKPKQKNKHKT00KPKLKO05QNK6QHSVQOKKKUKRK6K5Q10HK2K0KV00KA0SKNK.jpg"
# model = tf.keras.models.load_model('C:/Users/carte/Desktop/BeeMachine/model_InceptionV3_04-12-2020.h5')
# data = []
# img_array = cv2.imread(path)   
# img_array = cv2.resize(img_array, (299, 299))
# img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) 
# img_array = img_array.astype('float32')
# data.append(img_array)
# data = np.array(data).reshape(-1, 299, 299, 3)
# data = tf.cast(data, dtype=tf.float32)
# model_out = model.predict(data) 
# val = np.argmax(model_out)
# CATEGORIES[val]


# CORRECT
