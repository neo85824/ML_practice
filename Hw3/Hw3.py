import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, BatchNormalization, Conv2D, Cropping2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sys

def DataProcess(train_path, test_path):
    #---Training Data---
    train_data = pd.read_csv(train_path)
    #train_data= train_data[:1000]
    y_train = []
    x_train = []
    for i in range(train_data.shape[0]):
        y_train.append(train_data.iloc[i,0])
        x_train.append([])
        l = train_data.iloc[i,1].split(' ')
        for j in range(48*48):
            x_train[i].append(l[j])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    #reshape to (batch, channels, rows, columns)
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    y_train = y_train.reshape(y_train.shape[0], 1)

    #normalization
    x_train = x_train / 255

    #one-hot
    y_train = np_utils.to_categorical(y_train, 7)

    #--Testing Data---
    test_data = pd.read_csv(test_path)
    #test_data = test_data[:1000]
    y_test = []
    x_test = []
    for i in range(test_data.shape[0]):
        y_test.append(test_data.iloc[i,0])
        x_test.append([])
        l = test_data.iloc[i,1].split(' ')
        for j in range(48*48):
            x_test[i].append(l[j])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')

    #reshape to (batch, rows, columns, channels)
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    #normalization
    x_test = x_test / 255

    #one-hot
    y_test = np_utils.to_categorical(y_test, 7)

    return x_train, y_train,x_test, y_test




def main():
    #Fetch Data
    x_train, y_train, x_test, y_test = DataProcess(sys.argv[1],sys.argv[2])

    #parameters setup
    epochs = 2
    batch_size = 128

    #NN Structure

    model = Sequential()

    model.add(Convolution2D(filters=64, kernel_size=(3,3), padding='same' , activation='relu', input_shape=(48,48,1)) )
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3,3), padding='same' , activation='relu') )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding = 'same' ))    
    model.add(Dropout(0.4))

    model.add(Convolution2D(filters=128, kernel_size=(3,3), padding='same' , activation='relu') )
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3,3), padding='same' , activation='relu') )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding = 'same' ))    
    model.add(Dropout(0.4))

    model.add(Convolution2D(filters=256, kernel_size=(3,3), padding='same', activation='relu' ) )
    model.add(BatchNormalization()) 
    model.add(Convolution2D(filters=256, kernel_size=(3,3), padding='same', activation='relu' ) )
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=256, kernel_size=(3,3), padding='same', activation='relu' ) )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding = 'same' ))    
    model.add(Dropout(0.4))
    
    model.add(Convolution2D(filters=512, kernel_size=(3,3), padding='same', activation='relu' ) )
    model.add(BatchNormalization()) 
    model.add(Convolution2D(filters=512, kernel_size=(3,3), padding='same', activation='relu' ) )
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=512, kernel_size=(3,3), padding='same', activation='relu' ) )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding = 'same' )) 
    model.add(Dropout(0.4))


    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=7, activation='softmax'))

    #model summary
    model.summary()
    
    #optimizer
    sgd = SGD(lr=1e-5, momentum=0.2)
    adam = Adam(lr = 1e-4)

    #compile
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #checkpoint setup
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', monitor = 'val_loss', verbose=1, save_best_only = True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min', baseline=None)

    #Training
    val_x = x_train[int(x_train.shape[0]*0.9) : ]
    val_y = y_train[int(y_train.shape[0]*0.9) : ]
    x_train = x_train[ : int(x_train.shape[0]*0.9)]
    y_train = y_train[ : int(y_train.shape[0]*0.9)]

    datagen = ImageDataGenerator( featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True)
    datagen.fit(x_train, augment = True)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size= batch_size),
                    steps_per_epoch=5*len(x_train)//batch_size, epochs=epochs, validation_data=(val_x,val_y), shuffle=True , callbacks=[early_stopping], verbose=1)
    #history = model.fit(x_train, y_train, validation_split=0.1,batch_size=batch_size, shuffle=True , callbacks=[checkpointer], epochs=epochs, verbose=2)
    
    #Save
    model.save('weights.hdf5')

    #Load best weight
    #model.load_weights('weights_aug_0.6689.hdf5')

    #Plot Loss 
    
    eps = range(1,len(history.history['loss'])+1)
    plt.plot(eps, history.history['loss'], 'b', label='loss')
    plt.plot(eps, history.history['acc'], 'r', label='acc')
    plt.plot(eps, history.history['val_loss'],'g', label = 'val_loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    
 
    #Testing 
    result = model.predict(x_test)
    score = model.evaluate(x_test, y_test)
    print("Loss and Accu:", score)

if __name__ == "__main__":
   main()