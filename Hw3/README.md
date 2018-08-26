# Machine Learning Hw3
Best Accuracy: 0.6689

## Structure:
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
 '''
