from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
import time, pickle
from keras.utils import to_categorical
from keras.optimizers import SGD
import csv

nb_classes = 10
class_name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'training samples')
print(X_test.shape[0], 'validation samples')
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

x = Input(shape=(32, 32, 3))
y = x
y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)

y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)

y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)

y = Flatten()(y)
y = Dense(units=128, activation='relu', kernel_initializer='he_normal')(y)
y = Dropout(0.5)(y)
# y = Dense(units=nb_classes, activation='softmax', kernel_initializer='he_normal')(y)
y = Dense(units=nb_classes, activation='softmax')(y)

# SGD (Stochastic Gradient Descent)
# lrate = 0.01
# decay = lrate / nb_epoch
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model1 = Model(inputs=x, outputs=y, name='model1')

model1.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# model1.compile(optimizer='sgd',loss='categorical_crossentropy')

conv_layers = []
if conv_layers == None or len(conv_layers) == 0 : 
    for i in range(len(model1.layers)):
        if str(type(model1.layers[i])) == "<class 'keras.layers.convolutional.Conv2D'>":
            conv_layers.append(i)
def log_weights(weights):
    for i in conv_layers:
        filename = './weights/'+str(i)+'.csv'
        with open(filename,'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(list(weights[i].flatten()))
        csvFile.close()
    return
    
for epoch in range(10):
    print("Epoch Number: ",epoch)
    for i in range(499):
        if i%100==0:
            print("Batch Number: ",i)
        start = 0+i*100
        end = 99+i*100
        cost = model1.train_on_batch(X_train[start:end,],y_train[start:end,])
        log_weights(model1.get_weights())
