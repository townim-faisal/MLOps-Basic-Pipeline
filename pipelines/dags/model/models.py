import re
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# AlexNet model
class AlexNet(Model):
    def __init__(self, input_shape, num_classes):
        super(AlexNet, self).__init__(name='')

        self.conv1 = Conv2D(96, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal')
                        
        self.maxpool1 = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)

        self.conv2 = Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')
        self.maxpool2 = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)

        self.conv3 = Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')

        self.conv4 = Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')

        self.conv5 = Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')

        self.maxpool3 = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)
    
        self.flatten1 = Flatten()
        self.dropout1 = Dropout(0.5)
        self.fc1 = Dense(2048, activation= 'relu')
        self.dropout2 = Dropout(0.5)
        self.fc2 = Dense(2048, activation= 'relu')
        self.fc3 = Dense(1000, activation= 'relu')
        self.fc4 = Dense(num_classes, activation= 'softmax')
        
    def call(self, input_tensor, training=False):
        
        x = self.conv1(input_tensor)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
    
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.flatten1(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
    
        return x