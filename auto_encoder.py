# IMPORT PAKAGES
import math
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback



np.random.seed(1234)
tf.random.set_seed(1234)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(0)


# Auto Encoder Class

tf.keras.backend.clear_session()

# 사용자 정의 콜백 클래스

"""
n 에폭마다 모델 저장
model_save_path = '/path/model_epoch_{}.h5'
model_checkpoint_callback = CustomModelCheckPoint(freq = 10, directory = model_save_path)
model.fit(x,y,callbacks = [model_checkpoint_callback])
"""
class CustomModelCheckPoint(Callback):
    def __init__(self,freq, directory):
        super().__init__()
        self.freq = freq
        self.directory = directory
    def on_epoch_begin(self, epoch, logs = None):
        if self.freq > 0 and epoch % self.freq == 0:
            self.model.save(self.directory.format(str(epoch).zfill(3)))

class CustomProgress(Callback):
    def on_epoch_end(self, epoch, logs = None):
        if (epoch + 1) % 5 == 0 : 
            print('[{}] - EPOCH : {}, Ttrain Loss : {:.4f}, Valid Loss : {:.4f}'.format(datetime.datetime.now(),epoch + 1, logs['loss'],logs['val_loss']), flush = True)
            

            
"""
모델 저장후 불러올떄 custom activation 설정해 주어야됨

model = tf.keras.models.load_model(model_path, custom_objects = {'custom_activation' : custom_activation})
"""            
def custom_activation(x):
    return tf.sigmoid(x)*255



class CustomAEModel() :

    def __init__(self, input_shape, kernelN, kernelSize, kernelEx, strides, dropR, poolN) :
        """
        input_shape = (128,512,1)
        kernelN = (8,16,64,64,16,8)
        kernelSize = 3
        kernelEx = 1
        poolN = 3
        strides = 1
        dropR = 0
        """

        self.input_shape  = input_shape
        self.kernelN      = kernelN
        self.kernelSize   = kernelSize
        self.kernelEx     = kernelEx
        self.strides      = strides
        self.dropR        = dropR
        self.poolN        = poolN
        self.build()

    def ConvBlock(self, x, n_filters, kernel_size, strides) :
        conv = tf.keras.layers.Conv2D(n_filters, kernel_size = kernel_size, strides = strides, padding = 'same')(x)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('relu')(conv)
        return conv

    def build(self) :
        # Kernel Expand
        if self.kernelEx == 1:
            if self.input_shape[0] > self.input_shape[1] :
                kk = math.ceil(self.input_shape[0]/self.input_shape[1])
                _kernelSize = (self.kernelSize * kk , self.kernelSize)
                
            elif self.input_shape[0] < self.input_shape[1] :
                kk = math.ceil(self.input_shape[1]/self.input_shape[0]) 
                _kernelSize = (self.kernelSize , self.kernelSize * kk)
            else :
                _kernelSize = (self.kernelSize, self.kernelSize)
        else :
            _kernelSize = (self.kernelSize, self.kernelSize)
   
        # Encoder
        inputs = tf.keras.layers.Input(self.input_shape)
        enc_l = list(range(len(self.kernelN)//2))
        for encode_range in enc_l :
            if encode_range == 0 :
                conv = self.ConvBlock(x = inputs, n_filters = self.kernelN[encode_range], kernel_size = _kernelSize, strides = self.strides)
            else :
                conv = self.ConvBlock(x = conv, n_filters = self.kernelN[encode_range], kernel_size = _kernelSize, strides = self.strides)

            if encode_range in enc_l[-self.poolN:] :
                conv = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)

        if self.dropR > 0:
            conv = tf.keras.layers.Dropout(self.dropR)(conv)

        # Decoder
        dec_l = list(range(len(self.kernelN)//2, len(self.kernelN)))
        for decode_range in dec_l :
            if decode_range in dec_l[:self.poolN] :
                conv = tf.keras.layers.UpSampling2D(size = (2,2))(conv)
            conv = self.ConvBlock(x = conv, n_filters = self.kernelN[decode_range], kernel_size = _kernelSize, strides = self.strides)

        outputs = tf.keras.layers.Conv2D(self.input_shape[-1], 1, activation = custom_activation)(conv)
        self.model = tf.keras.Model(inputs = inputs, outputs = outputs)
        self.model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mse')
            

class Unet():
    
    def __init__(self, input_shape) :
        self.input_shape = input_shape
        self.build()
        
    def ConvBlock(self, x, n_filters) :
        conv = tf.keras.layers.Conv2D(n_filters, 3, padding = 'same')(x)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('relu')(conv)
        conv = tf.keras.layers.Conv2D(n_filters, 3, padding = 'same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('relu')(conv)
        
        return conv
    
    def EncoderBlock(self, x, n_filters):
        conv = self.ConvBlock(x,n_filters)
        
        # -- 가로방향만 Maxpooling
        pool = tf.keras.layers.MaxPooling2D(pool_size = (1,2))(conv)
        # -- 가로,세로 방향 Maxpooling
#         pool = tf.keras.layers.Maxpooling2D(pool_size = (2,2))(conv)
        return conv, pool

    def DecoderBlock(self,x,skip,n_filters):
        
        # 가로방향만 Up Sampling
        up = tf.keras.layers.UpSampling2D(size = (1,2))(x)
        up = tf.keras.layers.Conv2D(n_filters, 3, padding = 'same')(up)
        up = tf.keras.layers.BatchNormalization()(up)
        up = tf.keras.layers.Activation('relu')(up)
        merge = tf.keras.layers.concatenate([up, skip], axis = 3)
        conv = self.ConvBlock(merge, n_filters)
        
        return conv
    
    def build(self):
        inputs = tf.keras.layers.Input(self.input_shape)
        conv1, pool1 = self.EncoderBlock(inputs, 16)
        conv2, pool2 = self.EncoderBlock(pool1, 32)
        conv3        = self.ConvBlock(pool2, 64)
        drop3        = tf.keras.layers.Dropout(0.5)(conv3)
        conv4        = self.DecoderBlock(drop3, conv2, 32)
        conv5        = self.DecoderBlock(conv4, conv1, 16)
        outputs = tf.keras.layers.Conv2D(3,1, activation = custom_activation)(conv5)
        
        self.unet_model = tf.keras.Model(inputs = inputs, outputs = outputs)
        self.unet_model.compile(optimizer = tf.keras.optimizers.Adam(),loss='mse')
        
        return self.unet_model
    
    
    
class Mynet():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.build()
        
    def ConvBlock(self, x, n_filters) :
        conv = tf.keras.layers.Conv2D(n_filters, 3, padding = 'same')(x)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('relu')(conv)
        conv = tf.keras.layers.Conv2D(n_filters, 3, padding = 'same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('relu')(conv)
        
        return conv
    
    def build(self):
        
        inputs = tf.keras.layers.Input(self.input_shape)
        
        # Encoder
        conv = self.ConvBlock(inputs, 16)
        conv = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)
        
        conv = self.ConvBlock(conv,32)
        conv = tf.keras.layers.MaxPooling2D(pool_size = (1,2))(conv)
        
        conv = self.ConvBlock(conv,64)
        conv = tf.keras.layers.MaxPooling2D(pool_size = (1,2))(conv)
        
        conv = tf.keras.layers.Dropout(0.5)(conv)
        
        # Decoder
        conv = tf.keras.layers.UpSampling2D(size = (1,2))(conv)
        conv = self.ConvBlock(conv,64)
        
        conv = tf.keras.layers.UpSampling2D(size = (1,2))(conv)
        conv = self.ConvBlock(conv,32)
        
        conv = tf.keras.layers.UpSampling2D(size = (2,2))(conv)
        conv = self.ConvBlock(conv,16)
        
        outputs = tf.keras.layers.Conv2D(3,1,activation = custom_activation)(conv)
        
        self.mynet_model = tf.keras.Model(inputs = inputs, outputs = outputs)
        self.mynet_model.compile(optimizer=tf.keras.optimizers.Adam(), loss = "mse")
        
        return self.mynet_model
