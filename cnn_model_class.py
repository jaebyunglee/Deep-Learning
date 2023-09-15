import numpy as np
import tensorflow as tf

from tensorflow.keras.callbakcs import Callback
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score

class CnnModel():
    def __init__(self, input_shape, Depth, kernelN, kernelSize, strides, l2, dropR, init_bias, kinit):
        
        """
        >> 예제코드
        cnn_model = CnnModel((300,300,3), 8, 4, 3, 1, 0.01, 0.5, 0, 'orthogonal').cnn_model
        cnn_model.summary()
        
        >> init_bias True 이면 1번으로 아니면 2번으로
        1) init_bias = np.log(sum(TRAIN_Y ==1)/sum(TRAIN_Y == 0))
        2) init_bias = 0
        
        
        >> kinit 종류
        random_normal / glorot_normal / he_normal
        random_uniform / glorot_uniform / he_uniform
        orthogonal
        """
        self.input_shape = input_shape
        self.Depth       = Depth
        self.kernelN     = kernelN
        self.kernelSize  = kernelSize
        self.strides     = strides
        self.l2          = l2
        self.dropR       = dropR
        self.init_bias   = init_bias
        self.kinit       = kinit
        self.build()
        
        


    def build(self):

        #Input Layer
        self.input_layer = tf.keras.Input(shape = self.input_shape)

        #CNN Layer
        for _Depth in range(self.Depth):
            if _Depth == 0:
                self.conv = tf.keras.layers.Conv2D(self.kernelN * ((_Depth //2)+1), kernel_size = (self.kernelSize,self.kernelSize), padding = 'same', strides = self.strides, name = f'conv{_Depth+1}_filter', kernel_initializer = self.kinit)(self.input_layer)
            else :
                self.conv = tf.keras.layers.Conv2D(self.kernelN * ((_Depth //2)+1), kernel_size = (self.kernelSize,self.kernelSize), padding = 'same', strides = self.strides, name = f'conv{_Depth+1}_filter', kernel_initializer = self.kinit)(self.conv)

            self.conv = tf.keras.layers.BatchNormalization(name = f'conv_{_Depth+1}_nor')(self.conv)
            self.conv = tf.keras.layers.Activation('relu', name = f'conv_{_Depth+1}_act')(self.conv)

            # Depth/2 마다 max pooling
            if (_Depth + 1)%(int(self.Depth/2)) == 0:
                self.conv = tf.keras.layers.MaxPooling2D(pool_size = 2, name = f'conv_{_Depth+1}_pooling')(self.conv)

        # Flatten Layer
        self.flatten = tf.keras.layers.Flatten()(self.conv)
        self.flatten = tf.keras.layers.Dropout(self.dropR, seed=1234)(self.flatten)

        # Bias Setting
        if self.init_bias != 0:
            output_bias = tf.keras.initializers.Constant(self.init_bias)
        else :
            output_bias = None

        # Output Layer
        self.output_layer = tf.keras.layers.Dense(1, activation = 'sigmoid',
                                                  kernel_initializer = self.kinit,
                                                  kernel_regularizer = tf.keras.regularizers.l2(l = self.l2),
                                                  bias_initializer = output_bias,
                                                  name = 'dense')(self.flatten)
        self.cnn_model = tf.keras.Model(inputs = self.input_layer, outputs = self.output_layer)

        @tf.function
        def wacc(y_true, y_pred):

            def my_numpy_func(y_true, y_pred):
                y_true = y_true.squeeze()
                y_pred = y_pred.squeeze()

                length = len(y_true)
                try :
                    pos_w = length/len(y_true[y_true==1])
                    neg_w = length/len(y_true[y_true==0])
                except ZeroDivisionError :
                    pos_w = 1
                    neg_w = 1

                sample_weight = np.zeros(y_true.shape)
                sample_weight[y_true == 1] = pos_w
                sample_weight[y_true == 0] = neg_w
                y_pred = (y_pred>=0.5)+0

                w_acc_score = accracy_score(y_true, y_pred, sample_weight = sample_weight)
                w_acc_score = tf.cast(w_acc_score, tf.float32) # retunr 값은 tf.float32

                return w_acc_score

            score = tf.numpy_function(my_numpy_func, [y_true, y_pred], tf.float32) #파이썬 함수를 감싸서 tf로 사용
            score = tf.cast(score, tf.float32) # retunr 값은 tf.float32

            return score 
                
                
                    
        
        self.cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['AUC','acc',wacc]) 
        #self.cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['AUC','acc',wacc], run_eagerly=true) # Deberg

        return self.cnn_model
