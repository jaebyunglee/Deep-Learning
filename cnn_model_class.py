import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbakcs import Callback
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, fbeta_score


np.random.seed(1234)
np.random.set_seed(1234)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(0)

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
            

def w_acc_fn(y_true, y_pred):
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
    
    w_acc_score = accuracy_score(y_true, y_pred, sample_weight = sample_weight)
    return w_acc_score


class CnnModel():
    def __init__(self, input_shape, Depth, kernelN, kernelSize, strides, l2, lr, dropR, init_bias, kinit):
        
        """
        >> 예제코드
        cnn_model = CnnModel((300,300,3), 8, 4, 3, 1, 0.01, 0.003, 0.1, 0, 'orthogonal').cnn_model
        cnn_model.summary()
        
        >> init_bias True 이면 1번으로 아니면 2번으로
        1) init_bias = np.log(sum(TRAIN_Y ==1)/sum(TRAIN_Y == 0))
        2) init_bias = 0
        
        
        >> kinit 종류 (아래 종류를 스트링으로 넣으면됨)
        ex - self.kinit : 'orthogonal'
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
        self.lr          = lr
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

        """
        Custom Metric 예시
        1. Custom Loss function도 같음 -> numpy로 loss함수를 만들고 이거를 tf.numpy_function으로 감싸준다
        2. return값은 tf.cast(..., tf.float32)로 변경
        """
        @tf.function
        def wacc(y_true, y_pred):
            """
            Numpy로 작성된 코드
            """
            def my_numpy_func(y_true, y_pred):
                w_acc_score = w_acc_fn(y_true, y_pred)
                w_acc_score = tf.cast(w_acc_score, tf.float32) # return 값은 tf.float32
                return w_acc_score

            """
            Numpy로 작성된 함수를 tf에서 사용할수 있게 적용
            """
            score = tf.numpy_function(my_numpy_func, [y_true, y_pred], tf.float32) #파이썬 함수를 감싸서 tf로 사용
            score = tf.cast(score, tf.float32) # return 값은 tf.float32
            return score 
                
        @tf.function
        def fbeta(y_true, y_pred):
            def my_numpy_func(y_true, y_pred):
                """
                beta : 0.5 -> Pricision을 2배 중요하게 생각
                beta : 2.0 -> Recall을    2배 중요하게 생각
                """
                y_pred = (y_pred >= 0.5) + 0
                _score = fbeta_score(y_true, y_pred, beta = 0.5, zero_divison = 0)
                _score = tf.cast(_score, tf.float32) # return 값은 tf.float32
                return _score
            """
            Numpy로 작성된 함수를 tf에서 사용할수 있게 적용
            """
            score = tf.numpy_function(my_numpy_func, [y_true, y_pred], tf.float32) #파이썬 함수를 감싸서 tf로 사용
            score = tf.cast(score, tf.float32) # return 값은 tf.float32
            resutn score
                    
        self.cnn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr), loss = 'binary_crossentropy', metrics = ['AUC','acc',wacc]) 
        #self.cnn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)', loss = 'binary_crossentropy', metrics = ['AUC','acc',wacc], run_eagerly=true) # Deberg

        return self.cnn_model
