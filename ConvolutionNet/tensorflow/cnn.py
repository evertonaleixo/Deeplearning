# Choose the underlying compiler - tensorflow or theano
import json
import os 
with open(os.path.expanduser('~') + "/.keras/keras.json","r") as f:
    compiler_data = json.load(f)
compiler_data["backend"] = "tensorflow"   
with open(os.path.expanduser('~') + '/.keras/keras.json', 'w') as outfile:
    json.dump(compiler_data, outfile)

# import all the required packages
import numpy as np
from keras.models import Model
import keras.backend as K
from keras.layers import BatchNormalization, MaxPooling2D, Convolution2D, Activation, Input
from keras.optimizers import SGD
if False:
    from keras.engine.topology import Merge


# Load data from pickle object
import pickle
class_labels_count = 1
with open('/mnist/train', 'rb') as f:
    (train_data, train_label) = pickle.load(f)
    if (len(train_data.shape) == 3):        
        train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]).astype('float32') / 255   
    if (len(train_label.shape) == 1):
        from keras.utils import np_utils
        class_labels_count = len(set(train_label))
        train_label = np_utils.to_categorical(train_label, len(set(train_label)))
    else:
        class_labels_count = train_label.shape[1]

val_data = []
if(0 > 0): 
    train_data = train_data[:int(len(train_data) * (1-0)), :,:]
    train_label = train_label[:int(len(train_label)*(1-0))]
    val_data = train_data[int(len(train_data)*(1-0)):,:,:]
    val_label = train_label[int(len(train_label)*(1-0)):]
else:
    if('/mnist/valid'):
        with open('/mnist/valid', 'rb') as f:
            (val_data, val_label) = pickle.load(f)
            if (len(val_data.shape) == 3):
                val_data = val_data.reshape(val_data.shape[0], 1, val_data.shape[1], val_data.shape[2]).astype('float32') / 255
            if (len(val_label.shape) == 1):
                from keras.utils import np_utils
                val_label = np_utils.to_categorical(val_label, class_labels_count)
    else:
        print('Validation set details not provided')
  
test_data = []
if(0 > 0):  
    train_data = train_data[:int(len(train_data)*(1-0)),:,:]
    train_label = train_label[:int(len(train_label)*(1-0))]
    test_data = train_data[int(len(train_data)*(1-0)):,:,:]
    test_label = train_label[int(len(train_label)*(1-0)):]
else:
    if('/mnist/test'):
        with open('/mnist/test', 'rb') as f:
            (test_data, test_label) = pickle.load(f)
            if (len(test_data.shape) == 3):        
                test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2]).astype('float32') / 255
            if (len(test_label.shape) == 1):
                from keras.utils import np_utils
                test_label = np_utils.to_categorical(test_label, class_labels_count)

print(train_data.shape)
batch_input_shape_MeuMnist = train_data.shape[1:4]
train_batch_size = 16
test_batch_size = 16

# Choose the hardware platform - GPU or CPU
import tensorflow as tf
if('cpu' in 'CPU'.lower()):
    device_id = '/cpu'
if('gpu' in 'CPU'.lower()):
    device_id = '/gpu:0'
if('multigpu' in 'CPU'.lower()):
    device_id = '/gpu:1'

with tf.device(device_id):

    #Input Layer
    MeuMnist = Input(shape=batch_input_shape_MeuMnist)
    #Convolution2D Layer
    Convolution2D_1 = Convolution2D(64, 3, 3, init = 'lecun_uniform', border_mode = 'valid', subsample = (1, 1), dim_ordering = 'th', bias = False, name = '6754f5f0.26248c')(MeuMnist)
    #Batch Normalization Layer
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    Convolution2D_1 = BatchNormalization(axis=bn_axis,name='bn_6754f5f0.26248c')(Convolution2D_1)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_2 = Activation('relu', name = '8b41ac21.8609d')(Convolution2D_1)
    #Pooling2D Layer
    Pooling2D_3 = MaxPooling2D(pool_size = (2, 2), border_mode = 'valid', strides = (2, 2), name = '57cf03d3.4f3b9c')(ReLU_2)
    #Convolution2D Layer
    Convolution2D_4 = Convolution2D(64, 3, 3, init = 'lecun_uniform', border_mode = 'valid', subsample = (1, 1), dim_ordering = 'th', bias = False, name = '4a47d3e4.7d933c')(Pooling2D_3)
    #Batch Normalization Layer
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    Convolution2D_4 = BatchNormalization(axis=bn_axis,name='bn_4a47d3e4.7d933c')(Convolution2D_4)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_5 = Activation('relu', name = '22d09540.b53eba')(Convolution2D_4)
    #Softmax Activation Layer
    SoftmaxWithLoss_6 = Activation('softmax', name = '5594037a.5cb2ac')(ReLU_5)
    defined_loss = 'binary_crossentropy'

    # Define a keras model
    model = Model(input=[MeuMnist], output = [SoftmaxWithLoss_6])
    defined_metrics={}

    # Set the required hyperparameters    
    num_epochs = 100

    # Defining the optimizer function
    optimizer_fn = SGD(lr=0.01, momentum=0.9, decay=0.8)

    # Compile and train the model
    if not defined_metrics:
        defined_metrics=None
    model.compile(loss=defined_loss, optimizer=optimizer_fn, metrics=defined_metrics)
    model.fit(train_data, train_label, batch_size=train_batch_size, nb_epoch=num_epochs, verbose=1)

    # validate the model
    if (val_data):
        val_scores = model.evaluate(val_data, val_label, verbose=1)
        
    # test the model
    if (test_data):
        test_scores = model.evaluate(test_data, test_label, verbose=1)

