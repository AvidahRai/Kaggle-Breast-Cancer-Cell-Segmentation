"""
    FCN-8 models creator
    Dynamic parameters : 2 ^ base
    
    Refs
    - https://github.com/seth814/Semantic-Shapes/blob/master/models.py
    - https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, Dropout, Add
from tensorflow.keras.layers import MaxPooling2D, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import backend as K

from utilities import dice, iou, dice_loss, log_cosh_dice_loss

def FCN_8(input_shape, pre_trained=False, n_classes=1, base=4):
    
    MODEL_NAME = "FCN_8_base_" + str(base)
    
    if n_classes == 1:
        loss      = 'binary_crossentropy'
        final_act = 'sigmoid'
    elif n_classes > 1:
        loss      = 'categorical_crossentropy'
        final_act = 'softmax'

    b = base
    i = Input(input_shape)
    s = Lambda(lambda x: x / 255)(i)
    
    ## Block 1
    x = Conv2D(2**b, (3, 3), activation='relu', padding='same', name='block1_conv1')(s)
    x = Conv2D(2**b, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    
    # Block 2
    x = Conv2D(2**(b+1), (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(2**(b+1), (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(2**(b+2), (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    pool3 = x

    # Block 4
    x = Conv2D(2**(b+3), (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(2**(b+3), (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
    x = Conv2D(2**(b+3), (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    conv6 = Conv2D(2048 , (7, 7) , activation='relu' , padding='same', name="conv6")(pool5)
    conv6 = Dropout(0.5)(conv6)
    conv7 = Conv2D(2048 , (1, 1) , activation='relu' , padding='same', name="conv7")(conv6)
    conv7 = Dropout(0.5)(conv7)

    pool4_n = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(pool4)
    u2 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    u2_skip = Add()([pool4_n, u2])

    pool3_n = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(pool3)
    u4 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(u2_skip)
    u4_skip = Add()([pool3_n, u4])

    o = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), padding='same',
                        activation=final_act)(u4_skip)
      
    model = Model(inputs=i, outputs=o, name=MODEL_NAME)
    # model.compile(optimizer=Adam(1e-4), loss=loss, metrics=["accuracy", dice] )
    # model.compile(optimizer=Adam(1e-4), loss="mse", metrics=[ MeanIoU(num_classes=2) ] )
    model.compile( optimizer=Adam(), loss=log_cosh_dice_loss, metrics=[dice] )
    # model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=[dice])
    model.summary()
    
    if pre_trained:    
        checkpoint_path = os.path.join('saved_models', model.name )
        model.load_weights( checkpoint_path )
        print("MODEL RESTORED", model.name)

    return model