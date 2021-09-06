"""
    "ResUnet-a d6" models creator
    NOT COMPLETE
    6 Residual blocks
    Procedural style code
    Dynamic parameters : 2 ^ base, Default
    
    Refs
    - https://arxiv.org/abs/1904.00592
    - https://github.com/ashishpatel26/satellite-Image-Semantic-Segmentation-Unet-Tensorflow-keras/blob/main/models/ResUnet.py
    - https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb
    
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, add, UpSampling2D
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from utilities import dice, iou, dice_loss, log_cosh_dice_loss

def ResUnet(input_shape, pre_trained=False, n_classes=1, base=5):
    
    MODEL_NAME = "ResUnet_base_" + str(base)
    
    # Input
    i = Input((input_shape[0], input_shape[1], input_shape[2]))
    # i = Input((128, 128,3))
    # s = Lambda(lambda x: preprocess_input(x)) (i)
            
    # ENCODERS 
    # First - E.g. base 2^5 = 32     
    conv = Conv2D(2**(base), kernel_size=(1,1), padding='same', strides=1)(i)
    conv = __conv_block(conv, base, strides=1)
    
    shortcut = Conv2D(2**(base), kernel_size=(1,1), padding='same', strides=1)(i)
    shortcut = BatchNormalization()(shortcut)
    
    output1 = add([conv, shortcut])    
    
    # Residual blocks
    res1 = __residual_block(output1, base+1, strides=2) # E.g. base 2^6 = 64 
    res2 = __residual_block(res1, base+2, strides=2) # E.g. base 2^7 = 128
    res3 = __residual_block(res2, base+3, strides=2) # E.g. base 2^8 = 256
    res4 = __residual_block(res3, base+4, strides=2) # E.g. base 2^9 = 512
    res5 = __residual_block(res4 , base+5, strides=2) # E.g. base 2^10 = 1024
    
    # BRIDGE
    psppool = __conv_block(res5, base+5, strides=1)
    psppool = __conv_block(psppool, base+5, strides=1)
    
    # DECODERS
    upsamp0 = __upsample_combine_block(psppool, res4)
    dconv0  = __residual_block(upsamp0, base+5)    
    
    upsamp1 = __upsample_combine_block(dconv0, res3)
    dconv1  = __residual_block(upsamp1, base+4)
    
    upsamp2 = __upsample_combine_block(dconv1, res2)
    dconv2  = __residual_block(upsamp2, base+3)    

    upsamp3 = __upsample_combine_block(dconv2, res1)
    dconv3  = __residual_block(upsamp3, base+2)
    
    upsamp4 = __upsample_combine_block(dconv3, output1)
    dconv4  = __residual_block(upsamp4, base+1)
    
    # Output layer
        
    if n_classes == 1:
        final_act = 'sigmoid'
    elif n_classes > 1:
        final_act = 'softmax' 
    
    output_layer = Conv2D(n_classes, (3, 3), padding="same", activation=final_act )(dconv4)
    
    model = Model(i, output_layer, name=MODEL_NAME)
    model.compile( optimizer=Adam(), loss=dice_loss, metrics=[dice] )
    model.summary()
    
    if pre_trained:    
        checkpoint_path = os.path.join('saved_models', model.name )
        model.load_weights( checkpoint_path )
        print("MODEL RESTORED", model.name)
        
    return model    


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


# ResUnet Blocks    
def __conv_block(x, filter_base, strides=1 ):
    conv = BatchNormalization()(x)
    conv = Activation("relu")(conv)
    conv = Conv2D(2**(filter_base), kernel_size=(3,3), padding='same', strides=strides)(conv)
    return conv
    
def __residual_block(x, filter_base, strides=1 ):
    res = __conv_block(x, filter_base, strides=strides)    
    res = __conv_block(res, filter_base, strides=1)
    
    skip = Conv2D(2**(filter_base), kernel_size=(3,3), padding='same', strides=strides )(x)
    skip = BatchNormalization()(skip)
    
    output = add([skip, res])
    
    return output

def __upsample_combine_block(x, skip):
    upconv = UpSampling2D((2,2))(x)
    upconv = concatenate([upconv, skip])
    return upconv