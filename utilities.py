"""
Utilties Function

@author: Avinash Rai
"""
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

'''
    Dice Coefficient - Evaluation metric
    From :https://github.com/seth814/Semantic-Shapes/blob/master/models.py
'''
def dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)


'''
    IOU Intersection over Union - Evaluation metric
    From :https://github.com/nikhilroxtomar/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/blob/master/train.py
'''
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


"""
    https://arxiv.org/pdf/2006.14822.pdf
    From https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
"""
def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
        
'''
    Plot Figure of Training Accuracy and Loss
    @history | obj | Tensorflow fit Scalar 
    @return None
'''    
def plot_training_history( fit_history, metric_name="accuracy" ):
    plt.subplots(2,2,figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(fit_history.history[metric_name], label='Train', color="blue")
    plt.plot(fit_history.history[ "val_" + metric_name], label='Validation', color="green")
    plt.legend(loc="upper left")
    plt.title(metric_name.upper())
    plt.subplot(1,2,2)
    plt.plot(fit_history.history['loss'], label='Train', color="blue")
    plt.plot(fit_history.history['val_loss'], label='Validation', color="green")
    plt.legend(loc="upper left")
    plt.title("LOSS")
    plt.show()