"""
Initiate training using TensorflowGPU

@author: Avinash Rai
"""
import tensorflow
import os
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import shutil
from tensorflow.keras import backend

from DataGenerator_OpenCV import DataGenerator
# from DataGenerator_PIL import DataGenerator
from models_UNET import Unet
from models_FCN import FCN_8
from utilities import plot_training_history, dice, iou

# Free up RAM in case the model defintion cells were run multiple times
# backend.clear_session()

# Allow GPU memory growth
physical_devices = tensorflow.config.experimental.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Instance variables
n_classes      = 1
INPUT_SHAPE    = (768, 896, 3)
image_file_ext = r".tif"
masks_file_ext = r".TIF"
train_dir_p    = "datasets-binary/train"
val_dir_p      = "datasets-binary/validation"
model_choosen  = "fcn"
metric_choosen = "dice"


# Prepare Data generators
train_img_list = [os.path.join(train_dir_p + "/images", _) for _ in os.listdir(train_dir_p + "/images") if _.endswith(image_file_ext)]
train_mask_list = [os.path.join(train_dir_p + "/masks", _) for _ in os.listdir(train_dir_p + "/masks") if _.endswith(masks_file_ext)]
val_img_list = [os.path.join(val_dir_p + "/images", _) for _ in os.listdir(val_dir_p + "/images") if _.endswith(image_file_ext)]
val_mask_list = [os.path.join(val_dir_p + "/masks" , _) for _ in os.listdir(val_dir_p  + "/masks") if _.endswith(masks_file_ext)]

# Choose Segmentation Models
if model_choosen == "fcn":
    """ 
        Train FCN_8 Model
        - Base x ^ 5 / Parameters: 70,302,345
        - Input Batch Size: 3
    """
    trainDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = train_img_list,
                                       masks_paths  = train_mask_list,
                                       batch_size   = 3 )
    validDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = val_img_list,
                                       masks_paths  = val_mask_list,
                                       batch_size   = 3 )
                                       
    model = FCN_8(input_shape=INPUT_SHAPE, n_classes=n_classes, base=6)
                                        
elif model_choosen == "unet":
    
    """
        Train UNET Model
        - Base x ^ 6 / Parameters: 31,031,745
        - Input Batch Size: 1
    """
    trainDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = train_img_list,
                                       masks_paths  = train_mask_list,
                                       batch_size   = 1 )
    validDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = val_img_list,
                                       masks_paths  = val_mask_list,
                                       batch_size   = 1 )
                                       
    model = Unet(input_shape=INPUT_SHAPE, n_classes=n_classes, base=6 )


# Define Callbacks
callbacks = []

# Checkpoint - Save best weights
checkpoint = ModelCheckpoint( 
        os.path.join('saved_models', model.name ), 
        # monitor = 'val_mean_io_u',
        monitor = 'val_' + metric_choosen, 
        verbose = 1, 
        mode = 'max',
        save_weights_only = True,
        save_best_only = True)

# Default tensorBoard callback
log_directory = 'tb_logs\\' + model.name
if os.path.exists( log_directory ):
    shutil.rmtree( log_directory )
os.mkdir( log_directory )
tensorboardCallback = TensorBoard(log_dir=log_directory )

# Train model
history = model.fit( trainDataGenerator,
                    validation_data  = validDataGenerator,
                    steps_per_epoch  = len(trainDataGenerator),
                    #validation_steps = len(validDataGenerator),                       
                    epochs           = 50, 
                    verbose          = 1, 
                    callbacks        = [checkpoint, tensorboardCallback] )

plot_training_history(history, metric_name=metric_choosen )