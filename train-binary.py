"""
Initiate training using TensorflowGPU

Notes:
    - CHECK CONFIGURATIONS BEFORE TRAINING
    - Binary Segmentation FCN-8 and UNET
    - Input Batch Size: dynamic (* Cannot train more than this size on Nvidia GTX 980 ti)
    
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
from models_RESUNET import ResUnet

from utilities import plot_training_history, dice, iou

# Free up RAM in case the model defintion cells were run multiple times
# backend.clear_session()

# Allow GPU memory growth
physical_devices = tensorflow.config.experimental.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Instance variables/ ******* CHECK CONFIGURATIONS BEFORE TRAINING *********
N_CLASSES      = 1
INPUT_SHAPE    = (768, 896, 3)
image_file_ext = r".tif"
masks_file_ext = r".TIF"
train_dir_p    = "datasets-binary/train"
val_dir_p      = "datasets-binary/validation"

FCN_TRAIN_PARAMS  = dict(
                        experiment_1 = dict( batch_size=3, base=6, epochs = 100 ), # params:70,302,345,  Learning Rate:1e-4 | Max on Nvidia GTX 980ti | Dice 0.083 Late improvements 
                        experiment_2 = dict( batch_size=3, base=6, epochs = 150 ), # params:70,302,345, Learning Rate:0.01 | Converged at Dice 0.034 
                        experiment_3 = dict( batch_size=7, base=3, epochs = 200 ), # params:10,859,857, Learning Rate:0.01 | Converged at Dice 0.0298
                        experiment_4 = dict( batch_size=5, base=5, epochs = 100 ), # params:33,577,321, Learning Rate:3e-4  | Converged at Dice 0.0300
                        experiment_5 = dict( batch_size=3, base=6, epochs = 150 ) # params:70,302,345, Learning Rate:3e-4  | Converged at Dice 0.0300
                       )
UNET_TRAIN_PARAMS = dict( 
                        experiment_1 = dict( batch_size=1, base=6, epochs = 100 ), # params:31,031,745, Max on Nvidia GTX 980ti, Learning Rate:0.01 | Model Diverged
                        experiment_2 = dict( batch_size=4, base=4, epochs = 200 ), # params:1,941,105 | Converged at Dice 0.1631 
                        experiment_3 = dict( batch_size=5, base=3, epochs = 150 )  # params:485,817 | Converged at Dice 0.18756 
                        )
RESUNET_TRAIN_PARAMS = dict(
                        experiment_1 = dict( batch_size=1, base=4, epochs = 50 ), # params:25,180,913, Max batch size, Loss Log Cash Dice Loss 1 on Nvidia GTX 980ti, | Interrupted Overfitting
                        experiment_2 = dict( batch_size=1, base=4, epochs = 100 ), #Increased epochs, | Interrupted - Overfitting Low training loss/High Val loss 
                        experiment_3 = dict( batch_size=1, base=3, epochs = 100 ), #params:6,304,569 | Interrupted- Overfitting Low training loss/High Val loss 
                        experiment_4 = dict( batch_size=1, base=3, epochs = 70 ), # Removed Scaling | Interrupted- Overfitting Low training loss/High Val loss 
                        experiment_5 = dict( batch_size=1, base=3, epochs = 70 ), # Using dice loss | Best weights saved at Dice 0.24504, Overfitting Later
                        experiment_6 = dict( batch_size=1, base=2, epochs = 70 ), #params:1,580,813, Using dice loss | Best weights saved at Dice 0.25588, Overfitting Later
                        experiment_7 = dict( batch_size=1, base=4, epochs = 70 ), # Using dice loss | Interrupted Overfitting
                       )                       
                        
MODEL_CHOOSEN  = "resunet" # Either "fcn", "unet" or "resunet"
METRIC_CHOSEN  = "dice" # e.g. accuracy, iou, dice
ITERATION      = "experiment_7" # params keys


# Prepare Data generators
train_img_list = [os.path.join(train_dir_p + "/images", _) for _ in os.listdir(train_dir_p + "/images") if _.endswith(image_file_ext)]
train_mask_list = [os.path.join(train_dir_p + "/masks", _) for _ in os.listdir(train_dir_p + "/masks") if _.endswith(masks_file_ext)]
val_img_list = [os.path.join(val_dir_p + "/images", _) for _ in os.listdir(val_dir_p + "/images") if _.endswith(image_file_ext)]
val_mask_list = [os.path.join(val_dir_p + "/masks" , _) for _ in os.listdir(val_dir_p  + "/masks") if _.endswith(masks_file_ext)]

# Choose Segmentation Models
if MODEL_CHOOSEN == "fcn":

    trainDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = train_img_list,
                                       masks_paths  = train_mask_list,
                                       batch_size   = FCN_TRAIN_PARAMS[ITERATION]["batch_size"], 
                                       augmentation = True )
    validDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = val_img_list,
                                       masks_paths  = val_mask_list,
                                       batch_size   = FCN_TRAIN_PARAMS[ITERATION]["batch_size"] )
                                       
    model = FCN_8(input_shape=INPUT_SHAPE, n_classes=N_CLASSES, base=FCN_TRAIN_PARAMS[ITERATION]["base"] )
                                        
elif MODEL_CHOOSEN == "unet":
    
    trainDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = train_img_list,
                                       masks_paths  = train_mask_list,
                                       batch_size   = UNET_TRAIN_PARAMS[ITERATION]["batch_size"],
                                       augmentation = True )
    validDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = val_img_list,
                                       masks_paths  = val_mask_list,
                                       batch_size   = UNET_TRAIN_PARAMS[ITERATION]["batch_size"] )
                                       
    model = Unet(input_shape=INPUT_SHAPE, n_classes=N_CLASSES, base=UNET_TRAIN_PARAMS[ITERATION]["base"] )

elif MODEL_CHOOSEN == "resunet":
    
    trainDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = train_img_list,
                                       masks_paths  = train_mask_list,
                                       batch_size   = RESUNET_TRAIN_PARAMS[ITERATION]["batch_size"],
                                       augmentation = True )
    validDataGenerator = DataGenerator(INPUT_SHAPE, 
                                       images_paths = val_img_list,
                                       masks_paths  = val_mask_list,
                                       batch_size   = RESUNET_TRAIN_PARAMS[ITERATION]["batch_size"] )
                                    
    model = ResUnet(input_shape=INPUT_SHAPE, n_classes=N_CLASSES, base=RESUNET_TRAIN_PARAMS[ITERATION]["base"] )

# Define Callbacks
callbacks = []

# Checkpoint - Save best weights
checkpoint = ModelCheckpoint( 
        os.path.join('saved_models', model.name ), 
        # monitor = 'val_mean_io_u',
        monitor = 'val_' + METRIC_CHOSEN, 
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

if MODEL_CHOOSEN == "fcn":
    epochs = FCN_TRAIN_PARAMS[ITERATION]["epochs"]
elif MODEL_CHOOSEN == "unet":
    epochs = UNET_TRAIN_PARAMS[ITERATION]["epochs"]
elif MODEL_CHOOSEN == "resunet":
    epochs = RESUNET_TRAIN_PARAMS[ITERATION]["epochs"]
    
# Train model
history = model.fit( trainDataGenerator,
                    validation_data  = validDataGenerator,
                    steps_per_epoch  = len(trainDataGenerator),
                    #validation_steps = len(validDataGenerator),                       
                    epochs           = epochs, 
                    verbose          = 1, 
                    callbacks        = [checkpoint, tensorboardCallback] )

plot_training_history(history, metric_name=METRIC_CHOSEN )