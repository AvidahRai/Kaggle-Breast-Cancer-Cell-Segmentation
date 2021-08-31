"""    
    DataGenerator class inherited from tf.keras.utils.Sequence Class
    Uses keras.preprocessing.image Module
    
    Refs 
    - https://keras.io/examples/vision/oxford_pets_image_segmentation/
    
"""
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img
import PIL

class DataGenerator(Sequence):
    
    # Constructor
    def __init__(self, 
                 input_shape, 
                 images_paths, 
                 masks_paths, 
                 batch_size   = 3, 
                 augmentation = False,
                 n_classes    = 1 
                ):
        self.input_shape  = input_shape
        self.input_size   = (input_shape[0], input_shape[1])
        self.images_paths = images_paths
        self.masks_paths  = masks_paths
        self.batch_size   = batch_size
        self.augmentation = augmentation
        self.n_classes    = n_classes
        
        self.hues = {'benign':90, 'malignant':0}
        self.on_epoch_end()
    
    # Return a single X and y set
    def __getitem__(self,index):
        
        indexes = self.indexes[ index * self.batch_size:(index+1) * self.batch_size ]
        
        images_paths = [self.images_paths[i] for i in indexes]
        masks_paths = [self.masks_paths[i] for i in indexes]
                
        X, y = self.__transformData(images_paths, masks_paths)
        
        return X, y

    # No of batches per epoch
    def __len__(self):
        return int ( np.ceil( len(self.images_paths) / float(self.batch_size) ))

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.images_paths))
    
    '''
        PRIVATE METHODS
    '''
    def __transformData(self, images_paths, masks_paths):
        
        # Empty Numpy arrays
        X = np.zeros((self.batch_size,) + self.input_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.input_size + (1,), dtype="float32")     
        
        for i, (image_path, mask_path) in enumerate(zip(images_paths, masks_paths)):
                        
            image_cv = load_img(image_path, target_size=self.input_size)    
                
            # Check to perform data augmentation            
            if self.augmentation == True:
                pass
            
            mask_cv = load_img(mask_path, target_size=self.input_size, color_mode="grayscale")
            mask_cv = np.expand_dims(mask_cv, 2)
            
            # Store to batch
            X[i] = image_cv
            y[i] = mask_cv
            # y[i] -= 1
            
        return X, y
            
    def __performAugmentation(self, image ): 
        pass
        
    '''
        UNIT TEST METHODS
    ''' 
    def test__getSingleItem__(self,index):
        
        X, y = self.__transformData( self.images_paths[index:index+1], self.masks_paths[index:index+1] )
        
        return X, y 

    def test__getBatch__(self, index, batch_size):
        
        self.batch_size = batch_size
        indexes = self.indexes[ index * self.batch_size:(index+1) * self.batch_size ]
        
        images_paths = [self.images_paths[i] for i in indexes]
        masks_paths = [self.masks_paths[i] for i in indexes]
        
        print("Current indexes:", indexes)
        print("Current image paths:", images_paths)
        print("Current mask paths:", masks_paths)
        
        X, y = self.__transformData(images_paths, masks_paths)
        
        return X, y, indexes