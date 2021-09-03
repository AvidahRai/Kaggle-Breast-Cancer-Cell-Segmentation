"""
    DataGenerator class inherited from tf.keras.utils.Sequence Class
    - Custom pipeline for Tensorflow Models
    - Supports online-augmentation
    
    Refs:
    - https://github.com/seth814/Semantic-Shapes/blob/master/data_generator.py
    - https://philippschw.github.io/masking/image_segmentation/computer_vision/medicare-drug-cost/
"""
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import imgaug

np.random.seed(66)
from albumentations import ( Compose, GaussianBlur, Affine , HorizontalFlip, Rotate, RandomBrightnessContrast )

augment_null = Compose([])
augment = Compose([ 
    GaussianBlur(blur_limit=(3, 7), sigma_limit=2, p=0.5),
    Affine(p=0.5, shear=180),
    HorizontalFlip( p=0.5),              
    Rotate(limit=45, p=0.3),
    RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6)
])

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
        masks_paths  = [self.masks_paths[i] for i in indexes]
                
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
        X = np.empty((self.batch_size, 
                      self.input_shape[0],
                      self.input_shape[1], 
                      self.input_shape[2]), 
                      dtype=np.float32)
        Y = np.empty((self.batch_size, 
                      self.input_shape[0],
                      self.input_shape[1], 
                      self.n_classes), 
                      dtype=np.float32)        
        
        for i, (image_path, mask_path) in enumerate(zip(images_paths, masks_paths)):
                        
            image_cv = self.__readImage(image_path)
            
            mask_cv = self.__readMask(mask_path)
            
            # Check to perform data augmentation            
            if self.augmentation == True:
                image_cv, mask_cv = self.__performAugmentation(image_cv, mask_cv)
            
            mask_cv = np.expand_dims(mask_cv, axis=-1)
            mask_cv = mask_cv / 255        
            
            assert (np.max(mask_cv)<= 1.0 and np.min(mask_cv) >= 0)
            mask_cv[mask_cv > 0.5] = 1
            mask_cv[mask_cv < 0.5] = 0
            
            mask_cv = mask_cv.astype('float32')
            
            # Store to batch
            X[i,] = image_cv
            Y[i,] = mask_cv
            
        return X, Y
    
    def __readImage(self, image_path):
        # Create Image Numpy array
        
        if self.input_shape[2] == 1:
            image_cv = cv2.imread(image_path, 0)
            image_cv = cv2.resize(image_cv, ( self.input_shape[1], self.input_shape[0] ) )
            image_cv = np.expand_dims(image_cv, axis=-1)
            image_cv = image_cv.astype(np.float32)
            
        elif self.input_shape[2] == 3:
            image_cv = cv2.imread(image_path, 1)
            image_cv = cv2.resize(image_cv, ( self.input_shape[1], self.input_shape[0] ), cv2.INTER_CUBIC )
                    
        return image_cv
        
    def __readMask(self, mask_path):
        # Create Mask Numpy array
        if self.n_classes == 1:
            mask_cv = cv2.imread( mask_path, cv2.IMREAD_GRAYSCALE )
            mask_cv = cv2.resize( mask_cv, ( self.input_shape[1], self.input_shape[0] ), cv2.INTER_AREA )
         
            # mask_cv = np.expand_dims(mask_cv, axis=2)
            
        elif self.n_classes > 1:
            # Create mutli masks
            pass 
                    
        return mask_cv          
        
    def __performAugmentation(self, image_cv, mask_cv ): 
        
        augmented = augment(image=image_cv, mask=mask_cv)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        

        
        return aug_img, aug_mask
    
    
    '''
        UNIT TEST METHODS
    ''' 
    def test__getSingleItem__(self,index):
        
        X, y = self.__transformData( self.images_paths[index:index+1], self.masks_paths[index:index+1] )
        
        return X, y 

    def test__getBatch__(self, index, batch_size ):
        self.batch_size = batch_size
        indexes = self.indexes[ index * self.batch_size:(index+1) * self.batch_size ]
        
        images_paths = [self.images_paths[i] for i in indexes]
        masks_paths = [self.masks_paths[i] for i in indexes]
        
        print("Current indexes:", indexes)
        print("Current image paths:", images_paths)
        print("Current mask paths:", masks_paths)
        
        X, y = self.__transformData(images_paths, masks_paths)
        
        return X, y, indexes