import numpy as np
import cv2
from users_functions.augmentation import strong_aug


def train_generator(gen_df, batch_size):
    """The function save img_batch and nask_batch from DF.
    That allows not to keep all the pictures in RAM, but yield those step by step
        Parameters
    ----------
    gen_df : Data Frame
        batches takes pictures 
    path_to_mask_data : str
        Path to data with masks
    Returns
    -------
    list_of_img, list_of_mask"""
    
    while True:
        x_batch = []
        y_batch = []
        
        for i in range (batch_size):
            try:
                img_name, img_mask = gen_df.sample().values[0]
                img = cv2.imread('data/train/{}'.format(img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread('data/train_mask/{}'.format(img_mask), 0)
     
            except Exception as e:
                pass
#         меняем размер картинок
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            
#         добавим аугментации
            augmentation = strong_aug(p=0.9)
            data = {"image": img.astype('uint8'), "mask": mask}
            augmented = augmentation(**data)
            img, mask = augmented["image"], augmented["mask"]

#            добавляем картинки в батч 
            x_batch += [img]
            y_batch += [mask]

#         нормируем картинки
        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch) / 255.
        
        yield x_batch, np.expand_dims(y_batch, -1)


def valid_generator (gen_df, batch_size):
    """The function save img_batch and nask_batch from DF.
    That allows not to keep all the pictures in RAM, but yield those step by step
        Parameters
    ----------
    gen_df : Data Frame
        batches takes pictures 
    path_to_mask_data : str
        Path to data with masks
    Returns
    -------
    list_of_img, list_of_mask"""
    while True:
        x_val_batch = []
        y_val_batch = []
        
        for i in range (batch_size):
            try:
                img_name, img_mask = gen_df.sample(1).values[0]
                img = cv2.imread('data/valid/{}'.format(img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread('data/valid_mask/{}'.format(img_mask), 0)
            
            except Exception as e:
                pass
#         меняем размер картинок
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))

#            добавляем картинки в батч 
            x_val_batch += [img]
            y_val_batch += [mask]
            
#         нормируем картинки
        x_val_batch = np.array(x_val_batch) / 255.
        y_val_batch = np.array(y_val_batch) / 255.
        
        yield x_val_batch, np.expand_dims(y_val_batch, -1)