import os
import numpy as np
import pandas as pd

def create_train_df (path_to_img_dir:str, path_to_mask_dir:str):
	"""Create DF with names of train images and masks 
	----------
	path_to_img_data : str
	    Path to data with original images
	path_to_mask_data : str
		Path to data with masks
	Returns
	-------
	DataFrame"""
	train_image = []
	train_image_mask = []
	for img in os.listdir(path_to_img_dir):
		train_image.append(img)
	for img_mask in os.listdir(path_to_mask_dir):
		train_image_mask.append(img_mask)
	d_tr = {'train_image': train_image, 'train_image_mask': train_image_mask}
	train_df = pd.DataFrame(data=d_tr)
	return train_df

def create_valid_df (path_to_img_dir:str, path_to_mask_dir:str):
	"""Create DF with names of train images and masks 
	----------
	path_to_img_data : str
	    Path to data with original images
	path_to_mask_data : str
		Path to data with masks
	Returns
	-------
	DataFrame"""
	valid_image = []
	valid_image_mask = []
	for img in os.listdir(path_to_img_dir):
	    valid_image.append(img)
	for img_mask in os.listdir(path_to_mask_dir):
	    valid_image_mask.append(img_mask)
	d_val = {'valid_image': valid_image, 'valid_image_mask': valid_image_mask}
	valid_df = pd.DataFrame(data=d_val)
	return valid_df