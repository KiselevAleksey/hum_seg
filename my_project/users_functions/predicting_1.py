import os
import cv2
import numpy as np
import pandas as pd
from lib.utils import encode_rle, decode_rle

def import_data (path_to_img_dir:str, path_to_mask_dir:str, path_to_test_dir:str):
    """Colects validation and test data for prognosing and visualization.
    Parameters
    ----------
    path_to_img_dir : str
        Path to data with original val images
    path_to_mask_dir : str
        Path to data with val masks
    path_to_test_dir : str
        Path to data with test images 
    Returns
    -------
    list_of_img,
    list_of_mask,
    val_list,
    tests,
    test_list"""


    val = sorted(os.listdir(path_to_img_dir))
    val_mask = sorted(os.listdir(path_to_mask_dir))
    test_nums = sorted(os.listdir(path_to_test_dir))

    list_of_img = []
    list_of_mask = []
    for i in val:
        img = cv2.imread(path_to_img_dir+'{}'.format(i))
        img = cv2.resize(img, (256, 256))
        list_of_img.append(img / 255.)
    val_list = np.array(list_of_img)

    for i in val_mask:
        img = cv2.imread(path_to_mask_dir+'{}'.format(i), 0)
        np.expand_dims(img, -1)
        list_of_mask.append(img)

    tests = []
    for i in test_nums:
        img = cv2.imread(path_to_test_dir+'{}'.format(i))
        img = cv2.resize(img, (256, 256))
        tests.append(img / 255.)
    test_list = np.array(tests)

    yield list_of_img, list_of_mask, val_list, tests, test_list


def save_results(pict_num:int, path_save_results:str, pred_val_list:list):
    """Creates a new csv file with predictions in a certain dir.
    Parameters
    ----------
    pict_num:int
    	A number of the first image. It goes to col. 'id'
    path_save_results : str
    	A name of dir to which results should be saved. 
    pred_val_list:list
    	list of predicted values
    Returns:
    csv file
    ------- """
    im_num_counter = pict_num
    rle_mask = []
    im_id = []
    for i in pred_val_list:
        mask = encode_rle(i)
        rle_mask.append(mask)
        im_id.append(im_num_counter)
        im_num_counter+=1
        d_pred = {'id': im_id, 'rle_mask': rle_mask}
    pred_valid_template = pd.DataFrame(data=d_pred)
    pred_valid_template.to_csv(path_save_results, index=False)    