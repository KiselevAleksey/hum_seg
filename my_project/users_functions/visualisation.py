import matplotlib.pyplot as plt
import cv2

def pict_visualise (df, pict_num: int, path_to_img_dir:str, path_to_mask_dir:str):
    """Shows an image and its mask
    Parameters
    ----------
    df: DataFrame
    pict_num: int
    	Number of picture that a user like to see
    path_to_img_data : str
        Path to data with original images
    path_to_mask_data : str
    	Path to data with masks
    Returns
    -------
    img.jpg, mask.png"""

    image, image_mask = df.iloc[pict_num]
    img = cv2.imread(path_to_img_dir+'{}'.format(image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(path_to_mask_dir+'{}'.format(image_mask), 0)
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 8))
    axes[0].imshow(img)
    axes[1].imshow(mask)
    plt.show()
