import os
import cv2
import albumentations
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
def strong_aug(p=0.8):
	"""Find all the description of each function:https://github.com/albu/albumentations
	Probabilities:
	p1: decides if this augmentation will be applied. The most common case is p1=1 means that we always apply the transformations from above. p1=0 will mean that the transformation block will be ignored.
	p2: every augmentation has an option to be applied with some probability.
	p3: decide if OneOf will be applied.
	In the final run all the p1-p3 probabilities are multiplied.
	"""
	return Compose([
		ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.8, border_mode=cv2.BORDER_CONSTANT),
		OneOf([
			IAAAdditiveGaussianNoise(),
			GaussNoise(),
		 ], p=0.3),
		MedianBlur(blur_limit=3, p=0.7),
		OneOf([
			CLAHE(clip_limit=2, p=0.4),
			IAASharpen(p=0.4),
			IAAEmboss(p=0.4),
			RandomBrightnessContrast(p=0.6),
		HorizontalFlip(p=0.5)
		])], p=p)

