import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, MaxPooling2D, Concatenate, Activation
from keras import backend as K

def Unet_ResNet50 ():
	"""Concatinate some different layers of the same shape from the ResNet model 
	and returns unput and output of a new combine model"""
	# Выбор предобученной модели из библиотеки
	base_model = ResNet50(weights='imagenet', input_shape=(256,256,3), include_top=False)
	base_out = base_model.output

	# Строим U-Net связи в модели:

	conv_1 = base_model.get_layer('activation_1').output
	conv_2 = base_model.get_layer('activation_10').output
	conv_3 = base_model.get_layer('activation_22').output
	conv_4 = base_model.get_layer('activation_40').output
	conv_5 = base_model.get_layer('activation_49').output

	up1 = UpSampling2D(2, interpolation='bilinear')(conv_5)
	conc_1 = Concatenate()([up1, conv_4])
	conv_6 = Conv2D(256, (3, 3), padding='same')(conc_1)
	conv_6 = Activation('relu')(conv_6)

	up2 = UpSampling2D(2, interpolation='bilinear')(conv_6)
	conc_2 = Concatenate()([up2, conv_3])
	conv_7 = Conv2D(128, (3, 3), padding='same')(conc_2)
	conv_7 = Activation('relu')(conv_7)

	up3 = UpSampling2D(2, interpolation='bilinear')(conv_7)
	conc_3 = Concatenate()([up3, conv_2])
	conv_8 = Conv2D(64, (3, 3), padding='same')(conc_3)
	conv_8 = Activation('relu')(conv_8)

	up4 = UpSampling2D(2, interpolation='bilinear')(conv_8)
	conc_4 = Concatenate()([up4, conv_1])
	conv_9 = Conv2D(32, (3, 3), padding='same')(conc_4)
	conv_9 = Activation('relu')(conv_9)

	up5 = UpSampling2D(2, interpolation='bilinear')(conv_9)
	conv_10 = Conv2D(1, (3, 3), padding='same')(up5)
	conv_10 = Activation('sigmoid')(conv_10)

	model = Model(input=base_model.input, output=conv_10)
	
	return model