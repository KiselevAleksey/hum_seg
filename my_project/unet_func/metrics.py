import keras
from keras import backend as K
import numpy as np

def get_iou_vector(A, B):
	# Numpy version

	batch_size = A.shape[0]
	metric = 0.0
	for batch in range(batch_size):
		t, p = A[batch], B[batch]
		true = np.sum(t)
		pred = np.sum(p)

		# deal with empty mask first
		if true == 0:
			metric += (pred == 0)
			continue

		# non empty mask case.  Union is never empty 
		# hence it is safe to divide by its number of pixels
		intersection = np.sum(t * p)
		union = true + pred - intersection
		iou = intersection / union

		# iou metrric is a stepwise approximation of the real iou over 0.5
		iou = np.floor(max(0, (iou - 0.45)*20)) / 10

		metric += iou

	# teake the average over all images in batch
	metric /= batch_size
	return metric


def my_iou_metric(y_true, y_pred):
	# Tensorflow version
	return K.tf.py_func(get_iou_vector, [y_true, y_pred > 0.5], K.tf.float64)