import numpy as np
import loader.utils as utils
from lib.generate_anchors import generate_anchors
from lib.bbox_overlaps import bbox_overlaps
from lib.remove_extraboxes import remove_extraboxes


def create_Labels_For_Loss(gt_boxes, image_size=(224, 224), feature_shape=(14, 14), feat_stride=16, \
						   scales=np.array([8, 16, 32]), ratios=[0.5, 0.8, 1], \
						   ):
	"""This Function is processed before network input
	Number of Candicate Anchors is Feature Map width * heights
	Number of Predicted Anchors is Batch Num * Feature Map Width * Heights * 9
	"""
	width = feature_shape[0]
	height = feature_shape[1]
	batch_size = gt_boxes.shape[0]
	# shifts is the all candicate anchors(prediction of bounding boxes)
	center_x = np.arange(0, height) * feat_stride
	center_y = np.arange(0, width) * feat_stride
	center_x, center_y = np.meshgrid(center_x, center_y)
	# Shape is [Batch, Width*Height, 4]
	centers = np.zeros((batch_size, width*height, 4))
	centers[:] = np.vstack((center_x.ravel(), center_y.ravel(),
						center_x.ravel(), center_y.ravel())).transpose()
	A = scales.shape[0] * len(ratios)
	K = width * height # width * height
	anchors = np.zeros((batch_size, A, 4))
	anchors = generate_anchors(scales=scales, ratios=ratios) # Shape is [A, 4]
	candicate_anchors = centers.reshape(batch_size, K, 1, 4) + anchors # [Batch, K, A, 4]
	# shape is [B, K, A]
	is_inside = utils.batch_inside_image(candicate_anchors, image_size[1], image_size[0])
	# candicate_anchors: Shape is [Batch, K, A, 4]
	# gt_boxes: Shape is [Batch, G, 4]
	# true_index: Shape is [Batch, K, A]
	# false_index: Shape is [Batch, K, A]
	candicate_anchors, true_index, false_index = bbox_overlaps(
		np.ascontiguousarray(candicate_anchors, dtype=np.float),
		is_inside,
		gt_boxes)
	for i in range(batch_size):
		true_where = np.where(true_index[i] == 1)
		num_true = len(true_where[0])

		if num_true > 64:
			select = np.random.choice(num_true, num_true - 64, replace=False)
			num_true = 64
			batch = np.ones((select.shape[0]), dtype=np.int) * i
			true_where = remove_extraboxes(true_where[0], true_where[1], select, batch)
			true_index[true_where] = 0
		false_where = np.where(false_index[i] == 1)
		num_false = len(false_where[0])
		select = np.random.choice(num_false, num_false - (128-num_true), replace=False)
		batch = np.ones((select.shape[0]), dtype=np.int) * i
		false_where = remove_extraboxes(false_where[0], false_where[1], select, batch)
		false_index[false_where] = 0
	return candicate_anchors, true_index, false_index
