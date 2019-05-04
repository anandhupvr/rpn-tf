import numpy as np
# from lib.generate_anchors import generate_anchors
# from lib.utils import bbox_transform_inv, clip_boxes, non_max_suppression_fast



def extractor(rpn_bbox, rpn_cls, im_dims=(224, 224), scales=np.array([8, 16, 32]), ratios=[0.5, 0.8, 1]):
	anchors = generate_anchors(scales=scales, ratios=ratios)
	num_anchors = anchors.shape[0]
	_feat_stride = 16
	width, height = 14, 14
	shift_x = np.arange(0, width) * _feat_stride
	shift_y = np.arange(0, height) * _feat_stride
	shift_x, shift_y = np.meshgrid(shift_x, shift_y)
	shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
						shift_x.ravel(), shift_y.ravel())).transpose()
	A = num_anchors
	K = shifts.shape[0]
	anchors = anchors.reshape((1, A, 4)) + \
				shifts.reshape((1, K, 4)).transpose((1, 0, 2))
	anchors = anchors.reshape((K*A, 4))
	proposals = bbox_transform_inv(anchors, rpn_bbox[0])
	proposals = clip_boxes(proposals, im_dims)
	keep = _filter_boxes(proposals, 16)
	proposals = proposals[keep, :]
	scores = rpn_cls[0][:, 1:]
	scores = scores[keep, :]
	order = scores.ravel().argsort()[::-1]
	postNMS = 100
	if postNMS > 0:
		order = order[:postNMS]
	proposals = proposals[order, :]
	scores = scores[order]
	box = non_max_suppression_fast(proposals, 0.5)
	return proposals, scores, box




def _filter_boxes(boxes, min_size):
	"""Remove all boxes with any side smaller than min_size."""
	ws = boxes[:, 2] - boxes[:, 0] + 1
	hs = boxes[:, 3] - boxes[:, 1] + 1
	keep = np.where((ws >= min_size) & (hs >= min_size))[0]
	return keep

from lib.anchors import Anchor


def inverse(regr, config, features):
	anchors, anchors_tag = Anchor(config.RPN_ANCHOR_HEIGHTS,
								config.RPN_ANCHOR_WIDTHS,
								config.RPN_ANCHOR_BASE_SIZE,
								config.RPN_ANCHOR_RATIOS,
								config.RPN_ANCHOR_SCALES,
								config.BACKBONE_STRIDE, name='gen_anchors')(features)


	valid_anchor_indices = tf.where(anchors_tag)[:, 0]
	anchors = tf.gather(anchors, valid_anchor_indices)
	height = anchors[:, 2] - anchors[:, 0] + 1.0
	width = anchors[:, 3] - anchors[:, 1] + 1.0
	ctr_x = anchors[:, 1] * 0.5 * width
	ctr_y = anchors[:, 0] * 0.5 * height

	dy = regr[:, 0::4]
	dx = regr[:, 1::4]
	dh = regr[:, 2::4]
	dw = regr[:, 3::4]

	pred_ctr_x = dx * width[:, tf.newaxis] + ctr_x[:, tf.newaxis]
	pred_ctr_y = dy * height[:, tf.newaxis] + ctr_y[:, tf.newaxis]
	pred_w = tf.exp(dw) * width[:, tf.newaxis]
	pred_h = tf.exp(dh) * height[:, tf.newaxis]

	pred_boxes = tf.zeros(regr.shape, dtype=regr.dtype)

	pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
	pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
	pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
	pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

	return pred_boxes