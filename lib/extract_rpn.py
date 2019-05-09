import numpy as np
import tensorflow as tf
from lib.generate_anchors import generate_anchors
# from lib.utils import bbox_transform_inv, clip_boxes, non_max_suppression_fast



def extractor(rpn_bbox, rpn_cls, im_dims=(416, 416), scales=np.array([8, 16, 32]), ratios=[0.5, 0.8, 1]):
	anchors = generate_anchors(scales=scales, ratios=ratios)
	num_anchors = anchors.shape[0]
	_feat_stride = 16
	width, height = 26, 26
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
	proposals = bbox_transform_inv(anchors, rpn_bbox)
	order = rpn_cls[:, [0]].ravel().argsort()
	proposals = proposals[order]
	proposals = clip_boxes(proposals, im_dims)
	box = non_max_suppression_fast(proposals, 0.8)
	box = box[:50]
	# keep = _filter_boxes(proposals, 16)
	# proposals = proposals[keep, :]
	# scores = rpn_cls[0][:, 1:]
	# scores = scores[keep, :]
	# order = scores.ravel().argsort()[::-1]
	# postNMS = 100
	# if postNMS > 0:
	# 	order = order[:postNMS]
	# proposals = proposals[order, :]
	# scores = scores[order]

	# return proposals, scores, box
	return box




def _filter_boxes(boxes, min_size):
	"""Remove all boxes with any side smaller than min_size."""
	ws = boxes[:, 2] - boxes[:, 0] + 1
	hs = boxes[:, 3] - boxes[:, 1] + 1
	keep = np.where((ws >= min_size) & (hs >= min_size))[0]
	return keep

from lib.anchors import Anchor


def bbox_transform_inv(boxes, deltas):
    '''
    Applies deltas to box coordinates to obtain new boxes, as described by 
    deltas
    '''   
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # # x2
    # pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # # y2
    # pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_w
    # y2
    pred_boxes[:, 3::4] = pred_h

    return pred_boxes

def inverse(regr, config, features):
	features = tf.expand_dims(features, axis=0)
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
	import pdb; pdb.set_trace()

	pred_boxes = np.zeros(regr.shape, dtype=regr.dtype)

	pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
	pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
	pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
	pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

	return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")