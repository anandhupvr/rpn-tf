import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np



def bbox_plot(img, box):

	fig, ax = plt.subplots(1)
	# import pdb; pdb.set_trace()
	ax.imshow(img)
	for i in range(len(box)):
		k = 0
		s = patches.Rectangle((box[i][k], box[i][k+1]), box[i][k+2], box[i][k+3], linewidth=1, edgecolor='g', facecolor="none")
		# s = patches.Rectangle((box[i][0], box[i][1]), box[i][2], box[i][3], linewidth=1, edgecolor='g', facecolor="none")
		ax.add_patch(s)
	plt.show()
	# plt.savefig("prediction.png")

def iou_distance(box_a, box_b):
    """
    iou距离
    :param box_a: [h,w]
    :param box_b: [h,w]
    :return:
    """
    if len(np.shape(box_a)) == 1:
        ha, wa = box_a[0], box_a[1]
        hb, wb = box_b[0], box_b[1]
        overlap = min(ha, hb) * min(wa, wb)
        iou = overlap / (ha * wa + hb * wb - overlap)
    else:
        ha, wa = box_a[:, 0], box_a[:, 1]
        hb, wb = box_b[:, 0], box_b[:, 1]
        overlap = np.minimum(ha, hb) * np.minimum(wa, wb)
        iou = overlap / (ha * wa + hb * wb - overlap)
    return 1. - iou


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


def compute_iou(ha, wa, hb, wb):
    """
    根据长宽计算iou
    :param ha: [n]
    :param wa: [n]
    :param hb: [m]
    :param wb: [m]
    :return:
    """
    # 扩维
    ha, wa = ha[:, np.newaxis], wa[:, np.newaxis]
    hb, wb = hb[np.newaxis, :], wb[np.newaxis, :]
    overlap = np.minimum(ha, hb) * np.minimum(wa, wb)  # [n,m]
    iou = overlap / (ha * wa + hb * wb - overlap)
    return iou


def analyze_anchors(gt_boxes, gt_labels, h, w):
    """
    分析anchor 长宽效果;
    :param gt_boxes: [n,(y1,x1,y2,x2)]
    :param gt_labels: [n]
    :param h: [m]
    :param w: [m]
    :return:
    """
    gt_h = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_w = gt_boxes[:, 3] - gt_boxes[:, 1]

    num_classes = np.max(gt_labels) + 1
    iou_dict = dict()
    for label in np.arange(1, num_classes):
        indices = np.where(gt_labels == label)
        iou = compute_iou(gt_h[indices], gt_w[indices], h, w)  # [boxes_num,anchors_num]
        iou_dict[label] = np.mean(np.max(iou, axis=1))

    return iou_dict








def compute_overlaps(boxes1, boxes2):
	'''Computes IoU overlaps between two sets of boxes.
	boxes1, boxes2: [N, (y1, x1, y2, x2)].
	'''
	# 1. Tile boxes2 and repeate boxes1. This allows us to compare
	# every boxes1 against every boxes2 without loops.
	# TF doesn't have an equivalent to np.repeate() so simulate it
	# using tf.tile() and tf.reshape.
	b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
							[1, 1, tf.shape(boxes2)[0]]), [-1, 4])
	b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
	# 2. Compute intersections
	b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
	b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
	y1 = tf.maximum(b1_y1, b2_y1)
	x1 = tf.maximum(b1_x1, b2_x1)
	y2 = tf.minimum(b1_y2, b2_y2)
	x2 = tf.minimum(b1_x2, b2_x2)
	intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
	# 3. Compute unions
	b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
	b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
	union = b1_area + b2_area - intersection
	# 4. Compute IoU and reshape to [boxes1, boxes2]
	iou = intersection / union
	overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
	return overlaps

def trim_zeros(boxes, name=None):
	'''
	Often boxes are represented with matrices of shape [N, 4] and
	are padded with zeros. This removes zero boxes.
	
	Args
	---
		boxes: [N, 4] matrix of boxes.
		non_zeros: [N] a 1D boolean mask identifying the rows to keep
	'''
	non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
	boxes = tf.boolean_mask(boxes, non_zeros, name=name)
	return boxes, non_zeros
