import os
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import tensorflow as tf




def box_plot(img, box):
    # a = (box[0][:].split(","))
    # print (a)

    fig, ax = plt.subplots(1)
    im = np.array(Image.open(img),dtype=np.uint8)
    ax.imshow(im)
    for i in range(3):
        k = 0
        s = patches.Rectangle((box[i][k+1], box[i][k+2]), box[i][k+3] - box[i][k+1], box[i][k+2] - box[i][k+2], linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(s)
    plt.show()


def bbox_transform_inv(boxes, regr):

    if boxes.shape[0] == 0:
        return np.zeros((0, regr.shape[1]), dtype=regr.dtype)

    boxes = boxes.astype('float32', copy=False)
    width = boxes[:, 2] - boxes[:, 0] + 1.0
    height = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * width
    ctr_y = boxes[:, 1] + 0.5 * height

    # dx = regr[:, 0::4]
    # dy = regr[:, 1::4]
    # dw = regr[:, 2::4]
    # dh = regr[:, 3::4]

    dy = regr[:, 0::4]
    dx = regr[:, 1::4]
    dh = regr[:, 2::4]
    dw = regr[:, 3::4]


    pred_ctr_x = dx * width[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * height[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * width[:, np.newaxis]
    pred_h = np.exp(dh) * height[:, np.newaxis]

    pred_boxes = np.zeros(regr.shape, dtype=regr.dtype)

    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w

    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    # pred_boxes = tf.zeros([regr.shape[0],regr.shape[1]])
    # pred_boxes[:, 0::4].assign(pred_ctr_x - 0.5 * pred_w)
    # pred_boxes[:, 1::4].assign(pred_ctr_y - 0.5 * pred_h)
    # pred_boxes[:, 2::4].assign(pred_ctr_x + 0.5 * pred_w)
    # pred_boxes[:, 3::4].assign(pred_ctr_y + 0.5 * pred_h)

    return pred_boxes
def clip_boxes(boxes, im_shape):
    # im_shape = tf.convert_to_tensor(im_shape, dtype=object)
    # print (type(boxes[:, 0::4]))
    # print (type(im_shape[1]))

    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)

    # boxes[:, 0::4].assign(tf.maximum(tf.minimum(boxes[:, 0::4], im_shape[1] - 1), 0))
    # boxes[:, 1::4].assign(tf.maximum(tf.minimum(boxes[:, 1::4], im_shape[0] - 1), 0))
    # boxes[:, 2::4].assign(tf.maximum(tf.minimum(boxes[:, 2::4], im_shape[1] - 1), 0))
    # boxes[:, 3::4].assign(tf.maximum(tf.minimum(boxes[:, 3::4], im_shape[0] - 1), 0))


    return boxes
def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = tf.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[0]
    y1 = dets[1]
    x2 = dets[2]
    y2 = dets[3]
    scores = dets[4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order = scores.argsort()[::-1]
    order = tf.contrib.framework.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = tf.maximum(x1[i], x1[order[1:]])
        yy1 = tf.maximum(y1[i], y1[order[1:]])
        xx2 = tf.minimum(x2[i], x2[order[1:]])
        yy2 = tf.minimum(y2[i], y2[order[1:]])

        w = tf.maximum(0.0, xx2 - xx1 + 1)
        h = tf.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = tf.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes = boxes.astype(int)
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)

                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = (iw * ih / ua)

    return overlaps

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))

    targets = np.transpose(targets)

    return targets

def bbox_plot(img, box):
    im = np.array(Image.open(img),dtype=np.uint8)
    fig, ax = plt.subplots(1)
    # import pdb; pdb.set_trace()
    ax.imshow(im)
    for i in range(len(box)):
        k = 0
        s = patches.Rectangle((box[i][k], box[i][k+1]), box[i][k+2], box[i][k+3], linewidth=1, edgecolor='g', facecolor="none")
        # s = patches.Rectangle((box[i][0], box[i][1]), box[i][2], box[i][3], linewidth=1, edgecolor='g', facecolor="none")
        ax.add_patch(s)
    plt.show()
    
def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def bbox_transform_inv_tf(boxes, deltas):
    boxes = tf.cast(boxes, deltas.dtype)
    widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
    heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
    ctr_x = tf.add(boxes[:, 0], widths * 0.5)
    ctr_y = tf.add(boxes[:, 1], heights * 0.5)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)

    pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
    pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
    pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)

def clip_boxes_tf(boxes, im_info):
    b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou



def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3     

def compute_iou(box1, box2):
    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box2[2]-box2[0], box2[3]-box2[0]
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union


def batch_inside_image(anchors, width, height):
    # [b, k, A]
    B = anchors.shape[0]
    K = anchors.shape[1]
    A = anchors.shape[2]
    inds_inside = np.zeros((B, K, A), dtype=np.int32)
    for b in range(B):
        for k in range(K):
            for a in range(A):
                if anchors[b, k, a, 0] >= 0:
                    if anchors[b, k, a, 1] >= 0:
                        if anchors[b, k, a, 2] < width:
                            if anchors[b, k, a, 3] < height:
                                inds_inside[b, k, a] = 1


    return inds_inside



