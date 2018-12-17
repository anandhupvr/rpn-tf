import numpy as np
import random
from loader import get_anchor
from loader.utils import bbox_overlaps
from lib.bbox_transform import bbox_transform


def anchor_target_layer_python(rpn_cls_score, _gt_boxes, im_dims, feat_stride):


    allowded_border = 0
    # im_dims = im_dims[0]
    _gt_boxes = _gt_boxes[:, :-1]
    anchor_ratios=(0.5, 1, 2)
    anchor_scales=(8, 16, 32)
    anchors = get_anchor.generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    num_anchors = anchors.shape[0]

    height = rpn_cls_score.shape[1]
    width = rpn_cls_score.shape[2]


    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid( shift_x, shift_y )
    shifts = np.vstack( ( shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel() ) )
    shifts = shifts.transpose()

    K = shifts.shape[0]
    b = anchors.reshape((1, num_anchors, 4))
    c = shifts.reshape((1, K, 4)).transpose(1, 0, 2)
    all_anchors = b + c
    all_anchors = all_anchors.reshape( ( K * num_anchors, 4) )
    total_anchors = int( K * num_anchors )

    if im_dims is not None:
        inds_inside = np.where( 
                            ( all_anchors[:,0] >= 0 ) & 
                            ( all_anchors[:,1] >= 0 ) & 
                            ( all_anchors[:,2] <  im_dims[1] +0 ) &   # width
                            ( all_anchors[:,3] <  im_dims[0])+0)[0]   # take the row index

    anchors = all_anchors[inds_inside, :]
    labels = np.empty( (len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    overlaps = bbox_overlaps(
                            np.ascontiguousarray(anchors, dtype = np.float), 
                            np.ascontiguousarray(_gt_boxes, dtype = np.float)       
                            )  # ( #inds_inside x gt_boxes.shape[0])

    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_argmax_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_argmax_overlaps)[0]
    labels[max_overlaps < 0.3] = 0
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps > 0.7] = 1

    num_fg = int(0.5 * 256)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    num_bg = 256 - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # bbox targets : the deltas (relative to anchors) that faster R-CNN should try to predict at each anchor
    bbox_targets = np.zeros( (len(inds_inside), 4), dtype = np.float32 )
    bbox_targets = compute_target( anchors, _gt_boxes[argmax_overlaps,:])

    # it is used for p*_{i} in cost function (equation 1): Inside weights is for specifying anchors or rois with positive labe
    bbox_inside_weights = np.zeros( (len(inds_inside), 4), dtype = np.float32 )
    bbox_inside_weights[labels == 1] = np.array((1.0, 1.0, 1.0, 1.0)) 
    bbox_outside_weights  = np.zeros( (len(inds_inside), 4), dtype = np.float32)
    
    # uniform weight per sample : Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1,4)) * 1.0/num_examples 
    negative_weights = np.ones((1,4)) * 1.0/num_examples

    bbox_outside_weights[labels == 1,: ] = positive_weights
    bbox_outside_weights[labels == 0,: ] = negative_weights

    # map to oriignal set of anchors
    labels = unmap(labels, total_anchors, inds_inside, fill = -1)
    bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill = 0 )
    bbox_inside_weights = unmap(bbox_inside_weights, total_anchors, inds_inside, fill = 0 )
    bbox_outside_weights = unmap(bbox_outside_weights,total_anchors, inds_inside, fill = 0 )
    # labels
    labels = labels.reshape((1, height, width, num_anchors))
    labels = labels.reshape((1,1, num_anchors * height* width))
    rpn_labels = labels
    rpn_bbox_targets = bbox_targets.reshape((1, height, width, num_anchors * 4))
    rpn_bbox_inside_weights = bbox_inside_weights.reshape ((1, height, width, num_anchors * 4))
    rpn_bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, num_anchors * 4))
    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights

def unmap( data, count, inds, fill = 0):
    """
    Unmap a subset of item (data) back to the original set of items (of size count)
    """
    if len(data.shape) == 1 :
        ret = np.empty( (count,), dtype= np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret         = np.empty( (count, ) + data.shape[1:], dtype = np.float32)
        ret.fill(fill)
        ret[inds,:] = data

    return ret

def compute_target(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    return bbox_transform(ex_rois, gt_rois[:,:4].astype(np.float32, copy = False) )