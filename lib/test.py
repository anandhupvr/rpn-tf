import loader.utils as utils
import numpy.random as npr
from loader import get_anchor




def result(net, rpn_cls, rpn_bbox, img, data):

    anchors = get_anchor.generate_anchors()
    import pdb; pdb.set_trace()
    num_anchors =  anchors.shape[0]
    width = int(np.shape(net)[1])
    height = int(np.shape(net)[2])
    img_width = int(img[0])
    img_height = int(img[1])

    num_feature_map = width * height

    # Calculate output w, h stride
    w_stride = img_width / width
    h_stride = img_height / height





    shift_x = np.arange(0, width) * w_stride
    shift_y = np.arange(0, height) * h_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel())).transpose()

    all_anchors = (anchors.reshape( (1, 9, 4)) +
                                    shifts.reshape( (1, num_feature_map, 4) ).transpose((1, 0, 2)) )


    total_anchors = num_feature_map * 9
    all_anchors = all_anchors.reshape((total_anchors, 4))
    # utils.bbox_plot(all_anchors)

    rpn_bbox =np.reshape(rpn_bbox, (-1, 4))
    rpn_cls = np.reshape(rpn_cls, (-1, 1))

    proposals = utils.bbox_transform_inv(all_anchors, rpn_bbox)

    proposals = utils.clip_boxes(proposals, (np.array([int(img[0]), int(img[1])], dtype='float32')))
    keep = utils.filter_boxes(proposals, 40)

    proposals = proposals[keep, :]
    scores = rpn_cls[keep]


    box = box.reshape(1, 4)

    pre_nms_topN = 6000
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    post_nms_topN = 300
    keep = utils.py_cpu_nms(np.hstack((proposals, scores)), 0.7)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    FG_FRAC=.25
    FG_THRESH=.5
    BG_THRESH_HI=.5
    BG_THRESH_LO=.1
    BATCH = 256
    proposals = np.vstack((proposals, box))

    overlaps = utils.bbox_overlaps(proposals, box)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)

    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]
    fg_rois_per_this_image = min(int(BATCH * FG_FRAC), fg_inds.size)

    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) &
                       (max_overlaps >= BG_THRESH_LO))[0]
    bg_rois_per_this_image = BATCH - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    # labels = labels[keep_inds]
    rois = proposals[keep_inds]
    gt_rois = box[gt_assignment[keep_inds]]


    return rois, fg_inds, bg_inds, rpn_cls, rpn_bbox, box