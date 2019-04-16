import numpy as np
from lib.generate_anchors import generate_anchors


def anchor_target_func(rpn_cls, gt_boxes, im_dims, feat_stride, achor_scales):
	allowed_border = 0
	im_dims = im_dims
	anchor_scales = np.array(anchor_scales)