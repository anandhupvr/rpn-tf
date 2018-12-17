import numpy as np
import tensorflow as tf
from lib.anchor_pre import generate_anchors_pre
from lib.proposal_layer import proposal_layer
from lib.anchor_target import anchor_target_layer
from lib.proposal_target_layer import proposal_target_layer
import numpy.random as npr




class arrange:
    def __init__(self):
        self._feat_stride = [16.]

    def _softmax(self, rpn_cls, name):
        if name == 'rpn_cls_softmax':
            shape = tf.shape(rpn_cls)
            reshape_ = tf.reshape(rpn_cls, [-1, shape[-1]])
            reshaped_score = tf.nn.softmax(reshape_, name=name)
            return tf.reshape(reshaped_score, shape)

    def _reshape(self, rpn_cls, num, name):
        with tf.variable_scope(name):
            to_caffe = tf.transpose(rpn_cls, [0, 3, 1, 2])
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[self._batch_size], [num, -1], [tf.shape(rpn_cls)[2]]]))
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _anchor_component(self, rpn_cls):
        import pdb; pdb.set_trace()
        anchors, length, img_info = tf.py_func(generate_anchors_pre,
                                    [rpn_cls, self.feat_stride],
                                    [tf.float32, tf.int32, tf.float32], name="generate_anchors")
        
        anchors.set_shape([None, 4])
        length.set_shape([])
        img_info.set_shape([None, None])
        # self.anchors = anchors
        # self.length = length
        # self.img_info = img_info
        # self.rpn_cls = rpn_cls
        return anchors, length, img_info

    def proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):

        rois, rpn_scores = tf.py_func(proposal_layer,
                                        [rpn_cls_prob, rpn_bbox_pred, self.img_info, self.anchors, self.length],
                                        [tf.float32, tf.float32])

        rois.set_shape([None, 4])
        rpn_scores.set_shape([None, 1])
        return rois, rpn_scores

    def anchor_targets(self, rpn_cls_score):


        num = 9

        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(anchor_target_layer,
                                                                [rpn_cls_score, self._gt_boxes, self.img_info, img, self._feat_stride, _anchors, self.length],
                                                                [tf.float32, tf.float32, tf.float32, tf.float32])
        rpn_labels.set_shape([1, 1, None, None])
        # rpn_labels.set_shape([None])
        rpn_bbox_targets.set_shape([1, None, None, 9 * 4])
        rpn_bbox_inside_weights.set_shape([1, None, None, 9 * 4])
        rpn_bbox_outside_weights.set_shape([1, None, None, 9 * 4])

        rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
        self._anchor_targets['rpn_labels'] = rpn_labels
        self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights


        return rpn_labels

    def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        import pdb; pdb.set_trace()
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _proposal_target_layer(self,rois, roi_scores):
        rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(proposal_target_layer,
                                                                                            [rois, roi_scores, self._gt_boxes, self.class_num],
                                                                                            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        rois.set_shape([1, 4])
        roi_scores.set_shape([1])
        labels.set_shape([1, 1])
        bbox_targets.set_shape([1, 1 * 4])
        bbox_inside_weights.set_shape([1, 1 * 4])
        bbox_outside_weights.set_shape([1, 1 * 4])

        self._proposal_targets['rois'] = rois
        self._proposal_targets['labels'] = tf.to_int32(labels, name='to_int32')
        self._proposal_targets['bbox_targets'] = bbox_targets
        self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
        self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights


        return rois, roi_scores

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be backpropagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def losses(self, bbox_pred, cls_score):
        rpn_cls_score = tf.reshape(self.rpn_cls, [-1, 2])
        rpn_label_los = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
        rpn_select = tf.where(tf.not_equal(rpn_label_los, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_label_los = tf.reshape(tf.gather(rpn_label_los, rpn_select), [-1])
        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label_los))

        # RPN , bbox loss
        rpn_bbox_pred = self.rpn_bbox_pred
        rpn_bbox_targets_los = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights_los = self._anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights_los = self._anchor_targets['rpn_bbox_outside_weights']

        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets_los, rpn_bbox_inside_weights_los,
                            rpn_bbox_outside_weights_los, sigma=3.0, dim=[1, 2, 3])
        # loss = losses(fg_inds, bg_inds, rpn_bbox, rpn_cls, box)
        # RCNN , class loss
        # cls_score = self.rpn_cls
        label = tf.reshape(self._proposal_targets['labels'], [-1])

        cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, class_num]), labels=label))
        # RCNN , bbox loss
        # bbox_pred = self.rpn_bbox_pred
        bbox_targets = self._proposal_targets['bbox_targets']
        bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
        bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

        loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)


        self._losses['cross_entropy'] = cross_entropy
        self._losses['loss_box'] = loss_box
        self._losses['rpn_cross_entropy'] = rpn_cross_entropy
        self._losses['rpn_loss_box'] = rpn_loss_box
        self.loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        
        return self.loss