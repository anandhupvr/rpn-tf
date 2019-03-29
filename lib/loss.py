import tensorflow as tf
import numpy as np



def smooth_L1(x):
    l2 = 0.5 * (x**2.0)
    l1 = tf.abs(x) - 0.5

    condition = tf.less(tf.abs(x), 1.0)
    loss = tf.where(condition, l2, l1)
    return loss

def rpn_loss(rpn_cls, rpn_bbox):
    """Calculate Class Loss and Bounding Regression Loss.

    # Args:
        obj_class: Prediction of object class. Shape is [ROIs*Batch_Size, 2]
        bbox_regression: Prediction of bounding box. Shape is [ROIs*Batch_Size, 4]
    """
    rpn_shape = rpn_cls.get_shape().as_list()
    g_bbox = tf.placeholder(tf.float32, [rpn_shape[0], rpn_shape[1], rpn_shape[2], 4])
    true_index = tf.placeholder(tf.float32, [rpn_shape[0], rpn_shape[1], rpn_shape[2]])
    false_index = tf.placeholder(tf.float32, [rpn_shape[0], rpn_shape[1], rpn_shape[2]])
    elosion = 0.00001
    true_obj_loss = -tf.reduce_sum(tf.multiply(tf.log(rpn_cls[:, :, :, 0]+elosion), true_index))
    false_obj_loss = -tf.reduce_sum(tf.multiply(tf.log(rpn_cls[:, :, :, 1]+elosion), false_index))
    obj_loss = tf.add(true_obj_loss, false_obj_loss)
    cls_loss = tf.div(obj_loss, 16) # L(cls) / N(cls) N=batch size

    bbox_loss = smooth_L1(tf.subtract(rpn_bbox, g_bbox))
    bbox_loss = tf.reduce_sum(tf.multiply(tf.reduce_sum(bbox_loss, 3), true_index))
    bbox_loss = tf.multiply(tf.div(bbox_loss, 1197), 100) # rpn_shape[1]*rpn_shape[2]
    # bbox_loss = bbox_loss / rpn_shape[1]

    total_loss = tf.add(cls_loss, bbox_loss)
    return total_loss, cls_loss, bbox_loss, true_obj_loss, false_obj_loss, g_bbox, true_index, false_index
