import tensorflow as tf
import loader.utils as utils
import numpy as np
from matplotlib import pyplot as plt
from models import net_vgg
# slim = tf.contrib.slim
from lib.setup import setup



anchor_scales = [8, 16, 32]

checkpoints_dir = 'vgg_16_2016_08_28/vgg16.ckpt'


class RPN:
    def __init__(self):
        self._batch_size = 1

        self.x = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 224, 224, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 4])
        # self.im_info = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 2])
        self.box = []
        self.class_num = 1
        # self.im_info = self.x.shape[1], self.x.shape[2]
        self.feat_stride = [16,]
        self.rois_ = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 4])
        self._anchor_targets = {}
        # self.img = cv2.imread('/home/food/Music/frcnn-tf/dataset/images/apple/apple_10.jpg')
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

    def vgg_16(self):
        num_anchors = 9
        
        conv1 = tf.layers.conv2d(self.x,
                                    filters=64,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_1")
        conv2 = tf.layers.conv2d(conv1,
                                    filters=64,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_2")
        pool1 = tf.layers.max_pooling2d(conv2,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            name="vgg/pool_1")
        conv3 = tf.layers.conv2d(pool1,
                                    filters=128,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_3")
        conv4 = tf.layers.conv2d(conv3,
                                    filters=128,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_4")

        pool2 = tf.layers.max_pooling2d(conv4,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            name="vgg/pool_2")

        conv5 = tf.layers.conv2d(pool2,
                                    filters=256,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_5")

        conv6 = tf.layers.conv2d(conv5,
                                    filters=256,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_6")
        conv7 = tf.layers.conv2d(conv6,
                                    filters=256,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_7")

        pool3 = tf.layers.max_pooling2d(conv7,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            name="vgg/pool_3")


        conv8 = tf.layers.conv2d(pool3,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_8")
        conv9 = tf.layers.conv2d(conv8,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_9")
        conv10 = tf.layers.conv2d(conv9,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_10")

        pool3 = tf.layers.max_pooling2d(conv10,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            name="vgg/pool_4")

        conv11 = tf.layers.conv2d(pool3,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_11")
        conv12 = tf.layers.conv2d(conv11,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_12")
        conv13 = tf.layers.conv2d(conv12,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name = "vgg/conv_13")

        rpn1 = tf.layers.conv2d(conv13,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    kernel_initializer ='normal' ,
                                    name='npn_conv/3x3')
        rpn_cls = tf.layers.conv2d(rpn1,
                                    filters= num_anchors * 2,
                                    kernel_size=(1, 1),
                                    activation='sigmoid',
                                    kernel_initializer='uniform',
                                    name="rpn_out_class")
        rpn_bbox = tf.layers.conv2d(rpn1,
                                    filters=num_anchors * 4,
                                    kernel_size=(1, 1),
                                    activation='linear',
                                    kernel_initializer='uniform',
                                    name='rpn_out_regre')
        # rpn_shape = rpn_cls.shape
        num = 2
        rpn_cls_ = self._reshape(rpn_cls, num, 'rpn_cls_scores_reshape')
        
        rpn_cls_score = self._softmax(rpn_cls_, 'rpn_cls_softmax')
        rpn_cls_prob = self._reshape(rpn_cls_score, num_anchors * 2, "rpn_cls_prob")
        # rpn_cls = tf.reshape(rpn_cls, [rpn_shape[0], rpn_shape[1]*rpn_shape[2], num_anchors, 2])
        
        # rpn_bbox = tf.reshape(rpn_bbox, [rpn_shape[0], rpn_shape[1]*rpn_shape[2], num_anchors, 4])
        

        return rpn_cls_prob, rpn_bbox, conv13



    def classification(fc7):

        classe = tf.layers.conv2d(fc7,
                                filters= 1,
                                kernel_size=(1, 1),
                                activation='sigmoid',
                                kernel_initializer='uniform',
                                name="rpn_out_class")
        classe = tf.nn.softmax(classe)
        
        return classe
    def regression(fc7):

        reg = tf.layers.conv2d(fc7,
                                filters=4,
                                kernel_size=(1, 1),
                                activation='linear',
                                kernel_initializer='uniform',
                                name='rpn_out_regre')
        return reg

    def flat(pooled):
        pool = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 7, 7, None])
        fc6 = tf.layers.conv2d(pooled, 4096, [7, 7], padding='VALID')
        fc7 = tf.layers.conv2d(fc6, 4096, [1, 1])
        label = classification(fc7)
        cored = regression(fc7)

        return label, cored

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

    def smooth_l1(x):
        l2 = 0.5 * (x**2.0)
        l1 = tf.abs(x) - 0.5

        condition = tf.less(tf.abs(x), 1.0)
        loss = tf.where(condition, l2, l1)
        return loss

    def losses(self, fg_inds, bg_inds, rpn_bbox, rpn_cls, box):
        elosion = 0.00001
        true_obj_loss = -tf.reduce_sum(tf.multiply(tf.log(rpn_cls+elosion), fg_inds))
        false_obj_loss = -tf.reduce_sum(tf.multiply(tf.log(rpn_cls+elosion), bg_inds))
        obj_loss = tf.add(true_obj_loss, false_obj_loss)
        cls_loss = tf.div(obj_loss, 16)

        bbox_loss = smooth_l1(tf.subtract(rpn_bbox, box))
        bbox_loss = tf.reduce_sum(tf.multiply(tf.reduce_sum(bbox_loss), fg_inds))
        bbox_loss = tf.multiply(tf.div(bbox_loss, 1197), 100)
        total_loss = tf.add(cls_loss, bbox_loss)

        return total_loss


    def frcnn(net):
        
        pooled = self._crop_pool_layer(net, self.rois_, self.class_num)
        label, cored = self.flat(pooled)

        return label, cored
        
    def getPlaceholders(self):
        return self.x, self._gt_boxes, self.rois_




