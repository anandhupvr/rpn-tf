import numpy as np
import tensorflow as tf
from models import vgg



class network():
    def __init__(self, batch_size=1):
        self._batch_size = None

        self.x = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, None, None, 3], name="input_image")
        self.cls_plc = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 18], name="rpn_cls")
        self.box_plc = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 72], name="rpn_box")

    def build_network(self):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        vgg_16 = vgg.ConvNetVgg16('vgg16.npy')
        cnn = vgg_16.inference(self.x)
        features = vgg_16.get_features()


        rpn_cls_score, rpn_bbox_pred = self.build_rpn(features, initializer)
        return [rpn_cls_score, rpn_bbox_pred, features]


    def build_rpn(self, net, initializer):
        num_anchors = 9
        rpn1 = tf.layers.conv2d(net,
                                    filters=512,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    kernel_initializer = initializer,
                                    name='npn_conv/3x3')
        rpn_cls_score = tf.layers.conv2d(rpn1,
                                    filters=num_anchors,
                                    kernel_size=(1, 1),
                                    activation='sigmoid',
                                    kernel_initializer = initializer,
                                    name="rpn_out_class")
        rpn_bbox_pred = tf.layers.conv2d(rpn1,
                                    filters=num_anchors * 4,
                                    kernel_size=(1, 1),
                                    activation='linear',
                                    kernel_initializer = initializer,
                                    name='rpn_out_regre')
        # rpn_cls = tf.reshape(rpn_cls_score, [4, 14, 14, 9], name='rpn_cls_pred')
        # rpn_bbox = tf.reshape(rpn_bbox_pred, [4, 14, 14, 36], name='rpn_bbox_pred')

        # num = 2
        # rpn_cls_score_reshape = self._reshape(rpn_cls_score, num, 'rpn_cls_scores_reshape')
        
        # rpn_cls_score_reshape = self._softmax(rpn_cls_score_reshape, 'rpn_cls_softmax')
        # rpn_cls_score_reshape = self._softmax(rpn_cls_score_reshape, 'rpn_cls_softmax')
        # rpn_cls_prob = self._reshape(rpn_cls_score, num_anchors , "rpn_cls_prob")

        return rpn_cls_score, rpn_bbox_pred

    def get_placeholder(self):
        return self.x, self.cls_plc, self.box_plc