import numpy as np
import tensorflow as tf
from models import vgg
from models.rpn import RPN
from loss.losses import rpn_cls_loss, rpn_regress_loss
from lib.anchors import Anchor
# from models import resnet
# from models import fpn
from lib.target import RpnTarget
from keras.layers import Lambda


class network():
	def __init__(self):
		self._batch_size = 2
		self.net = RPN()
		# self.backbone = resnet.ResNet(depth=101, name='res_net')
		# self.neck = fpn.FPN(name='fpn')
		self.x = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 416, 416, 3], name="input_image")
		# self.im_size = tf.placeholder(dtype=tf.float32, shape=[None, None], name="img_shape")
		self.bbox = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 1, 4], name="input_bbox")
		self.class_id = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, 1], name="input_class")


	def build(self, config):
		vgg_16 = vgg.ConvNetVgg16('vgg16.npy')
		cnn = vgg_16.inference(self.x)
		features = vgg_16.get_features()
		rpn_class_logits, rpn_probs, rpn_deltas = self.net.rpn(features)
		
		anchors, anchors_tag = Anchor(config.RPN_ANCHOR_HEIGHTS,
										config.RPN_ANCHOR_WIDTHS,
										config.RPN_ANCHOR_BASE_SIZE,
										config.RPN_ANCHOR_RATIOS,
										config.RPN_ANCHOR_SCALES,
										config.BACKBONE_STRIDE, name='gen_anchors')(features)
		rpn_targets = RpnTarget(self._batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, name='rpn_target')(
								[self.bbox, self.class_id, anchors, anchors_tag])								
		deltas, cls_ids, anchor_indices = rpn_targets[:3]



		cls_loss = Lambda(lambda x: rpn_cls_loss(*x), name='rpn_class_loss')(
					[rpn_class_logits, cls_ids, anchor_indices])
		regress_loss = Lambda(lambda x: rpn_regress_loss(*x), name='rpn_bbox_loss')(
						[rpn_deltas, deltas, anchor_indices])

		return cls_loss, regress_loss

	def get_placeholder(self):
		return self.x, self.bbox, self.class_id


'''
	def build_network(self):
		# initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
		# initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
		# vgg_16 = vgg.ConvNetVgg16('vgg16.npy')
		# cnn = vgg_16.inference(self.x)
		# features = vgg_16.get_features()
		C2, C3, C4, C5 = self.backbone(self.x, training=True)

		P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5],training=True)
		import pdb; pdb.set_trace()
		rpn_class_logits, rpn_probs, rpn_deltas = self.net.rpn([P2, P3, P4, P5, P6])
		
		# proposals = self.net.get_proposals(rpn_probs, rpn_deltas, self.im_size)



		# rpn_cls_score, rpn_bbox_pred = self.build_model(features, initializer)
		# return [rpn_cls_score, rpn_bbox_pred, features], self.x
		return rpn_class_logits, rpn_probs, rpn_deltas

	def rpn_loss(self, rpn_class_logits, rpn_deltas):

		import pdb; pdb.set_trace()
		rpn_target_matchs, rpn_target_deltas = self.net.target(self.bbox)

		rpn_class_loss = losses.rpn_class_loss(rpn_target_matchs, rpn_class_logits)

		rpn_bbox_loss = losses.rpn_bbox_loss(rpn_target_deltas, rpn_target_matchs, rpn_deltas)

		return rpn_class_loss, rpn_bbox_loss



	def build_rpn(self, net, initializer):
		num_anchors = 9
		rpn1 = tf.layers.conv2d(net,
									filters=512,
									kernel_size=(3, 3),
									padding='same',
									kernel_initializer = initializer,
									name='npn_conv/3x3')
		rpn_cls_score = tf.layers.conv2d(rpn1,
									filters=num_anchors * 2,
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
		rpn_cls = tf.reshape(rpn_cls_score, [-1, 14, 14, 18], name='rpn_cls_pred')
		rpn_bbox = tf.reshape(rpn_bbox_pred, [-1, 14, 14, 36], name='rpn_bbox_pred')

		# num = 2
		# rpn_cls_score_reshape = self._reshape(rpn_cls_score, num, 'rpn_cls_scores_reshape')
		
		# rpn_cls_score_reshape = self._softmax(rpn_cls_score_reshape, 'rpn_cls_softmax')
		# rpn_cls_score_reshape = self._softmax(rpn_cls_score_reshape, 'rpn_cls_softmax')
		# rpn_cls_prob = self._reshape(rpn_cls_score, num_anchors , "rpn_cls_prob")

		return rpn_cls_score, rpn_bbox_pred

'''
