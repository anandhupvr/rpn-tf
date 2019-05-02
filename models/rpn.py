import tensorflow as tf
import numpy as np
from lib.anchor_generator import AnchorGenerator
from lib import transform, anchor_target


class RPN:

	def __init__(self):

		self.anchor_scales=(16)
		self.anchor_ratios=(0.5, 1, 2)
		self.anchor_feature_stride = 14

		self.rpn_target_means = (0., 0., 0., 0.)
		self.rpn_target_stds = (0.1, 0.1, 0.2, 0.2)

		self.num_rpn_deltas = 256
		self.rpn_positive_fraction = 0.5
		self.rpn_pos_iou_thr = 0.7
		self.rpn_neg_iou_thr = 0.3

		self.proposal_count=2000
		self.nms_threshold=0.7

		self.generator = AnchorGenerator()
		self.anchor_target = anchor_target.AnchorTarget()

	def rpn(self, features):
		# layers_outputs = []
		# for feat in features:
		rpn_conv_shared_ = tf.layers.conv2d(features, 512, (3, 3), padding='same',
							kernel_initializer='he_normal',
							name='rpn_conv_shared',
							reuse=tf.AUTO_REUSE)

		shared = tf.nn.relu(rpn_conv_shared_)

		x = tf.layers.conv2d(shared, 9*2, (1, 1),
				kernel_initializer='he_normal',
				name='rpn_class_raw',
				reuse=tf.AUTO_REUSE)
		rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
		rpn_probs = tf.nn.softmax(rpn_class_logits)

		x = tf.layers.conv2d(x, 9 * 4, (1, 1),
			kernel_initializer='he_normal',
			name='rpn_bbox_pred',
			reuse=tf.AUTO_REUSE)

		rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])
		# layers_outputs.append([rpn_class_logits, rpn_probs, rpn_deltas])
		print(rpn_class_logits.shape, rpn_probs.shape, rpn_deltas.shape)
			
		# import pdb; pdb.set_trace()
		# outputs = list(zip((layers_outputs)))
		# outputs = [tf.concat(list(o), axis=1) for o in outputs]
		# rpn_class_logits, rpn_probs, rpn_deltas = outputs
		
		
		return rpn_class_logits, rpn_probs, rpn_deltas


	def target(self, gt_boxes, img_size=[512, 512], batch_size=4):
		import pdb; pdb.set_trace()
		anchors, valid_flags = self.generator.generate_pyramid_anchors(img_size)
		rpn_target_matchs, rpn_target_deltas = self.anchor_target.build_targets(
												anchors, valid_flags,
												gt_boxes, batch_size)

		return rpn_target_matchs, rpn_target_deltas


	def get_proposals(self, rpn_probs, rpn_deltas, img_size, with_probs=False):

		import pdb; pdb.set_trace()
		anchors, valid_flags = self.generator.generate_pyramid_anchors(img_size)

		rpn_probs = rpn_probs[:, :, 1]
		pad_shapes = [224, 224]
		proposal_list = self._get_proposals_single(rpn_probs,
												rpn_deltas, anchors,
												valid_flags, pad_shapes, 
												with_probs)

		return proposal_list






	def _get_proposals_single(self,
							rpn_probs,
							rpn_deltas, 
							anchors, 
							valid_flags, 
							img_shape, 
							with_probs):
		'''
		Calculate proposals.
		
		Args
		---
			rpn_probs: [num_anchors]
			rpn_deltas: [num_anchors, (dy, dx, log(dh), log(dw))]
			anchors: [num_anchors, (y1, x1, y2, x2)] anchors defined in 
				pixel coordinates.
			valid_flags: [num_anchors]
			img_shape: np.ndarray. [2]. (img_height, img_width)
			with_probs: bool.
		
		Returns
		---
			proposals: [num_proposals, (y1, x1, y2, x2)] in normalized 
				coordinates.
		'''

		H, W = img_shape
		
		# filter invalid anchors, int => bool
		valid_flags = tf.cast(valid_flags, tf.bool)
		# [369303] => [215169], respectively
		rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
		rpn_deltas = tf.boolean_mask(rpn_deltas, valid_flags)
		anchors = tf.boolean_mask(anchors, valid_flags)

		# Improve performance
		pre_nms_limit = min(6000, anchors.shape[0]) # min(6000, 215169) => 6000
		ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=True).indices
		# [215169] => [6000], respectively
		rpn_probs = tf.gather(rpn_probs, ix)
		rpn_deltas = tf.gather(rpn_deltas, ix)
		anchors = tf.gather(anchors, ix)
	
		# Get refined anchors, => [6000, 4]
		proposals = transform.delta2bbox(anchors, rpn_deltas, 
										self.rpn_target_means, self.rpn_target_stds)
		# clipping to valid area, [6000, 4]
		window = tf.constant([0., 0., H, W], dtype=tf.float32)
		proposals = transform.bbox_clip(proposals, window)
		
		# Normalize, (y1, x1, y2, x2)
		proposals = proposals / tf.constant([H, W, H, W], dtype=tf.float32)
		
		# NMS, indices: [2000]
		indices = tf.image.non_max_suppression(
			proposals, rpn_probs, self.proposal_count, self.nms_threshold)
		proposals = tf.gather(proposals, indices) # [2000, 4]
		
		if with_probs:
			proposal_probs = tf.expand_dims(tf.gather(rpn_probs, indices), axis=1)
			proposals = tf.concat([proposals, proposal_probs], axis=1)
 
		return proposals