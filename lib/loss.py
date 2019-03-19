from keras import backend as K
from keras.objectives import categorical_crossentropy
import numpy as np


if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4



def rpn_loss_regr_org(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'th':
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
		else:
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'th':
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(np.float32(x))
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + np.float32(y_true[:, :4 * num_anchors, :, :]))
		else:
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + np.float32(y_true[:, :, :, :4 * num_anchors]))

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'tf':
			return lambda_rpn_class * K.sum(np.float32(y_true[:, :, :, :num_anchors]) * K.binary_crossentropy(tf.convert_to_tensor(y_pred[:, :, :, num_anchors:]), tf.convert_to_tensor(y_true[:, :, :, num_anchors:], np.float32))) / K.sum(np.float32(epsilon + y_true[:, :, :, :num_anchors]))
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num



def rpn_loss_cls_org(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'tf':
			return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num


def smoothL1(y_true, y_pred):
	nd=K.tf.where(K.tf.not_equal(y_true,0))
	y_true=K.tf.gather_nd(y_true,nd)
	y_pred=K.tf.gather_nd(y_pred,nd)
	x = K.tf.losses.huber_loss(y_true,y_pred)
#     x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
	return x

def loss_cls(y_true, y_pred):
	condition = K.not_equal(y_true, -1)
	indices = K.tf.where(condition)

	target = K.tf.gather_nd(y_true, indices)
	output = K.tf.gather_nd(y_pred, indices)
	loss = K.binary_crossentropy(target, output)
	return K.mean(loss)




def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		import pdb; pdb.set_trace()
		x = y_true[:, :, 4*num_classes:] - y_pred
		# x = y_true - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))


# def _compute_loss(prediction_tensor, target_tensor, weights):
# """Compute loss function.

# Args:
#   prediction_tensor: A float tensor of shape [batch_size, num_anchors,
# 	code_size] representing the (encoded) predicted locations of objects.
#   target_tensor: A float tensor of shape [batch_size, num_anchors,
# 	code_size] representing the regression targets
#   weights: a float tensor of shape [batch_size, num_anchors]

# Returns:
#   loss: a float tensor of shape [batch_size, num_anchors] tensor
# 	representing the value of the loss function.
# """
# 	return tf.reduce_sum(treturn tf.reduce_sum(tf.losses.huber_loss(
# 		target_tensor,
# 		prediction_tensor,
# 		delta=1.0,
# 		weights=tf.expand_dims(weights, axis=2),
# 		loss_collection=None,
# 		reduction=tf.losses.Reduction.NONE
# 	), axis=2)
def regr_loss(y_true, y_pred):
	x = tf.abs(y_true - y_pred)
	loc_loss = ((x < 1) * 0.5 * x**2) + ((x >= 1) * (x-0.5))