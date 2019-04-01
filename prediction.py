import tensorflow as tf
import cv2
import sys
# from loader.DataLoader import load
import lib.utils as utils
import cv2
import numpy as np
from config.parameters import Config
import os


tf.reset_default_graph()



C = Config()
bbox_threshold = 0.2
img = []

img_ = cv2.imread(sys.argv[1])
width, height = img_.shape[1:3]
x_ratio, y_ratio = width/224, height/224
img = cv2.resize(img_, (224, 224))
img = np.expand_dims(img, axis=0)

new_graph = tf.Graph()


with tf.Session(graph=new_graph) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.import_meta_graph('weight/model_900.ckpt.meta')
    checkpoint = tf.train.latest_checkpoint('weight')
    saver.restore(sess, checkpoint)
    print ("model restored")
    image_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')
    rpn_cls = tf.get_default_graph().get_tensor_by_name('rpn_cls_reshaped:0')
    rpn_box = tf.get_default_graph().get_tensor_by_name('rpn_bbox_reshaped:0')
    rpn = sess.run([rpn_cls, rpn_box], feed_dict={image_tensor:img})
    boxes = rpn[1].reshape(rpn[1].shape[0], rpn[1].shape[1]*rpn[1].shape[2], rpn[1].shape[3])
    boxes[0][:,[0, 2]] = boxes[0][:,[0, 2]] * 224
    boxes[0][:, [1, 3]] = boxes[0][:, [1, 3]] * 224
    import pdb; pdb.set_trace()
    nms_box = utils.non_max_suppression_fast(np.abs(boxes[0]), 0.7)
    nms_box = nms_box[:50]
    utils.bbox_plot(img[0], nms_box)