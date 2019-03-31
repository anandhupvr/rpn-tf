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
# load = load(dataset_path)
# imgs = os.listdir(sys.argv[1])
# for im in imgs[0:4]:
#     img_ = cv2.imread(os.path.join(sys.argv[1], im))
#     img_ = cv2.resize(img_, (224, 224))
#     img.append(img_)
img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)
# im_w, im_h = img_.size()
# img_shaped = np.array(img)
new_graph = tf.Graph()


with tf.Session(graph=new_graph) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.import_meta_graph('weight/model_900.ckpt.meta')
    checkpoint = tf.train.latest_checkpoint('weight')
    import pdb; pdb.set_trace()
    saver.restore(sess, checkpoint)
    print ("model restored")
    # img = np.expand_dims(img_.resize([224, 224]), axis=0)

    image_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')
    rpn_cls = tf.get_default_graph().get_tensor_by_name('rpn_cls_reshaped:0')
    rpn_box = tf.get_default_graph().get_tensor_by_name('rpn_bbox_reshaped:0')

    rpn = sess.run([rpn_cls, rpn_box], feed_dict={image_tensor:img})
    boxes = rpn[1].reshape(rpn[1].shape[0], rpn[1].shape[1]*rpn[1].shape[2], rpn[1].shape[3])
    nms_box = utils.non_max_suppression_fast(boxes[0])
    utils.bbox_plot(nms_box)
    # R = utils.rpn_to_roi(P_rpn[0], P_rpn[1], C, 'tf', overlap_thresh=0.7)
    print ("test")
