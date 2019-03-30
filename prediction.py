import tensorflow as tf
from PIL import Image
import sys
# from loader.DataLoader import load
import lib.utils as utils
import cv2
import numpy as np
from config.parameters import Config



tf.reset_default_graph()



C = Config()

bbox_threshold = 0.2

# load = load(dataset_path)


img_ = Image.open(sys.argv[1])
# im_w, im_h = img_.size()

new_graph = tf.Graph()


with tf.Session(graph=new_graph) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.import_meta_graph('weight/model_900.ckpt.meta')
    checkpoint = tf.train.latest_checkpoint('weight')
    import pdb; db.set_trace()
    saver.restore(sess, checkpoint)
    print ("model restored")
    img = np.expand_dims(img_.resize([224, 224]), axis=0)

    image_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')
    rpn_cls = tf.get_default_graph().get_tensor_by_name('rpn_cls_reshaped:0')
    rpn_box = tf.get_default_graph().get_tensor_by_name('rpn_bbox_reshaped:0')

    rpn = sess.run([rpn_cls, rpn_box], feed_dict={image_tensor:img})
    # R = utils.rpn_to_roi(P_rpn[0], P_rpn[1], C, 'tf', overlap_thresh=0.7)
    print ("test")
