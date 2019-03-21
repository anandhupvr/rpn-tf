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


img = Image.open(sys.argv[1])

new_graph = tf.Graph()


with tf.Session(graph=new_graph) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.import_meta_graph('weight/model_900.ckpt.meta')
    checkpoint = tf.train.latest_checkpoint('weight')

    saver.restore(sess, checkpoint)
    print ("model restored")
    img = np.expand_dims(img.resize([224, 224]), axis=0)

    image_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')
    rpn_reg_out = tf.get_default_graph().get_tensor_by_name('rpn_out_regre:0')
    rpn_cls_out = tf.get_default_graph().get_tensor_by_name('rpn_out_class:0')

    base_layer = tf.get_default_graph().get_tensor_by_name('conv5_3/Relu:0')

    import pdb; pdb.set_trace()
    P_rpn = sess.run([rpn_cls_out, rpn_reg_out, base_layer], feed_dict={image_tensor:img})
