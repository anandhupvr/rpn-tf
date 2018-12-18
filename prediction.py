import tensorflow as tf
import cv2
import numpy as np


img_path = '/home/user1/Downloads/meat_balls.jpg'



tf.reset_default_graph()



model_path = '/home/christie/junk/weights/'

new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:

    tf.global_variables_initializer().run()
    saver = tf.train.import_meta_graph(model_path + "model_0.ckpt.meta")
    checkpoint = tf.train.latest_checkpoint(model_path)
    import pdb; pdb.set_trace()
    saver.restore(sess, checkpoint)
    img = np.expand_dims(cv2.imread(img_path), axis=0).astype('float32')
    print ("model restored!")
    im_dims = tf.placeholder(tf.float32, shape=[2])
    img_info = (img.shape[1], img.shape[2])
    inp = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
    inp_dim = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
    out = tf.get_default_graph().get_tensor_by_name("f-rcnn/rpn_out_regre/bias/Adam_511:0")
    p = sess.run(out, feed_dict={inp:img})
    print (p)
    # p = sess.run(out,feed_dict={inp:img})
    # p = np.reshape(p,[1, config["GRID_H"], config["GRID_W"], config["BOX"], 4 + 1 + config["CLASS"]])
    # print (set(p[:,:,:,:,4].flatten()))
