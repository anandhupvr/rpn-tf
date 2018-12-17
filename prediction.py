import tensorflow as tf
import cv2


img_path = ''
img = np.expand_dims(cv2.imread(img_path), axis=0).astype('float32')

tf.reset_default_graph()

im = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, None, None, 3])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    print ("Model restored")
     