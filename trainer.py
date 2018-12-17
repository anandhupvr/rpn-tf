import tensorflow as tf
import sys
from loader.DataLoader import load
from models.net import network
from lib.loss import losses


num_epo = 500
dataset_path = sys.argv[1]
data_loader = load(dataset_path)
net = network()


rpn_cls_score_reshape, rpn_labels, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = net.build_network()

x, gt_boxes, im_dims = net.getPlaceholders()

saver = tf.train.Saver()

# init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_epo):
        for _ in range(len(open("train.txt", "r").readlines())):
            data = data_loader.data_batch()
            img, gt_box, im_info = data[0][0], data[0][1], data[0][2]
            loss = losses(rpn_cls_score_reshape, rpn_labels, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            train_step = optimizer.minimize(loss)
            sess.run(train_step, feed_dict={x:img, gt_boxes:gt_box, im_dims:im_info})
            ls_val = sess.run(loss, feed_dict={x:img, gt_boxes:gt_box, im_dims:(im_info)})
            print ('loss : {}       --> : {}'.format(ls_val, _))
        print ('loss : {}      epoch --> : {}'.format(ls_val, i))
    if i%5 == 0:
        save_path = saver.save(sess, 'weights/'+"model_{}.ckpt".format(i))
        print ("Model at {} epoch saved at {}".format(i, save_path))

