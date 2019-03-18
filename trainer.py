from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf
import sys
from loader.data import load
from models.net import network
import lib.loss as losses
from config.parameters import Config
import lib.utils as utils


C = Config()
def get_img_output_length(width, height):
    return (int(width/16),int(height/16))



num_epo = 500
dataset_path = sys.argv[1]
load = load(dataset_path)


data = load.get_data()
num_anchors = 9
data_gen = load.get_anchor_gt(data, C, get_img_output_length, mode='train')

net = network()


rpn_out = net.build_network()
x, cls_plc, box_plc = net.get_placeholder()

lsr = losses.rpn_loss_cls_org(9)
lgr = losses.rpn_loss_regr_org(9)
los_c = lsr(cls_plc, rpn_out[0])
los_b = lgr(box_plc, rpn_out[1])
rpn_loss = los_c + los_b

tf.summary.scalar("loss", rpn_loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(rpn_loss)

saver = tf.train.Saver()


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter( 'logs/', sess.graph)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    for i in range(num_epo):
        los = 0
        for _ in range(108):
            X, Y, image_data, debug_img, debug_num_pos = next(data_gen)
            summary = sess.run([merged, train_step], feed_dict={x:X, cls_plc:Y[0], box_plc:Y[1]})
            ls_val = sess.run(rpn_loss, feed_dict={x:X, cls_plc:Y[0], box_plc:Y[1]})
            loss_ = ls_val + los
            los = loss_
            print (loss_)
        train_writer.add_summary(summary[0], i)
        print ("epoch : %s  ***** avg losss : %s ***** "%(i, loss_/108))

        if i%100 == 0:
            save_path = saver.save(sess, 'weight/'+"model_{}.ckpt".format(i))
            print ("epoch : %s   saved at  %s "%(i, save_path))