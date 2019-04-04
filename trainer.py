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



num_epo = 901
dataset_path = sys.argv[1]
load = load(dataset_path)


data = load.get_data()
num_anchors = 9


data_get = load.get_rpn(data, C, get_img_output_length)

net = network()

rpn_out,x = net.build_network()
# x, cls_plc, box_plc = net.get_placeholder()
total_loss, cls_loss, bbox_loss, true_obj_loss, false_obj_loss, g_bbox, true_index, false_index = losses.rpn_loss(rpn_out[0], rpn_out[1])



tf.summary.scalar("total loss", total_loss)
tf.summary.scalar("rpn cls loss", cls_loss)
tf.summary.scalar("rpn bbox loss", bbox_loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
# train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(rpn_loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter( 'logs/', sess.graph)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    for i in range(num_epo):
        los = 0
        for _ in range(108):
            x_img, anchors, true_index_batch, false_index_batch = next(data_get)
            summary = sess.run([merged, train_step], feed_dict={x:x_img, g_bbox:anchors, true_index:true_index_batch, false_index:false_index_batch})
            ls_val = sess.run(total_loss, feed_dict={x:x_img, g_bbox:anchors, true_index:true_index_batch, false_index:false_index_batch})
            loss_ = ls_val + los
            los = loss_
            print (ls_val)
        train_writer.add_summary(summary[0], i)
        print ("epoch : %s  ***** avg losss : %s ***** "%(i, loss_/108))

        if i%100 == 0:
            save_path = saver.save(sess, 'weight/'+"model_{}.ckpt".format(i))
            print ("epoch : %s   saved at  %s "%(i, save_path))