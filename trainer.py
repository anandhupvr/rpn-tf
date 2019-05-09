from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf
import sys
from loader.data import Load
from models.net import network
import lib.loss as losses
from config.parameters import Config
# import lib.utils as utils


C = Config()


tf.reset_default_graph()

num_epo = 901
dataset_path = sys.argv[1]
load = Load(dataset_path)


inputs = load.data()

net = network()

x, bbox, cls_id = net.get_placeholder()


cls_loss, regress_loss = net.build(C)
total_loss = tf.add(cls_loss, regress_loss)

tf.summary.scalar("class loss", cls_loss)
tf.summary.scalar("bbox loss", regress_loss)
tf.summary.scalar("total loss", total_loss)


train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(total_loss)

saver = tf.train.Saver()

with tf.Session() as sess:
	train_writer = tf.summary.FileWriter( 'logs/', sess.graph)
	merged = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())
	for i in range(num_epo):
		los = 0
		# manualy done as per batch size
		for _ in range(242):
			x_img, box, class_id = next(inputs)
			summary = sess.run([merged, train_step], feed_dict={x:x_img, bbox:box, cls_id:class_id})
			ls_val = sess.run(total_loss, feed_dict={x:x_img, bbox:box, cls_id:class_id})
			loss_ = ls_val + los
			los = loss_
			if _ % 100 == 0: print (ls_val)
		train_writer.add_summary(summary[0], i)
		print ("epoch : %s  ***** avg losss : %s ***** "%(i, loss_/242))

		if i%100 == 0:
			save_path = saver.save(sess, 'weight/'+"model_{}.ckpt".format(i))
			print ("epoch : %s   saved at  %s "%(i, save_path))