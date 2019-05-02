import os
import numpy as np
import sys
import random
import lib.rpn_labels as rpn_utils
import cv2


class Load:
	def __init__(self, dataset_path):
		self.dataset_path = dataset_path
		self.images_path = os.path.join(dataset_path, 'images')
		self.classes = os.listdir(self.images_path)
		self.build()

	def build(self):
		self.image_list = self._get_image_files()
		self.train_test_split()

	def _get_image_files(self):
		images_list = []
		for cat in os.listdir(self.images_path):
			for im in os.listdir(os.path.join(self.images_path, cat)):
				images_list.append(os.path.join(self.images_path, cat, im))
		return images_list

	def train_test_split(self):
		random.shuffle(self.image_list)
		train_count = int(len(self.image_list)*.9)
		train_file = open('train.txt','w')
		for img in self.image_list[:train_count]:
			train_file.write("%s\n"%img)
		test_file = open('test.txt','w')
		for img in self.image_list[train_count:]:
			test_file.write("%s\n"%img)


	def data_dict(self):
		i = 1
		all_imgs = {} 
		with open('train.txt','r') as f:
			print('Parsing annotation files')
			for line in f:
				sys.stdout.write('\r'+'idx=' + str(i))
				i += 1
				label_file = open(line.strip().replace('images','labelsbbox').replace('jpg', 'txt')).readlines()
				clas_name = int(label_file[0].strip())
				(x1, y1, x2, y2) = label_file[1].strip().split(' ')
				filename = line
				if filename not in all_imgs:
					all_imgs[filename] = {}
					all_imgs[filename]['filepath'] = filename
					all_imgs[filename]['bboxes'] = []
				all_imgs[filename]['bboxes'].append({'class':clas_name, 'x1': int(x1), 'x2': int(x2), 'y1':int(y1), 'y2':int(y2)})
			all_data = []
			for key in all_imgs:
				all_data.append(all_imgs[key])
		return all_data

	def data(self):
		all_img_data = self.data_dict()
		batch_size = 2
		image_size = (416, 416)
		# width, height = get_img_output_length(224, 224)
		while True:
			j = 0
			for i in range(0, len(all_img_data), batch_size):
				imgs = all_img_data[i:i+batch_size]
				x_img = []
				gt_box = []
				im_size = []
				class_id = []

				for img in imgs:
					x1 = img['bboxes'][0]['x1']
					x2 = img['bboxes'][0]['x2']
					y1 = img['bboxes'][0]['y1']
					y2 = img['bboxes'][0]['y2']
					cls_id = img['bboxes'][0]['class']
					# gta = [x1, y2, x2-x1, y2-y1]
					x_img_ = cv2.imread(img['filepath'].strip())
					height, width = size = x_img_.shape[0:2]
					gta = [y1*(416/width), x1*(416/height), y2*(416/width), x2*(416/height)]
					gta = np.expand_dims(np.array(gta), axis=0)
					gt_box.append(gta)
					x_img_ = cv2.resize(x_img_, image_size)
					x_img.append(x_img_)
					class_id.append(cls_id)
					# im_size.append(list(size))
				# anchors, true_index, false_index = rpn_utils.create_Labels_For_Loss(np.array(gt_box))
				x_img = np.array(x_img)
				# x_img = self.convert_imgslist_to_ndarray(x_img)
				j += 1
				# , anchors, true_index, false_index, np.array(im_size)
				class_id = np.expand_dims(np.array(class_id), axis=1)
				yield x_img, np.array(gt_box), class_id
