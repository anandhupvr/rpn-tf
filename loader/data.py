from loader.utils import Bbox
import loader.utils as utils
import os
from PIL import Image
import numpy as np
import sys
import copy
import random
import loader.proposal_targets as rpn_target
import lib.rpn_labels as rpn_utils
import cv2

class load:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = ('mango', 'meatballs_dish')
        self._class_to_ind = dict(list(zip(self.classes, list(range(len(self.classes))))))
        self.imdb = {}
        self.images_path = os.path.join(self.dataset_path, 'images')
        # utils.train_test_split(self.dataset_path+"images/racoon")
        self.ptr = 0
        self.cats = os.listdir(self.dataset_path + "/images")
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


    def get_data(self):
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

    def get_rpn(self, all_img_data, C, get_img_output_length):

        batch_size = 4
        image_size = (224, 224)
        # width, height = get_img_output_length(224, 224)
        while True:
            j = 0
            for i in range(0, len(all_img_data), batch_size):
                imgs = all_img_data[i:i+batch_size]
                # rpn_labels = []
                # rpn_bbox_targets = []
                # rpn_bbox_inside_weights = []
                # rpn_bbox_outside_weights = []
                # bbox_targets = []
                # bbox_inside_weights = []
                # bbox_outside_weights = []
                x_img = []
                gt_box = []

                for img in imgs:
                    x1 = img['bboxes'][0]['x1']
                    x2 = img['bboxes'][0]['x2']
                    y1 = img['bboxes'][0]['y1']
                    y2 = img['bboxes'][0]['y2']
                    gta = [x1, y2, x2-x1, y2-y1]
                    gta = np.expand_dims(np.array(gta), axis=0)
                    gt_box.append(gta)
                    x_img_ = cv2.imread(img['filepath'].strip())
                    x_img_ = cv2.resize(x_img_, (224, 224))
                    # x_img_ = Image.open(img['filepath'].strip())
                    # x_img_ = np.array(x_img_.resize((224, 224), Image.ANTIALIAS))
                    x_img.append(x_img_)
                anchors, true_index, false_index = rpn_utils.create_Labels_For_Loss(np.array(gt_box))
                x_img = np.array(x_img)
                j += 1

                yield x_img, anchors, true_index, false_index




    def union(self, au, bu, area_intersection):
        area_a = (au[2] - au[0]) * (au[3] - au[1])
        area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
        area_union = area_a + area_b - area_intersection
        return area_union

    def intersection(self, ai, bi):
        x = max(ai[0], bi[0])
        y = max(ai[1], bi[1])
        w = min(ai[2], bi[2]) - x
        h = min(ai[3], bi[3]) - y
        if w < 0 or h < 0:
            return 0
        return w*h

    def iou(self, a, b):
    # a and b should be (x1,y1,x2,y2)

        if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
            return 0.0

        area_i = self.intersection(a, b)
        area_u = self.union(a, b, area_i)

        return float(area_i) / float(area_u + 1e-6)





    def calc_rpn(self, C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
        """(Important part!) Calculate the rpn for all anchors 
            If feature map has shape 38x50=1900, there are 1900x9=17100 potential anchors
        
        Args:
            C: config
            img_data: augmented image data
            width: original image width (e.g. 600)
            height: original image height (e.g. 800)
            resized_width: resized image width according to C.im_size (e.g. 300)
            resized_height: resized image height according to C.im_size (e.g. 400)
            img_length_calc_function: function to calculate final layer's feature map (of base model) size according to input image size

        Returns:
            y_rpn_cls: list(num_bboxes, y_is_box_valid + y_rpn_overlap)
                y_is_box_valid: 0 or 1 (0 means the box is invalid, 1 means the box is valid)
                y_rpn_overlap: 0 or 1 (0 means the box is not an object, 1 means the box is an object)
            y_rpn_regr: list(num_bboxes, 4*y_rpn_overlap + y_rpn_regr)
                y_rpn_regr: x1,y1,x2,y2 bunding boxes coordinates
        """
        downscale = float(C.rpn_stride) 
        anchor_sizes = C.anchor_box_scales   # 128, 256, 512
        anchor_ratios = C.anchor_box_ratios  # 1:1, 1:2*sqrt(2), 2*sqrt(2):1
        num_anchors = len(anchor_sizes) * len(anchor_ratios) # 3x3=9

        # calculate the output map size based on the network architecture
        (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

        n_anchratios = len(anchor_ratios)    # 3

        # initialise empty output objectives
        y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
        y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
        y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

        num_bboxes = len(img_data['bboxes'])

        num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
        best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
        best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
        best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
        best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

        # get the GT box coordinates, and resize to account for image resizing
        gta = np.zeros((num_bboxes, 4))
        for bbox_num, bbox in enumerate(img_data['bboxes']):
            # get the GT box coordinates, and resize to account for image resizing
            gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
            gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
            gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
            gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
        
        # rpn ground truth

        for anchor_size_idx in range(len(anchor_sizes)):
            for anchor_ratio_idx in range(n_anchratios):
                anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
                anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]   
                
                for ix in range(output_width):                  
                    # x-coordinates of the current anchor box   
                    x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                    x2_anc = downscale * (ix + 0.5) + anchor_x / 2  
                    
                    # ignore boxes that go across image boundaries                  
                    if x1_anc < 0 or x2_anc > resized_width:
                        continue
                        
                    for jy in range(output_height):

                        # y-coordinates of the current anchor box
                        y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                        y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                        # ignore boxes that go across image boundaries
                        if y1_anc < 0 or y2_anc > resized_height:
                            continue

                        # bbox_type indicates whether an anchor should be a target
                        # Initialize with 'negative'
                        bbox_type = 'neg'

                        # this is the best IOU for the (x,y) coord and the current anchor
                        # note that this is different from the best IOU for a GT bbox
                        best_iou_for_loc = 0.0

                        for bbox_num in range(num_bboxes):
                            
                            # get IOU of the current GT box and the current anchor box
                            curr_iou = self.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
                            # calculate the regression targets if they will be needed
                            if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                                cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                                cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                                cxa = (x1_anc + x2_anc)/2.0
                                cya = (y1_anc + y2_anc)/2.0

                                # x,y are the center point of ground-truth bbox
                                # xa,ya are the center point of anchor bbox (xa=downscale * (ix + 0.5); ya=downscale * (iy+0.5))
                                # w,h are the width and height of ground-truth bbox
                                # wa,ha are the width and height of anchor bboxe
                                # tx = (x - xa) / wa
                                # ty = (y - ya) / ha
                                # tw = log(w / wa)
                                # th = log(h / ha)
                                tx = (cx - cxa) / (x2_anc - x1_anc)
                                ty = (cy - cya) / (y2_anc - y1_anc)
                                tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                                th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
                            
                            if img_data['bboxes'][bbox_num]['class'] != 'bg':

                                # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                                if curr_iou > best_iou_for_bbox[bbox_num]:
                                    best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                    best_iou_for_bbox[bbox_num] = curr_iou
                                    best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                    best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

                                # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                                if curr_iou > C.rpn_max_overlap:
                                    bbox_type = 'pos'
                                    num_anchors_for_bbox[bbox_num] += 1
                                    # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                    if curr_iou > best_iou_for_loc:
                                        best_iou_for_loc = curr_iou
                                        best_regr = (tx, ty, tw, th)

                                # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                                if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                    # gray zone between neg and pos
                                    if bbox_type != 'pos':
                                        bbox_type = 'neutral'

                        # turn on or off outputs depending on IOUs
                        if bbox_type == 'neg':
                            y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                            y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        elif bbox_type == 'neutral':
                            y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                            y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        elif bbox_type == 'pos':
                            y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                            y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                            start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                            y_rpn_regr[jy, ix, start:start+4] = best_regr

        # we ensure that every bbox has at least one positive RPN region

        for idx in range(num_anchors_for_bbox.shape[0]):
            if num_anchors_for_bbox[idx] == 0:
                # no box with an IOU greater than zero ...
                if best_anchor_for_bbox[idx, 0] == -1:
                    continue
                y_is_box_valid[
                    best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                    best_anchor_for_bbox[idx,3]] = 1
                y_rpn_overlap[
                    best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                    best_anchor_for_bbox[idx,3]] = 1
                start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
                y_rpn_regr[
                    best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

        y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
        y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

        y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
        y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

        y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
        y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

        pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
        neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

        num_pos = len(pos_locs[0])

        # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
        # regions. We also limit it to 256 regions.
        num_regions = 256

        if len(pos_locs[0]) > num_regions/2:
            val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
            y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
            num_pos = num_regions/2

        if len(neg_locs[0]) + num_pos > num_regions:
            val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
            y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
        y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
        y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

        return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos

    def get_new_img_size(self, width, height, img_min_side=224):
        # if width <= height:
        #   f = float(img_min_side) / width
        #   resized_height = int(f * height)
        #   resized_width = img_min_side
        # else:
        #   f = float(img_min_side) / height
        #   resized_width = int(f * width)
        #   resized_height = img_min_side
        resized_width = img_min_side
        resized_height = img_min_side

        return resized_width, resized_height

    def augment(self, img_data, config, augment=True):
        assert 'filepath' in img_data
        assert 'bboxes' in img_data
        # assert 'width' in img_data
        # assert 'height' in img_data
        img_data_aug = copy.deepcopy(img_data)

        img = cv2.imread(img_data_aug['filepath'].strip())

        if augment:
            rows, cols = img.shape[:2]
            if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
                img = cv2.flip(img, 1)
                for bbox in img_data_aug['bboxes']:
                    x1 = bbox['x1']
                    x2 = bbox['x2']
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2

            if config.use_vertical_flips and np.random.randint(0, 2) == 0:
                img = cv2.flip(img, 0)
                for bbox in img_data_aug['bboxes']:
                    y1 = bbox['y1']
                    y2 = bbox['y2']
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2

            if config.rot_90:
                angle = np.random.choice([0,90,180,270],1)[0]
                if angle == 270:
                    img = np.transpose(img, (1,0,2))
                    img = cv2.flip(img, 0)
                elif angle == 180:
                    img = cv2.flip(img, -1)
                elif angle == 90:
                    img = np.transpose(img, (1,0,2))
                    img = cv2.flip(img, 1)
                elif angle == 0:
                    pass
                for bbox in img_data_aug['bboxes']:
                    x1 = bbox['x1']
                    x2 = bbox['x2']
                    y1 = bbox['y1']
                    y2 = bbox['y2']
                    if angle == 270:
                        bbox['x1'] = y1
                        bbox['x2'] = y2
                        bbox['y1'] = cols - x2
                        bbox['y2'] = cols - x1
                    elif angle == 180:
                        bbox['x2'] = cols - x1
                        bbox['x1'] = cols - x2
                        bbox['y2'] = rows - y1
                        bbox['y1'] = rows - y2
                    elif angle == 90:
                        bbox['x1'] = rows - y2
                        bbox['x2'] = rows - y1
                        bbox['y1'] = x1
                        bbox['y2'] = x2        
                    elif angle == 0:
                        pass

        img_data_aug['width'] = img.shape[1]
        img_data_aug['height'] = img.shape[0]
        return img_data_aug, img

    def get_anchor_gt(self, all_img_data, C, img_length_calc_function, mode):
        batch_size = 4
        while True:
            j = 0
            for i in range(0, len(all_img_data), batch_size):
                imgs = all_img_data[i:i+batch_size]
                x_img_ = []
                y_rpn_cls_ = []
                y_rpn_regr_ = []
                img_data_aug_ = []
                debug_img_ = []
                num_pos_ = []

                for j, img_data in zip(range(len(imgs)), imgs):
                    # read in image, and optionally add augmentation
                    if mode == 'train':
                        img_data_aug, x_img = self.augment(img_data, C, augment=True)
                    else:
                        img_data_aug, x_img = self.augment(img_data, C, augment=False)

                    (width, height) = (img_data_aug['width'], img_data_aug['height'])
                    (rows, cols, _) = x_img.shape

                    assert cols == width
                    assert rows == height

                    # get image dimensions for resizing
                    (resized_width, resized_height) = self.get_new_img_size(width, height, C.im_size)

                    # resize the image so that smalles side is length = 300px
                    x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                    debug_img = x_img.copy()
                    try:
                        y_rpn_cls, y_rpn_regr, num_pos = self.calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
                    except:
                        continue

                    # Zero-center by mean pixel, and preprocess image

                    x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
                    x_img = x_img.astype(np.float32)
                    x_img[:, :, 0] -= C.img_channel_mean[0]
                    x_img[:, :, 1] -= C.img_channel_mean[1]
                    x_img[:, :, 2] -= C.img_channel_mean[2]
                    x_img /= C.img_scaling_factor

                    x_img = np.transpose(x_img, (2, 0, 1))
                    x_img = np.expand_dims(x_img, axis=0)
                    y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling
                    x_img = np.transpose(x_img, (0, 2, 3, 1))[0, :, :, :]
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))[0, :, :, :]
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))[0, :, :, :]
                    x_img_.append(x_img)
                    y_rpn_cls_.append(y_rpn_cls)
                    y_rpn_regr_.append(y_rpn_regr)
                    img_data_aug_.append(img_data_aug)
                    debug_img_.append(debug_img_)
                    num_pos_.append(num_pos)
                j += 1

                yield np.copy(x_img_), [np.copy(y_rpn_cls_), np.copy(y_rpn_regr_)], img_data_aug_, debug_img_, num_pos_