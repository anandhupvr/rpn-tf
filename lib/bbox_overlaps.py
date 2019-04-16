import numpy as np



def bbox_overlaps(anchors, is_inside, gt_boxes):

	Batch_Size = anchors.shape[0]
	K = anchors.shape[1]
	A = anchors.shape[2]
	gt = gt_boxes.shape[1]


	true_index = np.zeros((Batch_Size, K, A), dtype=np.int32)
	flase_index = np.zeros((Batch_Size, K, A), dtype=np.int32)

	max_g = 0

	for b in range(Batch_Size):

		# for j in range(gt):
		# 	gt_boxes[b,j, 0] = gt_boxes[b,j, 0] / 16.
		# 	gt_boxes[b,j, 1] = gt_boxes[b,j, 1] / 16.
		# 	gt_boxes[b,j, 2] = gt_boxes[b,j, 2] / 16.
		# 	gt_boxes[b,j, 3] = gt_boxes[b,j, 3] / 16.
		# for k in range(K):
		# 	for a in range(A):
		# 		if is_inside[b, k, a] == 1:
		# 			anchors[b, k, a, 0] = anchors[b, k, a, 0] / 16.
		# 			anchors[b, k, a, 1] = anchors[b, k, a, 1] / 16.
		# 			anchors[b, k, a, 2] = anchors[b, k, a, 2] / 16.
		# 			anchors[b, k, a, 3] = anchors[b, k, a, 3] / 16.

		if max_g < gt_boxes[b].shape[0]:
			max_g = gt_boxes[b].shape[0]
	overlaps = np.zeros((Batch_Size, K, A, max_g))

	for b in range(Batch_Size):
		G = gt_boxes[b].shape[0]
		for g in range(G):
			box_area = ( 
						(gt_boxes[b][g, 2] - gt_boxes[b][g, 0] + 1) *
						(gt_boxes[b][g, 3] - gt_boxes[b][g, 1] + 1)
						)
			max_overlap = 0
			max_k = 0
			max_a = 0
			for k in range(K):
				for a in range(A):
					if is_inside[b, k, a] == 1:
						iw = ( 
							min(anchors[b, k, a, 2], gt_boxes[b][g, 2]) - 
							max(anchors[b, k, a, 0], gt_boxes[b][g, 0]) + 1
							)
						if iw > 0:
							ih = (
								min(anchors[b, k, a, 3], gt_boxes[b][g, 3]) -
								max(anchors[b, k, a, 1], gt_boxes[b][g, 1]) + 1
								)
							if ih > 0:
								ua = float(
											(anchors[b, k, a, 2] - anchors[b, k, a, 0] + 1) * 
											(anchors[b, k, a, 3] - anchors[b, k, a, 1] + 1) + 
											box_area - iw * ih
											)
								overlaps[b, k, a, g] = iw * ih / ua
								if max_overlap < ((iw * ih / ua)):
									max_overlap = iw * ih / ua
									max_k = k
									max_a = a

			true_index[b, max_k, max_a] = 1


		for k in range(K):
			for a in range(A):
				if is_inside[b, k, a] == 1:
					max_overlap = 0
					max_g = 0
					for g in range(G):
						if overlaps[b, k, a, g] > 0:
							max_overlap = overlaps[b, k, a, g]
							max_g = g
					if max_overlap > 0.6:
						true_index[b, k, a] = 1
					else:
						if max_overlap <= 0.35:
							flase_index[b, k, a] = 1

					if true_index[b, k, a] == 1:
						ex_width = anchors[b, k, a, 2] - anchors[b, k, a, 0] + 1
						ex_height = anchors[b, k, a, 3] - anchors[b, k, a, 1] + 1
						ex_center_x = anchors[b, k, a, 0] + ex_width / 2.0
						ex_center_y = anchors[b, k, a, 1] + ex_height / 2.0
						gt_width = gt_boxes[b][max_g, 2] - gt_boxes[b][max_g, 0] + 1
						gt_height = gt_boxes[b][max_g, 3] - gt_boxes[b][max_g, 1] + 1
						gt_center_x = gt_boxes[b][max_g, 0] + gt_width / 2.0
						gt_center_y = gt_boxes[b][max_g, 1] + gt_height / 2.0
						anchors[b, k, a, 0] = (gt_center_x - ex_center_x) / (ex_width)
						anchors[b, k, a, 1] = (gt_center_y - ex_center_y) / (ex_height)
						anchors[b, k, a, 2] = np.log(gt_width / (ex_width))
						anchors[b, k, a, 3] = np.log(gt_height / (ex_height))

	return anchors, true_index, flase_index