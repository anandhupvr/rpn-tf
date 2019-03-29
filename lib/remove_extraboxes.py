import numpy as np


def remove_extraboxes(array1, array2, select, batch):


	remove_size = select.shape[0]
	extract_array1 = np.zeros((remove_size), dtype=np.int32)
	extract_array2 = np.zeros((remove_size), dtype=np.int32)

	for rs in range(remove_size):
		extract_array1[rs] = array1[select[rs]]
		extract_array2[rs] = array2[select[rs]]
	return batch, extract_array1, extract_array2