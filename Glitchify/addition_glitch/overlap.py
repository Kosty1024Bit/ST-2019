import cv2
import numpy.random as random
import numpy as np

from common_file import labelMe_class
from common_file.return_class import RetClass



def intersection_point_vert_or_horizontal_line (line_1, line_2):
	((x_1_1, y_1_1),(x_1_2,y_1_2)) = line_1
	((x_2_1, y_2_1),(x_2_2,y_2_2)) = line_2

	if x_1_1 == x_1_2:
		x_vert = x_1_2
		y_horiz = y_2_1
	else:
		y_horiz = y_1_1
		x_vert = x_2_1


	return (x_vert, y_horiz)

def contiur_overlap(change_contour, new_contour):

	c_max_x = max(change_contour[0][0],change_contour[1][0],change_contour[2][0],change_contour[3][0])
	c_max_y = max(change_contour[0][1],change_contour[1][1],change_contour[2][1],change_contour[3][1])
	c_min_x = min(change_contour[0][0],change_contour[1][0],change_contour[2][0],change_contour[3][0])
	c_min_y = min(change_contour[0][1],change_contour[1][1],change_contour[2][1],change_contour[3][1])

	n_max_x = max(new_contour[0][0],new_contour[1][0],new_contour[2][0],new_contour[3][0])
	n_max_y = max(new_contour[0][1],new_contour[1][1],new_contour[2][1],new_contour[3][1])
	n_min_x = min(new_contour[0][0],new_contour[1][0],new_contour[2][0],new_contour[3][0])
	n_min_y = min(new_contour[0][1],new_contour[1][1],new_contour[2][1],new_contour[3][1])

	is_change_p =  [False,False,False,False]
	is_overlap_p = [False,False,False,False]

	count_point_overlap = 0

	for i in range(0,4):
		if  n_min_x <= change_contour[i][0] <= n_max_x and n_min_y <= change_contour[i][1] <= n_max_y:
			is_change_p[i] = True
			count_point_overlap+=1

		if  c_min_x < new_contour[i][0] < c_max_x and c_min_y < new_contour[i][1] < c_max_y:
			is_overlap_p[i] = True

	contour = []

	list_index = []
	list_after_del = []
	bool_revers = False
	for i in reversed(range(0,4)):

		if is_change_p[i] == False:
			if bool_revers:
				list_after_del.append(i)
			else:
				list_index.append(i)

		elif i == 1 or i == 2:
			bool_revers = True

	list_index = list_after_del + list_index


	for index in list_index:
			contour.append(change_contour[index])

	if count_point_overlap == 1:
		for i in range(0,4):
			if is_change_p[i] == True:
				c_p = change_contour[i]

			if is_overlap_p[i] == True:
				n_p = new_contour[i]

		point_1 = intersection_point_vert_or_horizontal_line((change_contour[list_index[2]],c_p),(n_p,new_contour[list_index[2]]))
		contour.append(point_1)
		contour.append(n_p)
		point_2 = intersection_point_vert_or_horizontal_line((change_contour[list_index[0]],c_p),(n_p,new_contour[list_index[0]]))
		contour.append(point_2)


	elif count_point_overlap == 2:

		for index in reversed(list_index):
			contour.append(new_contour[index])


	return contour


def contour_comparison (contour_1, contour_2):
	if len(contour_1) != len(contour_2):
		return False

	for i in range(len(contour_1)):
		if not (contour_1[i][0] == contour_2[i][0] and contour_1[i][1] == contour_2[i][1]):
			return False

	return True

def change_overlap_contours(contours):

	i = 1
	while(i < len(contours)):
		if not contour_comparison(contours[i-1], contours[i]):
			contours[i-1] = contiur_overlap(contours[i-1] ,contours[i])
			i+=1
		else:
			contours.pop(i)

	return contours


def overlap (img, label, lo = 5, hi = 10):

	h,w,_ = img.shape
	pic = np.copy(img)

	x0 = random.randint(0, int(w / 6))
	y0 = random.randint(0, int(h / 6))
	x1 = random.randint(int(3 * w / 6), w)
	y1 = random.randint(int(3 * h / 6), h)

	copy = np.copy(img[y0:y1,x0:x1,:])

	count_overlap =  random.randint(lo, hi+1)

	orientation_overlap_x = random.randint(-1, 2)
	orientation_overlap_y = random.randint(-1, 2)

	p_json_list = []

	while(orientation_overlap_x == 0 and orientation_overlap_y == 0):
		orientation_overlap_x = random.randint(-1, 2)
		orientation_overlap_y = random.randint(-1, 2)

	size_x = x1-x0
	size_y = y1-y0

	contours = [[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]]

	for i in range(count_overlap):
		offset_x =  random.randint(20, 20 + 1)
		offset_y =  random.randint(20, 20 + 1)

		x_o_0 = min(max(x0 + offset_x * (i+1) * orientation_overlap_x, 0),w-1)
		y_o_0 = min(max(y0 + offset_y * (i+1) * orientation_overlap_y, 0),h-1)

		size_o_x = min(max(x_o_0 + size_x, 0),w-1) - x_o_0
		size_o_y = min(max(y_o_0 + size_y, 0),h-1) - y_o_0

		pic[y_o_0: y_o_0+size_o_y, x_o_0:x_o_0+size_o_x,:] = copy[ :size_o_y,:size_o_x,:]

		contours.append([[x_o_0, y_o_0],[x_o_0 + size_o_x, y_o_0],[x_o_0 + size_o_x, y_o_0+size_o_y],[x_o_0, y_o_0+size_o_y]])

	change_overlap_contours(contours)

	for contour in contours:
		p_shapes = labelMe_class.Shapes(label, contour, None, "polygon", {})
		p_json_list.append(p_shapes)

	res = RetClass(pic, p_json_list)
	return res
