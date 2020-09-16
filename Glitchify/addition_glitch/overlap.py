import cv2
import numpy.random as random
import numpy as np

from common_file import labelMe_class
from common_file.return_class import RetClass

def to_int(value):
	return int(round(value))

def contur_to_list_int_points(contur):
	list_points = []

	for point in contur:
		(x,y) = point[0]
		list_points.append([int(x),int(y)])
	return list_points

def list_int_points_to_contur(list_points):
	contour = []

	for x,y in list_points:
		contour.append([np.array((x,y), dtype = np.int32)])

	array = np.array(contour, dtype = np.int32)
	return array


def change_overlap_contours(shape, contours):

	contors_out = []
	for i in range(len(contours)-1):
		mask = np.zeros((shape[0],shape[1], 1), np.uint8)

		change = list_int_points_to_contur(contours[i])
		cv2.drawContours(mask, [change], 0 , 255, -1)

		for j in range(i+1, len(contours)):
			new = list_int_points_to_contur(contours[j])
			cv2.drawContours(mask, [new],    0  , 0	, -1)

# 		cv2.imshow("test", mask)
# 		cv2.waitKey()
# 		cv2.destroyAllWindows()

		find_contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if len(find_contours) != 0:
			contors_out.append(contur_to_list_int_points(find_contours[0]))

	contors_out.append(contours[len(contours)-1])


	return contors_out


def overlap (img, label, lo = 5, hi = 10):

	h,w,_ = img.shape
	pic = np.copy(img)

	size_x = random.randint(to_int(w / 10), to_int(w/2) + 1)
	size_y = random.randint(to_int(h / 10), to_int(h/2) + 1)

	x0 = random.randint(to_int(w / 4), to_int(1 / 2 * w) +1)
	y0 = random.randint(to_int(h / 4), to_int(1 / 2 * h) +1)

	x1 = min(x0 + size_x, to_int(3 * w / 4))
	y1 = min(y0 + size_y, to_int(3 * w / 4))

	size_x = x1 - x0
	size_y = y1 - y0

	copy = np.copy(img[y0:y1,x0:x1,:])

	count_overlap =  random.randint(lo, hi+1)

	orientation_overlap_x = random.randint(-1, 2)
	orientation_overlap_y = random.randint(-1, 2)


	p_json_list = []

	while(orientation_overlap_x == 0 and orientation_overlap_y == 0):
		orientation_overlap_x = random.randint(-1, 2)
		orientation_overlap_y = random.randint(-1, 2)

	contours = [[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]]

	for i in range(count_overlap):
		offset_x =  random.randint(20, 40 + 1)
		offset_y =  random.randint(20, 40 + 1)

		x_o_0 = min(max(x0 + offset_x * orientation_overlap_x, 0),w-1)
		y_o_0 = min(max(y0 + offset_y * orientation_overlap_y, 0),h-1)

		size_o_x = min(max(x_o_0 + size_x, 0),w-1) - x_o_0
		size_o_y = min(max(y_o_0 + size_y, 0),h-1) - y_o_0

		if size_o_x == 0 or size_o_y == 0:
			break

		pic[y_o_0: y_o_0+size_o_y, x_o_0:x_o_0+size_o_x,:] = copy[ :size_o_y,:size_o_x,:]

		contours.append([[x_o_0, y_o_0],[x_o_0 + size_o_x, y_o_0],[x_o_0 + size_o_x, y_o_0+size_o_y],[x_o_0, y_o_0+size_o_y]])

		x0 = min(max(x0 + offset_x* orientation_overlap_x,0) ,w-1)
		y0 = min(max(y0 + offset_y* orientation_overlap_y,0) ,h-1)

	contours_out = change_overlap_contours(img.shape, contours)

	for contour in contours_out:
		p_shapes = labelMe_class.Shapes(label, contour, None, "polygon", {})
		p_json_list.append(p_shapes)

	res = RetClass(pic, p_json_list)
	return res
