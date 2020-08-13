'''
	The input is a python [Height * Width * 3] array, which is a picture to add glitches.
	Notice that the input value should range from 0 to 255.
	The output is a python [Height * Width * 3] array, which is a picture with glitch added.
'''

import cv2
import numpy.random as random
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

from common_file import labelMe_class
from common_file.tree_return_class import TreeRet

def check_val(value):
	if value > 255:
		value = 255
	if value < 0:
		value = 0
	return value


def intersection_check(list_coordinate, point1, point2):
	for (point_l_1, point_l_2) in list_coordinate:
		if not ((point1[1] < point_l_2[1] or point2[1] < point_l_1[1]) and (point2[0] < point_l_1[0] or point1[0] > point_l_2[0])):
				return True
	return False


def white_square(picture, label, lo = 2, hi = 15):
	height = picture.shape[0]
	width = picture.shape[1]
	number_of_patches = random.randint(lo,hi+1)

	overlay = picture.copy()
	pic = picture.copy()
	f_json_list = []
	max_x = 0
	max_y = 0
	min_x = width
	min_y = height

	list_coordinate_rectangle = []

	for i in range(number_of_patches):


		red = check_val(random.randint(240,256))
		green = check_val(random.randint(240,256))
		blue = check_val(random.randint(240,256))
		color = [blue, green, red]

		is_intersection = True

		count = 0
		while(is_intersection):
			forfeit = int(round(20 / 10000 * count))

			size_x = random.randint(25 - forfeit, 80 - forfeit)
			size_y = random.randint(25 - forfeit, 80 - forfeit)
			count += 1
			first_y = random.randint(int(height* 0.1), int(height*0.8))
			first_x = random.randint(int(width * 0.1), int(width *0.8))

			last_y = min(first_y + size_x, height - 1)
			last_x = min(first_x + size_y, width - 1)

			if(count < 10000):
				is_intersection = intersection_check(list_coordinate_rectangle, [first_x, first_y], [last_x, last_y])
			else:
				is_intersection = False
				print(list_coordinate_rectangle)
				print("out with ", first_x, first_y, last_x, last_y)

		list_coordinate_rectangle.append([[first_x,first_y], [last_x, last_y]])

		#на случай, если фигура не вписывается в картинку
		size_x = last_x - first_x
		size_y = last_y - first_y

		min_x = min(min_x, first_x, last_x)
		max_x = max(max_x, first_x, last_x)

		min_y = min(min_y, first_y, last_y)
		max_y = max(max_y, first_y, last_y)

		x_top   = random.randint(int(round(first_x + 0.1 * size_x)), int(round(last_x - 0.1 * size_x)))
		y_left  = random.randint(int(round(first_y + 0.1 * size_y)), int(round(last_y - 0.1 * size_y)))
		x_down  = random.randint(int(round(first_x + 0.1 * size_x)), int(round(last_x - 0.1 * size_x)))
		y_right = random.randint(int(round(first_y + 0.1 * size_y)), int(round(last_y - 0.1 * size_y)))

		pts = np.array(((x_top, last_y), (last_x, y_right), (x_down, first_y), (first_x,y_left)), dtype=int)
		cv2.fillConvexPoly(overlay, pts, color)

		f_shapes = labelMe_class.Shapes(label, [[first_x, first_y], [last_x, last_y]], None, "rectangle", {})

		f_json_list.append(f_shapes.to_string_form())

	alpha = 1
	cv2.addWeighted(overlay, alpha, pic, 1 - alpha, 0, pic)
	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y],[max_x, max_y]], None, "rectangle", {})

	res = TreeRet(pic, f_json_list, [r_shapes.to_string_form()])
	return res



def black_tree(picture, label, lo = 2, hi = 15):
	height = picture.shape[0]
	width = picture.shape[1]
	number_of_patches = random.randint(lo,hi+1)

	overlay = picture.copy()
	pic = picture.copy()
	f_json_list = []
	max_x = 0
	max_y = 0
	min_x = width
	min_y = height

	list_coordinate_rectangle = []

	for i in range(number_of_patches):


		red = check_val(random.randint(0,20))
		green = check_val(random.randint(0,20))
		blue = check_val(random.randint(0,20))
		color = [blue, green, red]

		is_intersection = True

		count = 0
		while(is_intersection):
			forfeit = int(round(20 / 10000 * count))

			size_x = random.randint(25 - forfeit, 60 - forfeit)
			size_y = random.randint(40 - forfeit, 60 - forfeit)
			count += 1
			first_y = random.randint(int(height* 0.1), int(height*0.8))
			first_x = random.randint(int(width * 0.1), int(width *0.8))

			last_y = min(first_y + size_x, height - 1)
			last_x = min(first_x + size_y, width - 1)

			if(count < 10000):
				is_intersection = intersection_check(list_coordinate_rectangle, [first_x, first_y], [last_x, last_y])
			else:
				is_intersection = False
				print(list_coordinate_rectangle)
				print("out with ", first_x, first_y, last_x, last_y)

		list_coordinate_rectangle.append([[first_x,first_y], [last_x, last_y]])

		#на случай, если фигура не вписывается в картинку
		size_x = last_x - first_x
		size_y = last_y - first_y

		min_x = min(min_x, first_x, last_x)
		max_x = max(max_x, first_x, last_x)

		min_y = min(min_y, first_y, last_y)
		max_y = max(max_y, first_y, last_y)

		x_top   = random.randint(int(round(first_x + 0.2 * size_x)), int(round(last_x - 0.2 * size_x)))
		y_left_1  = random.randint(int(round(first_y + 0.1 * size_y)), int(round(last_y - 0.5 * size_y)))
		y_left_2  = random.randint(int(round(first_y + 0.5 * size_y)), int(round(last_y - 0.1 * size_y)))
		x_down  = random.randint(int(round(first_x + 0.2 * size_x)), int(round(last_x - 0.2 * size_x)))
		y_right_1 = random.randint(int(round(first_y + 0.5 * size_y)), int(round(last_y - 0.1 * size_y)))
		y_right_2 = random.randint(int(round(first_y + 0.1 * size_y)), int(round(last_y - 0.5 * size_y)))

		pts = np.array(((x_top, last_y), (last_x, y_right_1), (last_x, y_right_2),(x_down, first_y), (first_x,y_left_1), (first_x,y_left_2)), dtype=int)
		cv2.fillConvexPoly(overlay, pts, color)

		f_shapes = labelMe_class.Shapes(label, [[first_x, first_y], [last_x, last_y]], None, "rectangle", {})

		f_json_list.append(f_shapes.to_string_form())

	alpha = 1
	cv2.addWeighted(overlay, alpha, pic, 1 - alpha, 0, pic)
	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y],[max_x, max_y]], None, "rectangle", {})

	res = TreeRet(pic, f_json_list, [r_shapes.to_string_form()])
	return res

