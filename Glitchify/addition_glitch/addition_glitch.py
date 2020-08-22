'''
	The input is a python [Height * Width * 3] array, which is a picture to add glitches.
	Notice that the input value should range from 0 to 255.
	The output is a python [Height * Width * 3] array, which is a picture with glitch added.
'''

import cv2
import numpy.random as random
import numpy as np

from common_file import labelMe_class
from common_file.tree_return_class import TreeRet


def intersection_check(list_coordinate, point1, point2):
	for (point_l_1, point_l_2) in list_coordinate:
		if not ((point1[1] < point_l_2[1] or point2[1] > point_l_1[1]) and (point2[0] < point_l_1[0] or point1[0] > point_l_2[0])):
				return True
	return False

def to_int(value):
	return int(round(value))

def get_random_color():
	b_int = random.randint(0,256)
	g_int = random.randint(0,256)
	r_int = random.randint(0,256)
	return b_int, g_int, r_int


def white_square(picture, label, allow_intersections, lo = 2, hi = 15):
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
	count_fail = 0

	for i in range(number_of_patches):

		(red, green, blue) = random.randint(240,256, size = 3) #верхняя граница не входит
		red   = int(red)
		green = int(green)
		blue  = int(blue)

		color = [blue, green, red]

		is_intersection = True

		count = 0
		while(is_intersection):
			forfeit = min((600 / 10000 * count), 300)
			forfeit2 = (30 / 10000 * count)

			(first_x,first_y) = random.random(2)
			(size_x, size_y) =  random.random(2)

			size_x = to_int(size_x * (300 - forfeit) + 80 - forfeit2)
			size_y = to_int(size_y * (300 - forfeit) + 80 - forfeit2)

			first_y = to_int(first_y * (height*0.7) + height* 0.1) #(int(height* 0.1), int(height*0.8))
			first_x = to_int(first_x * (width *0.7) + width * 0.1)

			last_y = min(first_y + size_y, height - 1)
			last_x = min(first_x + size_x, width - 1)

			#на случай, если фигура не вписывается в картинку
			size_x = last_x - first_x
			size_y = last_y - first_y

			if(size_x > 2 * size_y or size_y > 2 * size_x):
				#print ("cont")
				continue

			count += 1

			if allow_intersections:
				break
			if(count < 10000):
				is_intersection = intersection_check(list_coordinate_rectangle, [first_x, first_y], [last_x, last_y])
			else:
				is_intersection = False
				count_fail += 1

		list_coordinate_rectangle.append([[first_x,first_y], [last_x, last_y]])

		min_x = min(min_x, first_x, last_x)
		max_x = max(max_x, first_x, last_x)

		min_y = min(min_y, first_y, last_y)
		max_y = max(max_y, first_y, last_y)

		rotat = random.randint(0,2)

		if rotat == 0:
			x_top   = random.randint(first_x, to_int(last_x - 0.5 * size_x))
			y_left  = random.randint(first_y, to_int(last_y - 0.5 * size_y))
			x_down  = min(last_x + first_x - x_top  + random.randint(-to_int(size_x * 0.1)-1, to_int(size_x * 0.1)), last_x)
			y_right = min(last_y + first_y - y_left + random.randint(-to_int(size_y * 0.1)-1, to_int(size_y * 0.1)), last_y)
		else :
			x_top   = random.randint(to_int(first_x + 0.5 * size_x), last_x)
			y_left  = random.randint(to_int(first_y + 0.5 * size_y), last_y)
			x_down  = max(last_x + first_x - x_top  + random.randint(-to_int(size_x * 0.1), to_int(size_x * 0.1)+1) , first_x)
			y_right = max(last_y + first_y - y_left + random.randint(-to_int(size_y * 0.1), to_int(size_y * 0.1)+1), first_y)

		pts = np.array(((x_top, last_y), (last_x, y_right), (x_down, first_y), (first_x,y_left)), dtype=int)
		cv2.fillConvexPoly(overlay, pts, color)

		f_shapes = labelMe_class.Shapes(label, [[first_x, first_y], [last_x, last_y]], None, "rectangle", {})

		f_json_list.append(f_shapes.to_string_form())

	if not count_fail == 0:
		 print (count_fail)

	alpha = 1
	cv2.addWeighted(overlay, alpha, pic, 1 - alpha, 0, pic)
	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y],[max_x, max_y]], None, "rectangle", {})

	res = TreeRet(pic, f_json_list, [r_shapes.to_string_form()])
	return res



def black_tree(picture, label, allow_intersections, lo = 2, hi = 15):
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

	count_fail = 0
	for i in range(number_of_patches):

		(red, green, blue) = random.randint(0, 21, size = 3) #верхняя граница не входит
		red   = int(red)
		green = int(green)
		blue  = int(blue)
		color = [blue, green, red]

		is_intersection = True

		count = 0
		while(is_intersection):
			forfeit = min((300 / 10000 * count), 100)
			forfeit2 = (20 / 10000 * count)

			(first_y,first_x) = random.random(2)
			(size_x, size_y) =  random.random(2)

			size_y = to_int(size_y * (150 - forfeit) + 40 - forfeit2)
			size_x = to_int(size_x * (100 - forfeit) + 40 - forfeit2)

			first_y = to_int(first_y * (height*0.7) + height* 0.1) #(int(height* 0.1), int(height*0.8))
			first_x = to_int(first_x * (width *0.7) + width * 0.1)


			last_y = min(first_y + size_y, height - 1)
			last_x = min(first_x + size_x, width - 1)

			#на случай, если фигура не вписывается в картинку
			size_y = last_y - first_y
			size_x = last_x - first_x

			if size_x > size_y or size_y < 20 or size_x < 20:
				continue

			count += 1

			if allow_intersections:
				break
			if(count < 10000):
				is_intersection = intersection_check(list_coordinate_rectangle, [first_x, first_y], [last_x, last_y])
			else:
				is_intersection = False
				count_fail += 1

		list_coordinate_rectangle.append([[first_x,first_y], [last_x, last_y]])

		min_x = min(min_x, first_x, last_x)
		max_x = max(max_x, first_x, last_x)

		min_y = min(min_y, first_y, last_y)
		max_y = max(max_y, first_y, last_y)

		x_top     = random.randint(to_int(first_x + 0.2 * size_x), to_int(last_x - 0.2 * size_x))
		y_left_1  = random.randint(to_int(first_y + 0.1 * size_y), to_int(last_y - 0.5 * size_y))
		y_left_2  = random.randint(to_int(first_y + 0.5 * size_y), to_int(last_y - 0.1 * size_y))
		x_down    = random.randint(to_int(first_x + 0.2 * size_x), to_int(last_x - 0.2 * size_x))
		y_right_1 = random.randint(to_int(first_y + 0.5 * size_y), to_int(last_y - 0.1 * size_y))
		y_right_2 = random.randint(to_int(first_y + 0.1 * size_y), to_int(last_y - 0.5 * size_y))

		pts = np.array(((x_top, last_y), (last_x, y_right_1), (last_x, y_right_2),(x_down, first_y), (first_x,y_left_1), (first_x,y_left_2)), dtype=int)
		cv2.fillConvexPoly(overlay, pts, color)

		f_shapes = labelMe_class.Shapes(label, [[first_x, first_y], [last_x, last_y]], None, "rectangle", {})

		f_json_list.append(f_shapes.to_string_form())

	if not count_fail == 0:
		 print (count_fail)
	alpha = 1
	cv2.addWeighted(overlay, alpha, pic, 1 - alpha, 0, pic)
	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y],[max_x, max_y]], None, "rectangle", {})

	res = TreeRet(pic, f_json_list, [r_shapes.to_string_form()])
	return res



def add_random_patches_mods(img, label, lo = 3, hi = 20):
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	im = img.copy()

	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contours.sort(key = len)
	patch_number = np.random.randint(lo, hi+1)
	print (patch_number)
	min_x = imgray.shape[1]
	max_x = 0

	min_y = imgray.shape[0]
	max_y = 0

	f_json_list = []

	offset = 1

	for i in range(patch_number):
		color = random.randint(0, 7)
		intens_color =  np.random.randint(128,256)
		if color == 0:
			cv2.drawContours(img, contours,len(contours) - 1 - i - offset , (0,0,intens_color), -1)
		elif color == 1:
			cv2.drawContours(img, contours,len(contours) - 1 - i - offset , (0,intens_color,0), -1)
		elif color == 2:
			cv2.drawContours(img, contours,len(contours) - 1 - i - offset , (intens_color,0,0), -1)
		elif color == 3:

			ret, im_gray_mask = cv2.threshold(imgray, 255, 255, 0)

			# 	cv2.imshow ("img",im_gray_mask)
			# 	cv2.waitKey()

			cv2.drawContours(im_gray_mask, contours, len(contours) - 1 - i - offset , 255, -1)

			count_pixel = 0

			sum_r_color = 0
			sum_g_color = 0
			sum_b_color = 0

			h,w, _ = img.shape
			for y in range(0, h):
				for x in range(0, w):
					if im_gray_mask[y,x] == 255:
						count_pixel += 1
						(b,g,r) = im[y,x]
						sum_b_color += b
						sum_g_color += g
						sum_r_color += r

			if count_pixel != 0:
				sum_b_color /= count_pixel
				sum_g_color /= count_pixel
				sum_r_color /= count_pixel
			else:
				print("Div zero")
			cv2.drawContours(img, contours,len(contours) - 1 - i - offset , (sum_b_color,sum_g_color,sum_r_color), -1)

		else:
			b_int, g_int, r_int = get_random_color()
			cv2.drawContours(img, contours,len(contours) - 1 - i - offset , (b_int,g_int,r_int), -1)

		contour = contours[len(contours) - 1 - i - offset]

		(x,y) = contour[0,0]
		min_x_c = x
		max_x_c = x

		min_y_c = y
		max_y_c = y
		for point in contour:
			(x_c, y_c) = point[0]

			min_x_c = int(min(min_x_c, x_c))
			max_x_c = int(max(max_x_c, x_c))

			min_y_c = int(min(min_y_c, y_c))
			max_y_c = int(max(max_y_c, y_c))

		f_shapes = labelMe_class.Shapes(label, [[min_x_c, min_y_c], [max_x_c, max_y_c]], None, "rectangle", {})
		f_json_list.append(f_shapes.to_string_form())

		min_x = min(min_x, min_x_c)
		max_x = max(max_x, max_x_c)

		min_y = min(min_y, min_y_c)
		max_y = max(max_y, max_y_c)

	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y],[max_x, max_y]], None, "rectangle", {})
	res = TreeRet(img, f_json_list, [r_shapes.to_string_form()])

	return res

