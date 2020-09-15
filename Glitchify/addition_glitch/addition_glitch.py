'''
	The input is a python [Height * Width * 3] array, which is a picture to add glitches.
	Notice that the input value should range from 0 to 255.
	The output is a python [Height * Width * 3] array, which is a picture with glitch added.
'''

import cv2
import numpy.random as random
import numpy as np
import math

from common_file import labelMe_class
from common_file.return_class import RetClass

def check_val(value):
	if value > 255:
		value = 255
	if value < 0:
		value = 0
	return value


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


def np_array_to_list_int_points(arr):
	list_points = []
	for (x,y) in arr:
		list_points.append([int(x),int(y)])
	return list_points

def contur_to_list_int_points(contur):
	list_points = []
	for point in contur:
		(x,y) = point[0]
		list_points.append([int(x),int(y)])
	return list_points

# point to the left of the vector
def check_point_in_counter(mask, point_check):
	if 0 <= point_check[0] < mask.shape[1] and  0 <= point_check[1] < mask.shape[0]:
		if mask[point_check[1],point_check[0]] == 255:
			return True

	return False

def intensity_blur_in_contoure(img, contour, min_value, max_value):

	mask = np.zeros((img.shape[0],img.shape[1], 1), np.uint8)
	cv2.drawContours(mask, [contour], 0 , 255, -1)

	x_min, y_min, w, h = cv2.boundingRect (contour)
	x_max = x_min + w
	y_max = y_min + h

	for y in range(y_min, y_max):
		for x in range(x_min, x_max):
			if check_point_in_counter(mask, (x,y)):
				rand_color = random.randint(0,3)
				rand_operation = random.randint(-1,2)
				rand_value = random.randint(min_value, max_value+1)
				img[y,x][rand_color] = check_val(img[y,x][rand_color] + rand_operation * rand_value)
	return img


def add_spots(overlay, contour, fill_percentage):
	fill_percentage_now = 0

	mask = np.zeros((overlay.shape[0],overlay.shape[1], 1), np.uint8)
	cv2.drawContours(mask, [contour], 0 , 255, -1)

	x_min, y_min, w, h = cv2.boundingRect (contour)
	x_max = x_min + w
	y_max = y_min + h

	area =  cv2.contourArea(contour)
	area_now = 0


	count = 0
	while(fill_percentage_now < fill_percentage):

		if count == 100000:
			print("tired of writing circles")
			break
		count +=1

		radius = random.randint(1, 6)

		cycle_point = (random.randint(x_min, x_max), random.randint(y_min, y_max))

		#####################################################
		color = (0, 255, 255)
		#######################################################
		alpha = 0.4

		for y in range(-radius, radius+1):
			for x in range(-radius, radius+1):
				if math.sqrt(y**2 + x**2) <= radius:
					if check_point_in_counter(mask, (cycle_point[0]+x, cycle_point[1]+y)):
						if not all(overlay[cycle_point[1]+y, cycle_point[0]+x] == color):
							pic_color = overlay[cycle_point[1]+y, cycle_point[0]+x]
							result_color = (to_int(alpha * color[0] + (1 - alpha) * pic_color[0]), to_int(alpha * color[1] + (1 - alpha) * pic_color[1]), to_int(alpha * color[2] + (1 - alpha) * pic_color[2]))

							overlay[cycle_point[1]+y, cycle_point[0]+x] = result_color
							area_now += 1

		fill_percentage_now = area_now/area




def white_square(picture, label, allow_intersections, fill_percentage, lo = 2, hi = 15):
	height = picture.shape[0]
	width = picture.shape[1]
	number_of_patches = random.randint(lo,hi+1)

	overlay = picture.copy()
	pic = picture.copy()
	p_json_list = []

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

		orientation = random.randint(0,2)

		if orientation == 0:
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

##############################################################################
		intensity_blur_in_contoure(overlay, pts, -5, 5)

		add_spots(overlay, pts, fill_percentage)

		p_shapes = labelMe_class.Shapes(label, np_array_to_list_int_points(pts), None, "polygon", {})
		p_json_list.append(p_shapes)



	if not count_fail == 0:
		 print (count_fail)

	alpha = 1
	cv2.addWeighted(overlay, alpha, pic, 1 - alpha, 0, pic)


	res = RetClass(pic, p_json_list)
	return res

def create_tree(point_1, point_2):
	(first_x, first_y) = point_1
	(last_x, last_y) = point_2

	conturs = np.array([[1, 2]], dtype=int)

	#1 divided by number = probability of drawing a tree stump
	is_stump = random.randint(0, 10) # 10%

	w_05 = to_int((last_x - first_x)/2)
	h = last_y - first_y

	if (is_stump == 0):
		if (w_05 > 20):
			ps1_x = random.randint(to_int(3/4*w_05), w_05-4)
		else:
			ps1_x = random.randint(to_int(1/2*w_05), w_05-2)

		ps2_y = random.randint(3 + to_int(0.05 * h), to_int(0.1 * h)+4)

		conturs = np.array(((last_x - ps1_x, last_y - ps2_y), (last_x - ps1_x, last_y)), dtype=int)

		conturs = np.append(conturs, [[first_x + ps1_x, last_y]], axis = 0)
		conturs = np.append(conturs, [[first_x + ps1_x, last_y - ps2_y]], axis = 0)

	else:
		nps_x = random.randint(to_int(1/2*w_05), w_05+1)

		conturs = np.array(((last_x - nps_x, last_y),(first_x + nps_x, last_y)), dtype=int)


	point1_y = random.randint(to_int(0.1 * h), to_int(0.3 * h)+1)
	point2_y = random.randint(to_int(0.3 * h), to_int(0.35 * h)+1)

	point3_y = random.randint(to_int(0.45 * h), to_int(0.5 * h)+1)
	point3_x = random.randint(0, to_int(0.5 * w_05)+1)

	point4_y = random.randint(to_int(0.5 * h), to_int(0.55 * h)+2)
	point4_x = random.randint(0, to_int(0.5 * w_05)+1)

	point5_y = random.randint(to_int(0.55 * h), to_int(0.7 * h)+1)
	point6_y = random.randint(to_int(0.7 * h), to_int(0.9 * h)+1)

	point7_x = random.randint(to_int(0.7 * w_05), w_05+1)

	conturs = np.append(conturs, [[first_x,				 last_y - point1_y]], axis = 0)
	conturs = np.append(conturs, [[first_x,				 last_y - point2_y]], axis = 0)
	conturs = np.append(conturs, [[first_x + point3_x,	 last_y - point3_y]], axis = 0)
	conturs = np.append(conturs, [[first_x + point4_x,	 last_y - point4_y]], axis = 0)
	conturs = np.append(conturs, [[first_x, 			 last_y - point5_y]], axis = 0)
	conturs = np.append(conturs, [[first_x,				 last_y - point6_y]], axis = 0)
	conturs = np.append(conturs, [[first_x + point7_x,   first_y]]			, axis = 0)


	#symmetrical from the middle
	conturs = np.append(conturs, [[last_x - point7_x,    first_y]]			, axis = 0)
	conturs = np.append(conturs, [[last_x ,				 last_y - point6_y]], axis = 0)
	conturs = np.append(conturs, [[last_x , 			 last_y - point5_y]], axis = 0)
	conturs = np.append(conturs, [[last_x  - point4_x,	 last_y - point4_y]], axis = 0)
	conturs = np.append(conturs, [[last_x  - point3_x,	 last_y - point3_y]], axis = 0)
	conturs = np.append(conturs, [[last_x ,				 last_y - point2_y]], axis = 0)
	conturs = np.append(conturs, [[last_x ,				 last_y - point1_y]], axis = 0)

	return conturs


def black_tree(picture, label, allow_intersections, lo = 2, hi = 15):
	height = picture.shape[0]
	width = picture.shape[1]

	number_of_patches = random.randint(lo,hi+1)

	overlay = picture.copy()
	pic = picture.copy()
	p_json_list = []

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

		contur = create_tree([first_x, first_y], [last_x, last_y])

		cv2.fillConvexPoly(overlay, contur, color)

		p_shapes = labelMe_class.Shapes(label, np_array_to_list_int_points(contur), None, "polygon", {})

		p_json_list.append(p_shapes)

	if not count_fail == 0:
		 print (count_fail)
	alpha = 1
	cv2.addWeighted(overlay, alpha, pic, 1 - alpha, 0, pic)

	res = RetClass(pic, p_json_list)
	return res



def add_random_patches_mods(img, label, lo = 3, hi = 10):
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	im = img.copy()

	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contours.sort(key = len)
	patch_number = np.random.randint(lo, hi+1)
	print (patch_number)

	p_json_list = []

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

		p_shapes = labelMe_class.Shapes(label, contur_to_list_int_points(contour), None, "polygon", {})
		p_json_list.append(p_shapes)

	res = RetClass(img, p_json_list)
	return res


def color_cast(picture, label, allow_intersections, lo = 64, hi = 127):
	pic = picture.copy()
	h, w, _ = picture.shape

	chanal = random.randint(0,7)
	# может 128 ?
	rand_minus_color_value, rand_minus_color_value2 = random.randint(lo, hi+1, size = 2)

	#2/7 портит 1 канал, 1/7 портит 2
	if chanal == 0 or chanal == 1:
		for y in range(0, h):
				for x in range(0, w):
					pic[y,x][0] = check_val(pic[y,x][0] - rand_minus_color_value)

	elif chanal == 2 or chanal == 3:
		for y in range(0, h):
				for x in range(0, w):
					pic[y,x][1] = check_val(pic[y,x][1] - rand_minus_color_value)

	elif chanal == 4 or chanal == 5:
		for y in range(0, h):
				for x in range(0, w):
					pic[y,x][2] = check_val(pic[y,x][2] - rand_minus_color_value)

	else:
		list_chanal = [0,1,2]
		chanal1 = random.choice(list_chanal)
		list_chanal.remove(chanal1)
		chanal2 = random.choice(list_chanal)

		for y in range(0, h):
				for x in range(0, w):
					pic[y,x][chanal1] = check_val(pic[y,x][chanal1] - rand_minus_color_value)
					pic[y,x][chanal2] = check_val(pic[y,x][chanal2] - rand_minus_color_value2)


	p_shapes = labelMe_class.Shapes(label, [[0, 0],[w-1, 0],[w-1, h-1],[0, h-1]], None, "polygon", {})

	res = RetClass(pic, [p_shapes])
	return res


def create_discoloration_new(image, label):
	img = image.copy()
	threshold = random.randint(100, 140)
	new_intesity = random.randint(200, 256)

	imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, im_gray_mask = cv2.threshold(imgray, 255, 255, 0)


	color = random.randint(0, 6)
	if color == 0:
		for y in range(0, img.shape[0]):
			for x in range(0, img.shape[1]):
				if img[y,x][2] > threshold:
					img[y,x] = (0, 0, new_intesity)
					im_gray_mask[y,x] = 255

	elif color == 1:
		for y in range(0, img.shape[0]):
			for x in range(0, img.shape[1]):
				if img[y,x][1] > threshold:
					img[y,x] = (0, new_intesity, 0)
					im_gray_mask[y,x] = 255

	elif color == 2:
		for y in range(0, img.shape[0]):
			for x in range(0, img.shape[1]):
				if img[y,x][0] > threshold:
					img[y,x] = (new_intesity, 0, 0)
					im_gray_mask[y,x] = 255

	else:
		b_int, g_int, r_int = random.randint(200, 256, size = 3)

		for y in range(0, img.shape[0]):
			for x in range(0, img.shape[1]):
				if img[y,x][0] > threshold:
					img[y,x] = (b_int, g_int, r_int)
					im_gray_mask[y,x] = 255

	contours, hierarchy = cv2.findContours(im_gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	p_list = []

	for contour in contours:
		p_list.append(labelMe_class.Shapes(label, contur_to_list_int_points(contour), None, "polygon", {}))

	res = RetClass(img, p_list)
	return res

