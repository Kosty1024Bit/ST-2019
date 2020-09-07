
import cv2
import numpy as np
import numpy.random as npr
import random
import imutils

from itertools import tee
from common_file import labelMe_class

from common_file.return_class import RetClass

from common_file.return_class import TreeRet
import math

def get_random_color():
	b_int = npr.randint(0,256)
	g_int = npr.randint(0,256)
	r_int = npr.randint(0,256)
	return b_int, g_int, r_int


def contur_to_list_int_points(contur):
	list_points = []
	for point in contur:
		(x,y) = point[0]
		list_points.append([int(x),int(y)])
	return list_points

def add_vertical_pattern(img, label):
	color = np.random.randint(0,256,size = 3)
	(height, width, channel) = img.shape
	pattern_dist = int(width * 0.01)
	pattern_length = int(height * 0.04)
	segment_length = pattern_length // 8

	row_count = 0
# 	start_row_index = 0
	horizontal_shift = random.randint(0, pattern_dist)

	min_x = width
	max_x = 0

	min_y = height
	max_y = 0

	f_json_list = []

	for x in range(horizontal_shift, width, pattern_dist):
		if row_count % 4 == 0:
			vertical_shift = random.randint(0, pattern_length)

		if np.random.uniform() < 0.75:
			row_count += 1
			continue

		min_x_c = int(x)
		max_x_c = int(x+1)

		min_y_c = 0
		max_y_c = 0

		for y in range(0, height, pattern_length):
			if np.random.uniform() < 0.4:
				continue
			y1 = (vertical_shift + y) % height
			y2 = (vertical_shift + y + segment_length) % height
			y3 = (vertical_shift + y + 2 * segment_length)% height
			y3_step = (vertical_shift + y + 4 * segment_length)% height
			y4 = (vertical_shift + y + 5 * segment_length)% height
			y5 = (vertical_shift + y + 6 * segment_length)% height

			img[y1, x, :] = color
			img[y2, x, :] = color

			max_y_3_f = 0
			if y3_step != 0:
				for y_3_f in range(y3, height, y3_step):
					img[y_3_f, x] = color
					max_y_3_f = int(max(max_y_3_f, y_3_f))
			img[y4, x, :] = color
			img[y5, x, :] = color

			max_y_c = int(max(max_y_c, y1,y2,y3,y4,y5, max_y_3_f))

		row_count += 1

		f_shapes = labelMe_class.Shapes(label, [[min_x_c, min_y_c], [max_x_c, max_y_c]], None, "rectangle", {})
		f_json_list.append(f_shapes.to_string_form())

		min_x = min(min_x, min_x_c)
		max_x = max(max_x, max_x_c)

		min_y = min(min_y, min_y_c)
		max_y = max(max_y, max_y_c)

	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y],[max_x, max_y]], None, "rectangle", {})
	res = TreeRet(img, f_json_list, [r_shapes.to_string_form()])
	return res


def blurring(img, label):
	blur = cv2.bilateralFilter(img, 40, 100, 100)

	shapes = labelMe_class.Shapes(label, [[0, 0],[blur.shape[1]-1, blur.shape[0]-1]], None, "rectangle", {})
	res = TreeRet(blur, [shapes.to_string_form()], [shapes.to_string_form()])
	return res


def create_discoloration(image, label):
	img = image.copy()
	threshold = npr.randint(100, 140)
	new_intesity = npr.randint(200, 256)

	min_x = img.shape[1]
	min_y = img.shape[0]

	max_x = 0
	max_y = 0

	color = npr.randint(0, 6)
	if color == 0:
		for y in range(0, img.shape[0]):
			for x in range(0, img.shape[1]):
				if img[y,x][2] > threshold:
					img[y,x] = (0, 0, new_intesity)

					min_x = min(min_x, x)
					max_x = max(max_x, x)

					min_y = min(min_y, y)
					max_y = max(max_y, y)

	elif color == 1:
		for y in range(0, img.shape[0]):
			for x in range(0, img.shape[1]):
				if img[y,x][1] > threshold:
					img[y,x] = (0, new_intesity, 0)

					min_x = min(min_x, x)
					max_x = max(max_x, x)

					min_y = min(min_y, y)
					max_y = max(max_y, y)
	elif color == 2:
		for y in range(0, img.shape[0]):
			for x in range(0, img.shape[1]):
				if img[y,x][0] > threshold:
					img[y,x] = (new_intesity, 0, 0)

					min_x = min(min_x, x)
					max_x = max(max_x, x)

					min_y = min(min_y, y)
					max_y = max(max_y, y)
	else:
		b_int = npr.randint(new_intesity,256)
		g_int = npr.randint(new_intesity,256)
		r_int = npr.randint(new_intesity,256)
		for y in range(0, img.shape[0]):
			for x in range(0, img.shape[1]):
				if img[y,x][0] > threshold:
					img[y,x] = (b_int, g_int, r_int)

					min_x = min(min_x, x)
					max_x = max(max_x, x)

					min_y = min(min_y, y)
					max_y = max(max_y, y)

	shapes = labelMe_class.Shapes(label, [[min_x, min_y],[max_x, max_y]], None, "rectangle", {})
	res = TreeRet(img, [shapes.to_string_form()], [shapes.to_string_form()])
	return res


def triangulation(img, label):
	h,w,_ = img.shape
	grid_length = int(np.random.uniform(1.0 / 40, 1.0 / 25) * w)
	half_grid = grid_length // 2

	triangles = []

	for i in range(0,h,grid_length):
		for j in range(0,w,grid_length):
			pt1, pt2 = np.array([i,j]), np.array([i,min(j+ grid_length, w-1)])
			pt3 = np.array([min(i+half_grid, h-1),min(j+half_grid, w-1)])
			pt4, pt5 = np.array([min(i+grid_length,  h-1),j]), np.array([min(i+grid_length, h-1),min(j+grid_length, w-1)])


			pt1 = pt1[[1,0]]
			pt2 = pt2[[1,0]]
			pt3 = pt3[[1,0]]
			pt4 = pt4[[1,0]]
			pt5 = pt5[[1,0]]

			triangles.append(np.array([pt1,pt2,pt3]))
			triangles.append(np.array([pt1,pt4,pt3]))
			triangles.append(np.array([pt5,pt2,pt3]))
			triangles.append(np.array([pt5,pt4,pt3]))

	min_x = w
	max_x = 0

	min_y = h
	max_y = 0

	f_json_list = []

	for t in triangles:
		mid_pt = ((t[0] + t[1] + t[2])/3).astype(int)

		mid_pt = mid_pt[[1,0]]

		color = img[mid_pt[0], mid_pt[1],:]*0.85 + 0.05 * img[t[0,1], t[0,0], :] + 0.05 * img[t[1,1], t[1,0], :] + 0.05 * img[t[2,1], t[2,0], :]
		color = np.uint8(color)
		c = tuple(map(int, color))

		p = cv2.drawContours(img, [t], -1, c, -1)

		min_x_c = int(min(t[0,0],t[1,0],t[2,0]))
		max_x_c = int(max(t[0,0],t[1,0],t[2,0]))

		min_y_c = int(min(t[0,1],t[1,1],t[2,1]))
		max_y_c = int(max(t[0,1],t[1,1],t[2,1]))

		f_shapes = labelMe_class.Shapes(label, [[min_x_c, min_y_c], [max_x_c, max_y_c]], None, "rectangle", {})
		f_json_list.append(f_shapes.to_string_form())

		min_x = min(min_x, min_x_c)
		max_x = max(max_x, max_x_c)

		min_y = min(min_y, min_y_c)
		max_y = max(max_y, max_y_c)

	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y],[max_x, max_y]], None, "rectangle", {})
	res = TreeRet(p, f_json_list, [r_shapes.to_string_form()])
	return res


def add_random_patches(im, label, lo = 3, hi = 20):
	color = npr.randint(0, 6)
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contours.sort(key = len)
	patch_number = np.random.randint(lo,hi+1)
	b_int, g_int, r_int = get_random_color()

	p_json_list = []

	for i in range(patch_number):
		if color == 0:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (0,0,250), -1)
		elif color == 1:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (0,250,0), -1)
		elif color == 2:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (250,0,0), -1)
		else:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (b_int,g_int,r_int), -1)

		contour = contours[len(contours) - 1 - i]

		p_shapes = labelMe_class.Shapes(label, contur_to_list_int_points(contour), None, "polygon", {})
		p_json_list.append(p_shapes)

	res = RetClass(im, p_json_list)
	return res


def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

def point_two_line(x1_1,y1_1, x1_2,y1_2, x2_1,y2_1, x2_2,y2_2):
	# составляем формулы двух прямых

	x1_1 = float(x1_1)
	y1_1 = float(y1_1)
	x1_2 = float(x1_2)
	y1_2 = float(y1_2)
	x2_1 = float(x2_1)
	y2_1 = float(y2_1)
	x2_2 = float(x2_2)
	y2_2 = float(y2_2)

	xdiff = (x1_1 - x1_2, x2_1 - x2_2)
	ydiff = (y1_1 - y1_2, y2_1 - y2_2)

	div = det(xdiff, ydiff)
	if div == 0:
		return None

	d = (det([x1_1,y1_1],[x1_2, y1_2]), det([x2_1,y2_1],[x2_2, y2_2]))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div

	if (min(x1_1, x1_2) <= x <= max(x1_1, x1_2)) and (min(y1_1, y1_2) <= y <= max(y1_1, y1_2)):
		if(min(x2_1, x2_2) <= x <= max(x2_1, x2_2)) and (min(y2_1, y2_2) <= y <= max(y2_1, y2_2)):
			return (x,y)
	else:
		return None


#сделать ограничение именно на контур, а не на выпуклую фигуру (разбивка точек на внутр., наружн и границы (границы проверять на пересечения с другими сторонами))
def contntur_limitation(in_point_list, min_limit_x, min_limit_y, max_limit_x, max_limit_y):
	contur = []

	is_out_list = []

	for (x_f, y_f) in in_point_list:
		if(x_f < min_limit_x or x_f > max_limit_x or y_f < min_limit_y or y_f > max_limit_y):
			is_out_list.append(True)
		else:
			is_out_list.append(False)

	f_1_count = 0

	iter_first = iter(in_point_list)
	iter_first,iter_second = tee(iter_first)

	for (x,y) in iter_first:

		iter_first,iter_second = tee(iter_first)
		f_2_count = f_1_count + 1
		for (x2,y2) in iter_second:


			left_horizontal_limit  = point_two_line(x,y, x2,y2, min_limit_x, min_limit_y, min_limit_x, max_limit_y)
			right_horizontal_limit = point_two_line(x,y, x2,y2, max_limit_x, min_limit_y, max_limit_x, max_limit_y)

			bottom_vertical_limit  = point_two_line(x,y, x2,y2, min_limit_x, min_limit_y, max_limit_x, min_limit_y)
			top_vertical_limit 	   = point_two_line(x,y, x2,y2, min_limit_x, max_limit_y, max_limit_x, max_limit_y)

			if  left_horizontal_limit is not None:
				(x_p, y_p) = left_horizontal_limit
				contur.append([x_p,y_p])

			if right_horizontal_limit is not None:
				(x_p, y_p) = right_horizontal_limit
				contur.append([x_p,y_p])

			if bottom_vertical_limit is not None:
				(x_p, y_p) = bottom_vertical_limit
				contur.append([x_p,y_p])

			if top_vertical_limit is not None:
				(x_p, y_p) = top_vertical_limit
				contur.append([x_p,y_p])

			f_2_count += 1
		if(is_out_list[f_1_count] == False):
				contur.append([x,y])
						#print("ERROR POINT", f_1_count, " LiNE ", f_2_count ," \n", x, y, x2 ,y2, min_limit_x, max_limit_y, max_limit_x, max_limit_y)
		f_1_count += 1
	return contur

def add_shapes(im, label, lo = 2, hi = 5):
	img = im.copy()
	h, w, _ = img.shape

	# Find the darkest region of the image
	grid = (-1,-1)
	mean_shade = np.mean(img)

	x_step, y_step = int(w/6), int(h/4)
	for y in range(0, h, y_step):
		for x in range(0, w, x_step):
			new_shade = np.mean(img[y:y+y_step, x:x+x_step])
			if  new_shade <= mean_shade:
				mean_shade = new_shade
				grid = (x,y)

	f_json_list = []
	max_x = 0
	max_y = 0
	min_x = w
	min_y = h

	# Add shapes
	minLoc = (np.random.randint(grid[0], min(grid[0]+x_step, w)), np.random.randint(grid[1], min(grid[1]+x_step, h)))
	#minLoc = (np.random.randint(0.1 * w, 0.9 * w), np.random.randint(0.1 * h, 0.9 * h))
	num_shapes = np.random.randint(lo,hi+1)

	for i in range(num_shapes):
		stretch = np.random.randint(40, 100)
		diff1, diff2 = np.random.randint(-5,5), np.random.randint(-5,5)
		x1 = minLoc[0] + diff1 * stretch
		y1 = minLoc[1] + diff2 * stretch
		x2 = x1 + np.random.randint(1,12)/5 * diff1 * stretch
		y2 = y1 + np.random.randint(1,12)/5 * diff2 * stretch

		pts = np.array((minLoc, (x1, y1), (x2, y2)), dtype=int)
		contur = contntur_limitation([minLoc, [x1,y1], [x2,y2]]	, 0,0, w-1,h-1)

		(t_x1, t_y1) = contur[0]
		t_x1 = int(math.ceil(t_x1))
		t_y1 = int(math.ceil(t_y1))
		temp_min_x = t_x1
		temp_min_y = t_y1

		temp_max_x = t_x1
		temp_max_y = t_y1

		for (t_x,t_y) in contur:
			t_x = int(math.ceil(t_x))
			t_y = int(math.ceil(t_y))

			temp_min_x = min(t_x,temp_min_x)
			temp_min_y = min(t_y,temp_min_y)

			temp_max_x = max(t_x,temp_max_x)
			temp_max_y = max(t_y,temp_max_y)

		f_shapes = labelMe_class.Shapes(label, [[temp_min_x, temp_min_y], [temp_max_x, temp_max_y]], None, "rectangle", {})
		f_json_list.append(f_shapes.to_string_form())

		min_x = min(min_x, temp_min_x)
		max_x = max(max_x, temp_max_x)

		min_y = min(min_y, temp_min_y)
		max_y = max(max_y, temp_max_y)

		c1, c2, c3 = np.random.randint(0,50),np.random.randint(0,50),np.random.randint(0,50)
		cv2.fillConvexPoly(img, pts, color= (c1,c2,c3))

	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y], [max_x, max_y]], None, "rectangle", {})
	r_json_list = [r_shapes.to_string_form()]

	res = TreeRet(img, f_json_list, r_json_list)
	return res


def add_triangles(im, label, lo = 1, hi = 3):
	h, w, _ = im.shape
# 	colors = np.array((
#                    (250,206,135),
#                    (153,255, 255),
#                    (255, 203, 76)),dtype = int) #maybe expand this list of colors

	output = im.copy()
	overlay = im.copy()

	x_0, y_0 = np.random.randint(w), np.random.randint(h)
	x_1, y_1 = np.random.randint(w), np.random.randint(h)
	x_2, y_2 = np.random.randint(w), np.random.randint(h)
	pts = np.array(((x_0, y_0), (x_1, y_1), (x_2, y_2)), dtype=int)

	b_int, g_int, r_int = get_random_color()
	cv2.fillConvexPoly(overlay, pts, color= tuple([b_int, g_int, r_int]) )

	f_json_list = []
	max_x = max(x_0,x_1,x_2)
	max_y = max(y_0,y_1,y_2)
	min_x = min(x_0,x_1,x_2)
	min_y =	min(y_0,y_1,y_2)

	f_shapes = labelMe_class.Shapes(label, [[min_x, min_y], [max_x,max_y]], None, "rectangle", {})
	f_json_list.append(f_shapes.to_string_form())

	num_shapes = np.random.randint(lo, hi + 1)
	alpha = .95
	for i in range(num_shapes):
		x_1, y_1 = np.mean([x_1, x_0]) + np.random.randint(-60,60), np.mean([y_1,y_0])+ np.random.randint(-60,60)
		x_2, y_2 = np.mean([x_2, x_0]) + np.random.randint(-60,60), np.mean([y_2,y_0])+ np.random.randint(-60,60)

		pts = np.array(((x_0, y_0), (x_1, y_1), (x_2, y_2)), dtype=int)
		# if not is_random:
		# 	cv2.fillConvexPoly(overlay, pts, color= tuple([int(x) for x in colors[np.random.randint(3)]]) )

		b_int, g_int, r_int = get_random_color()
		cv2.fillConvexPoly(overlay, pts, color= tuple([b_int, g_int, r_int]) )

		temp_min_x = min(x_0,x_1,x_2)
		temp_min_y = min(y_0,y_1,y_2)

		temp_max_x = max(x_0,x_1,x_2)
		temp_max_y = max(y_0,y_1,y_2)

		f_shapes = labelMe_class.Shapes(label, [[temp_min_x, temp_min_y], [temp_max_x, temp_max_y]], None, "rectangle", {})
		f_json_list.append(f_shapes.to_string_form())

		min_x = min(min_x, temp_min_x)
		max_x = max(max_x, temp_max_x)

		min_y = min(min_y, temp_min_y)
		max_y = max(max_y, temp_max_y)


	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y], [max_x, max_y]], None, "rectangle", {})
	r_json_list = [r_shapes.to_string_form()]

	res = TreeRet(output, f_json_list, r_json_list)
	return res


#this function returns a color blend of the overlay and the original image. angle = 0 means the overlay
#will fade down and angle = 180 will cause fade up:
##################################
def gradient(img, overlay, angle = 0):
	alpha = 1

	img = imutils.rotate_bound(img, angle)
	overlay = imutils.rotate_bound(overlay, angle)

	for x in range(1, img.shape[0],10 ):
		cv2.addWeighted(overlay[x:x+10,:,:], alpha, img[x:x+10,:,:] , 1 - alpha, 0, img[x:x+10,:,:])
		alpha *= .98


	img = imutils.rotate_bound(img, -1 * angle)
	return img

#########
def color_blend(img, overlay1, overlay2, angle = 0):
	alpha = 1

	img = imutils.rotate_bound(img, angle)
	overlay1 = imutils.rotate_bound(overlay1, angle)
	overlay2 = imutils.rotate_bound(overlay2, angle)

	for x in range(1, overlay1.shape[0],10 ):
		cv2.addWeighted(overlay1[x:x+10,:,:], alpha, overlay2[x:x+10,:,:] , 1 - alpha, 0, img[x:x+10,:,:])
		alpha *= .95

	img = imutils.rotate_bound(img, -1 * angle)

# 	cv2.imshow ("overlay1", overlay1)
# 	cv2.imshow ("overlay2", overlay2)
# 	cv2.imshow ("img",img)
# 	cv2.waitKey()


	return img
#############################


def add_shaders(im, label, lo = 1, hi = 3):
	angles = np.array([0,90,180,270])

	h,w,_ = im.shape
	output = im.copy()
	overlay1 = im.copy()
	overlay2 = im.copy()

	#####big shaders in forms of n-gons
	num_shapes = np.random.randint(lo,hi+1)

	f_json_list = []
	max_x = 0
	max_y = 0
	min_x = w
	min_y = h

	for i in range(num_shapes):
		x_0, y_0 = np.random.randint(w), np.random.randint(h)
		x_1, y_1 = np.random.randint(-300,w+300), np.random.randint(-300,h+300)
		x_2, y_2 = np.random.randint(-300,w+300), np.random.randint(-300,h+300)

		pts = np.array(((x_0, y_0), (x_1, y_1), (x_2, y_2)), dtype=int)

		temp_max_x = max(x_0,x_1,x_2)
		temp_max_y = max(y_0,y_1,y_2)
		temp_min_x = min(x_0,x_1,x_2)
		temp_min_y = min(y_0,y_1,y_2)

		extra_n = np.random.randint(4)

		for i in range(extra_n): #extra number of points to make an n_gon
			temp_ran_x = np.random.randint(-300,h+300)
			temp_ran_y = np.random.randint(-300,w+300)
			pts = np.append(pts, [[temp_ran_x, temp_ran_y]], axis = 0)


		contur = contntur_limitation(pts, 0,0, w-1,h-1)

		(t_x1, t_y1) = contur[0]
		t_x1 = int(math.ceil(t_x1))
		t_y1 = int(math.ceil(t_y1))
		temp_min_x = t_x1
		temp_min_y = t_y1

		temp_max_x = t_x1
		temp_max_y = t_y1

		for (t_x,t_y) in contur:
			t_x = int(math.ceil(t_x))
			t_y = int(math.ceil(t_y))

			temp_min_x = min(t_x,temp_min_x)
			temp_min_y = min(t_y,temp_min_y)

			temp_max_x = max(t_x,temp_max_x)
			temp_max_y = max(t_y,temp_max_y)

		f_shapes = labelMe_class.Shapes(label, [[temp_min_x, temp_min_y], [temp_max_x, temp_max_y]], None, "rectangle", {})
		f_json_list.append(f_shapes.to_string_form())

		min_x = min(min_x, temp_min_x)
		max_x = max(max_x, temp_max_x)

		min_y = min(min_y, temp_min_y)
		max_y = max(max_y, temp_max_y)

		alpha = 1

		colors = np.empty([2, 3])
		start_x = min(max(0, x_0), h-1)
		start_y = min(max(0, y_0), w-1)

		colors[0, :] = im[start_x, start_y,:] + npr.randint(-30, 30, size = [3])
		mid_x = (x_1+x_2)//2
		mid_y = (y_1+y_2)//2

		mid_x = min(max(0, mid_x), h-1)
		mid_y = min(max(0, mid_y), w-1)

		colors[1, :] = im[mid_x,mid_y,:] + npr.randint(-30, 30, size = [3])

		colors = np.clip(colors, a_min = 0, a_max = 255)


		#f_shapes = labelMe_class.Shapes("2", [[start_x, start_y], [mid_x, mid_y]], None, "rectangle", {})
		#f_json_list.append(f_shapes.to_string_form())

		# colors[0,:] = npr.randint(0, 256, size = 3)
		# colors[1,:] = colors[0,:] + npr.randint(0, 100, size = 3)
		# colors[1,:] = np.clip(colors[1,:], 0, 255)


		cv2.fillConvexPoly(overlay1, pts, color= tuple([int(x) for x in colors[0]]) )
		cv2.fillConvexPoly(overlay2, pts, color= tuple([int(x) for x in colors[1]]) )


	############
	a1, a2 = random.choice(angles), random.choice(angles)

	t_img = gradient(output, color_blend(im, overlay1, overlay2, a1), a2)

	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y], [max_x, max_y]], None, "rectangle", {})
	r_json_list = [r_shapes.to_string_form()]

	res = TreeRet(t_img, f_json_list, r_json_list)
	return res