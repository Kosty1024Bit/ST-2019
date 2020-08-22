import matplotlib.pyplot as plt
import cv2
import os
from functools import partial
import numpy as np
import numpy.random as npr
import sys
import argparse
from desktop_glitch.desktop_glitch_one import *
from desktop_glitch.desktop_glitch_two import create_desktop_glitch_two
import random
import imutils
import ou_glitch.ou_glitch as og
from stuttering.stuttering import produce_stuttering
from line_pixelation.line_pixelation import line_pixelation
from addition_glitch import addition_glitch

import json
from itertools import tee
from common_file import labelMe_class
from common_file.tree_return_class import TreeRet
import math
import time

def get_random_color():
	b_int = npr.randint(0,256)
	g_int = npr.randint(0,256)
	r_int = npr.randint(0,256)
	return b_int, g_int, r_int

def add_vertical_pattern(img, label):
	color = np.random.randint(0,256,size = 3)
	(height, width, channel) = img.shape
	pattern_dist = int(width * 0.01)
	pattern_length = int(height * 0.04)
	segment_length = pattern_length // 8

	row_count = 0
	start_row_index = 0
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

	shapes = labelMe_class.Shapes(label, [[0, 0],[blur.shape[1], blur.shape[0]]], None, "rectangle", {})
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

	min_x = imgray.shape[1]
	max_x = 0

	min_y = imgray.shape[0]
	max_y = 0

	f_json_list = []

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
	res = TreeRet(im, f_json_list, [r_shapes.to_string_form()])

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

def write_files(original_img, img, is_margin_specified, filename, out, is_video, append_to_arr):
	if append_to_arr:
		if not is_output_resized:
			X_orig_list.append(original_img)
		else:
			original_img = cv2.resize(original_img, (new_width, new_height))
			X_orig_list.append(original_img)

	if is_margin_specified:
		original_img[x0:x1, y0:y1, :] = img
	else:
		original_img = img

	if not is_video:
		if not is_output_resized:
			cv2.imwrite(filename, original_img)
		else:
			original_img = cv2.resize(original_img ,(new_width, new_height))
			cv2.imwrite(filename, original_img)
	else:
		if not is_output_resized:
			out.write(original_img)
		else:
			original_img = cv2.resize(original_img, (new_width, new_height))
			out.write(original_img)


	if append_to_arr:
		if not is_output_resized:
			X_glitched_list.append(original_img)
		else:
			original_img = cv2.resize(original_img ,(new_width, new_height))
			X_glitched_list.append(original_img)

#add code
def write_full_json_files(f_json, is_f_json, original_name_file, filename, img):
	if is_f_json:
		#write json with adres filename
		write_f_json = labelMe_class.Json("0.0.0 version", {}, f_json, original_name_file, None, img.shape[0], img.shape[1])
		with open(filename + "_full.json", "w") as write_file:
			json.dump(write_f_json.to_string_form(), write_file, indent=4)

def write_region_json_files(r_json, is_r_json, original_name_file, filename, img):
	if is_r_json:
		#write json with adres filename
		write_r_json = labelMe_class.Json("0.0.0 version", {}, r_json, original_name_file, None, img.shape[0], img.shape[1])
		with open(filename + "_region.json", "w") as write_file:
			json.dump(write_r_json.to_string_form(), write_file, indent=4)



def is_video_file(filename):
	video_file_extensions = (
	'.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
	'.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
	'.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
	'.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
	'.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
	'.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
	'.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
	'.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
	'.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
	'.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
	'.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
	'.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
	'.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
	'.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
	'.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
	'.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
	'.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
	'.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
	'.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
	'.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
	'.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
	'.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
	'.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
	'.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg','.vem', '.vep', '.vf', '.vft',
	'.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
	'.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
	'.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
	'.zm1', '.zm2', '.zm3', '.zmv'  )

	if filename.endswith((video_file_extensions)):
		return True
	else:
		return False



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	global X_orig_list, X_glitched_list
	X_orig_list, X_glitched_list = [], []


	# Values
	parser.add_argument('-o', '--output', dest='output_foldername')
	parser.add_argument('-i', '--input', dest='input_foldername')
	parser.add_argument('-t', '--type', dest='glitch_type')
	parser.add_argument('-lo', dest='arg1')
	parser.add_argument('-hi', dest='arg2')

	parser.add_argument('-x0', dest = 'x0')
	parser.add_argument('-y0', dest = 'y0')
	parser.add_argument('-x1', dest = 'x1')
	parser.add_argument('-y1', dest = 'y1')
	parser.add_argument('-interval', dest = 'interval', default=10)

	parser.add_argument('-ot', '--output_type', dest = 'output_type', default= 'image')
	parser.add_argument('-save_normal_frames', dest= 'save_normal_frames', default = 'False')
	parser.add_argument('-output_array', dest = 'output_array', default = 'False')
	parser.add_argument('-is_output_resized', dest = 'resize_output', default = 'False')
	parser.add_argument('-new_height', dest = 'new_height', default = 224)
	parser.add_argument('-new_width', dest = 'new_width', default = 224)

#my add code
	parser.add_argument('-fj', '--full_json', dest = 'full_json')
	parser.add_argument('-rj', '--region_json', dest = 'region_json')

	parser.add_argument('-ofj', '--output_full_json', dest = 'output_foldername_full_json')
	parser.add_argument('-orj', '--output_region_json', dest = 'output_foldername_region_json')

	parser.add_argument('-bf', '--boolean_flag', dest = 'boolean_flag')


	options = parser.parse_args()
	global arg1, arg2, x0, y0, x1, y1, is_output_resized, new_height, new_width

	is_bound_specified = False
	is_margin_specified = False
	is_video = False
	output_array = False
	is_output_resized = False
	interval = int(options.interval)
	new_height = 224
	new_width = 224

#add code
	is_full_json = False
	is_region_json = False

	if options.resize_output == 'True' or options.resize_output == 'true':
		is_output_resized = True
		new_height = int(options.new_height)
		new_width = int(options.new_width)

	if options.boolean_flag == 'True' or options.boolean_flag == 'true':
		bool_flag = True
	else:
		bool_flag = False

#было закоменчено, видимо не закончено
	# if options.output_type == 'video' or options.output_type == 'Video':
	# 	is_video = True
	is_video = False

	if options.arg1 is not None and options.arg2 is not None:
		is_bound_specified = True
		arg1 = int(options.arg1)
		arg2 = int(options.arg2)


	if options.x0 is not None and options.y0 is not None and options.x1 is not None and options.y1 is not None:
		is_margin_specified = True
		x0 = int(options.x0)
		y0 = int(options.y0)
		x1 = int(options.x1)
		y1 = int(options.y1)


	if options.output_array == 'True' or options.output_array == 'true':
		output_array = True

	count = 0

	if options.input_foldername is None:
		print("Please specify input and output folder name")

	if options.output_foldername is None:
		options.output_foldername = "new_output_folder"

	if not os.path.isdir(options.output_foldername):
		os.mkdir(options.output_foldername)




#my add code defolt output Json
	if options.full_json is None or options.full_json == 'True' or options.full_json == 'true':
		is_full_json = True

	if options.region_json is None or options.region_json == 'True' or options.region_json == 'true':
		is_region_json = True

	if options.output_foldername_full_json is None:
		options.output_foldername_full_json = options.output_foldername

	if not os.path.isdir(options.output_foldername_full_json):
		if is_full_json is True:
			os.mkdir(options.output_foldername_full_json)

	if options.output_foldername_region_json is None:
		options.output_foldername_region_json = options.output_foldername

	if not os.path.isdir(options.output_foldername_region_json):
		if is_region_json is True:
			os.mkdir(options.output_foldername_region_json)




	for video_path in os.listdir(options.input_foldername):
		is_image = False

		if not is_video_file(video_path):
			if video_path.endswith('.jpg') or video_path.endswith('.png'):
				is_image = True
			else:
				continue

		if is_image and is_video:
			print("Input imgaes are skipped when producing glitched videos")
			continue

		cap = None
		frame_width = 0
		frame_height = 0
		out = None

		if not is_image:
			cap = cv2.VideoCapture(os.path.join(options.input_foldername, video_path))
			frame_width = int(cap.get(3))
			frame_height = int(cap.get(4))

			if is_output_resized:
				frame_width = new_width
				frame_height = new_height

		save_normal_frames = False

		if is_video:
			if options.glitch_type is None:
				out = cv2.VideoWriter(os.path.join(options.output_foldername,str(count) + '_normal_video.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))
			else:
				out = cv2.VideoWriter(os.path.join(options.output_foldername,str(count) + '_' + str(options.glitch_type) +'_video.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))

		if options.save_normal_frames == 'True' or options.save_normal_frames == 'true' or is_video:
			save_normal_frames  = True

		if not is_video and save_normal_frames and not os.path.isdir(os.path.join(options.output_foldername, 'normal')):
			os.mkdir(os.path.join(options.output_foldername, 'normal'))

		if output_array and not os.path.isdir(os.path.join(options.output_foldername, 'np_array')):
			os.mkdir(os.path.join(options.output_foldername, 'np_array'))

		this_count = 0
		global prev_img
		while(True):
			ret  =  False
			original_img = None

			if is_image:
				ret = True
				original_img = cv2.imread(os.path.join(options.input_foldername, video_path))
				if original_img is None:
					print ("open error video_path ", video_path)
			else:
				ret, original_img = cap.read()
				if not ret:
					break

			img = np.copy(original_img)

			if is_margin_specified:
				img = original_img[x0:x1, y0:y1, :]

			if this_count % interval != 0:
				this_count += 1
				if save_normal_frames:
					output_name = str(count) + "_" + str(time.time()) + "_normal.png"
					if is_video:
						output_name = str(count)  + '_' + str(this_count)+ "_normal.png"
					output_filename = os.path.join(options.output_foldername, 'normal')
					output_filename = os.path.join(output_filename, output_name)
					# print(output_filename)
					new_img = img
					write_files(original_img, new_img, is_margin_specified, output_filename, out, is_video, False)

					if not is_video:
						count += 1
				continue

			if this_count == 0:
				prev_img = img

			if options.glitch_type is None:
				output_name = str(count) + "_" + str(time.time()) + "_normal.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				# X_orig_list.append()
				if not is_video:
					count += 1
############################################# NONE ########################################################
#имя входной картинки и её типа - video_path


			if options.glitch_type == 'screen_tearing':
				if is_image:
					print("Single input image is skipped when producing screen tearing glitches")
					break

				if this_count == 0:
					this_count += 1
					if save_normal_frames:
						output_name = str(count) + "_" + str(time.time()) + "_normal.png"
						if is_video:
							output_name = str(count)  + '_' + str(this_count)+ "_normal.png"
						output_filename = os.path.join(options.output_foldername, 'normal')
						output_filename = os.path.join(output_filename, output_name)
						new_img = img
						write_files(original_img, new_img, is_margin_specified, output_filename, out, is_video, False)
					continue

				height, width, channels = img.shape
				r = np.random.rand(1) * 0.8 + 0.1

				new_img = np.copy(img)
				if np.random.uniform() < 0.6:
					target_height = np.rint(r * height)
					target_height = target_height.astype(int)
					diff = (img[0:target_height[0], :, :] - prev_img[0:target_height[0], :, :]) / 255
					area = width * target_height[0]

					ssq = np.sum(diff**2) / area
					if ssq < 0.4:
						this_count += 1
						continue

					new_img[0:target_height[0], :, :] = prev_img[0:target_height[0], :, :]
				else:
					target_width = np.rint(r * width)
					target_width = target_width.astype(int)
					diff = (img[:, 0:target_width[0], :] - prev_img[:, 0:target_width[0], :]) / 255
					area = height * target_width[0]

					ssq = np.sum(diff**2) / area
					if ssq < 0.4:
						this_count += 1
						continue

					new_img[:, 0:target_width[0], :] = prev_img[:, 0:target_width[0], :]

				prev_img = img
				output_name = str(count) + "_" + str(time.time()) + "_screen_tearing.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, new_img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == "desktop_glitch_one":
				# print(img.shape)
				new_list = create_desktop_glitch_one(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_desktop_glitch_one"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == "desktop_glitch_two":
				new_list = create_desktop_glitch_two(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_desktop_glitch_two"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == "discoloration":
				# print(img.shape)
				new_list = create_discoloration(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_discoloration"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				# cv2.imwrite(output_filename, img)
				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == "random_patch":
				if not bool_flag:
					if is_bound_specified:
						new_list = addition_glitch.add_random_patches_mods(img, "1", arg1, arg2)
					else:
						new_list = addition_glitch.add_random_patches_mods(img, "1")
				else:
					if is_bound_specified:
						new_list = add_random_patches(img, "1", arg1, arg2)
					else:
						new_list = add_random_patches(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_random_patch"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'shape':
				if is_bound_specified:
					new_list = add_shapes(img, "1", arg1, arg2)
				else:
					new_list = add_shapes(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_shape"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'triangle':
				if is_bound_specified:
					new_list = add_triangles(img, "1", arg1, arg2)
				else:
					new_list = add_triangles(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_triangle"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1


			if options.glitch_type == 'shader':
				if is_bound_specified:
					new_list = add_shaders(img, "1", arg1, arg2)
				else:
					new_list = add_shaders(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_shader"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'dotted_line':
				if is_bound_specified:
					new_list = og.dotted_lines(img, "1", arg1, arg2)
				else:
					new_list = og.dotted_lines(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_dotted_line"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'radial_dotted_line':
				if is_bound_specified:
					new_list = og.dotted_lines_radial(img, "1", arg1, arg2)
				else:
					new_list = og.dotted_lines_radial(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_radial_dotted_line"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'parallel_line':
				if is_bound_specified:
					new_list = og.parallel_lines(img, "1", arg1, arg2)
				else:
					new_list = og.parallel_lines(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_parallel_line"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1


			if options.glitch_type == 'square_patch':
				if is_bound_specified:
					new_list = og.square_patches(img, "1", arg1, arg2)
				else:
					new_list = og.square_patches(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_square_patch"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1


			if options.glitch_type == 'texture_popin':
				new_list = blurring(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_texture_popin"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'random_triangulation':
				print("Random Triangulation is removed from the list of glitches")


			if options.glitch_type == 'regular_triangulation':
				new_list = triangulation(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_regular_triangulation"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'morse_code':
				new_list = add_vertical_pattern(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_morse_code"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'stuttering':
				new_list = produce_stuttering(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_stuttering"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'line_pixelation':
				new_list = line_pixelation(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_line_pixelation"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'white_square':
				if is_bound_specified:
					new_list = addition_glitch.white_square(img, "1", bool_flag, arg1, arg2)
				else:
					new_list = addition_glitch.white_square(img, "1", bool_flag)

				output_name = str(count) + "_" + str(time.time()) + "_white_square"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'black_tree':
				if is_bound_specified:
					new_list = addition_glitch.black_tree(img, "1", bool_flag, arg1, arg2)
				else:
					new_list = addition_glitch.black_tree(img, "1", bool_flag)

				output_name = str(count) + "_" + str(time.time()) + "_black_tree"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			this_count += 1

			if is_image:
				break

		if is_video:
			count += 1
		if cap is not None:
			cap.release()
		if out is not None:
			out.release()


	if output_array:
		output_folder = os.path.join(options.output_foldername, 'np_array')
		X_orig = np.asarray(X_orig_list)
		X_glitched = np.asarray(X_glitched_list)

		print("Numpy arrays are saved in " +  output_folder)
		# print("Number of dimensions of saved arrays are :" + str(X_orig.ndim) + ", and " + str(X_glitched.ndim))
		if X_orig.ndim ==  1 or X_glitched.ndim == 1:
			print("Either the input is empty, or the input images and/or videos frames are not of the same dimension. Consider resizing the outputs.")
		# print(X_glitched.ndim)

		np.save(os.path.join(output_folder, 'X_orig.npy'), X_orig)
		np.save(os.path.join(output_folder, 'X_glitched.npy'), X_glitched)





