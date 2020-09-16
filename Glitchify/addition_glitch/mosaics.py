import cv2
import numpy.random as random
import numpy as np
import math

from common_file import labelMe_class
from common_file.return_class import RetClass

def to_int(value):
	return int(round(value))

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



def get_random_color(minimum , maximum, probability_of_emptiness_prosent):
	b,g,r = random.randint(minimum, maximum, size = 3)

	alpha = 0

	probability_of_emptiness = random.random()

	if probability_of_emptiness >= probability_of_emptiness_prosent:
		alpha = 1

	return (b,g,r, alpha)

def mosaics_texture(w,h,step_x, step_y, probability_of_emptiness_prosent):
	img = np.zeros((h, w, 4), np.uint8)

	for y in range(0, h+1, step_y):
		for x in range(0, w+1, step_x):
			color = get_random_color(0, 256, probability_of_emptiness_prosent)
			step_block_x = min(x+step_x, w) - x
			step_block_y = min(y+step_y, h) - y
			img[y:y+step_block_y, x:x+step_block_x, :] = color

	return img

def mosaics_texture_img(w,h, w_text, h_text, step_x, step_y, probability_of_emptiness_prosent):
	img_mosaic = np.zeros((h, w, 4), np.uint8)

	textur = mosaics_texture(w_text, h_text, step_x, step_y, probability_of_emptiness_prosent)

	for y in range(0, h+1, h_text):
		for x in range(0, w+1, w_text):
			step_block_img_x = min(x + w_text, w) - x
			step_block_img_y = min(y + h_text, h) - y
			img_mosaic[y:y+step_block_img_y, x:x+step_block_img_x, :] = textur[:step_block_img_y,:step_block_img_x, :]

	return img_mosaic


def contuer_filler (img, contour, texture):
	#min_x, min_y, max_x, max_y = cv2.boundingRect (contour)
	#h = max_y - min_y
	#w = max_x - min_x
	min_x, min_y, w, h = cv2.boundingRect (contour)
	max_x = min_x+w
	max_y = min_y+h

	texture = cv2.resize(texture,  (w,h))

	mask = np.zeros((img.shape[0],img.shape[1], 1), np.uint8)

	cv2.drawContours(mask, [contour], 0 , 255, -1)

	mask_contour = mask[min_y:max_y, min_x:max_x, 0]

	for y in range(h):
		for x in range(w):
			if mask_contour[y,x] == 255:
				if texture[y,x][3] == 0:
					img[min_y+y, min_x+x, :] = texture[y,x, 0:3]



	return img

def get_random_contoure(img):
	h,w,_ = img.shape

	x0 = random.randint(0, int(w / 6))
	y0 = random.randint(0, int(h / 6))
	x1 = random.randint(int(3 * w / 6), w)
	y1 = random.randint(int(3 * h / 6), h)

	size_x = x1-x0
	size_y = y1-y0

	x_top   = random.randint(x0, to_int(x1 - 0.5 * size_x))
	y_left  = random.randint(y0, to_int(y1 - 0.5 * size_y))
	x_down  = min(x1 + x0 - x_top  + random.randint(-to_int(size_x * 0.1)-1, to_int(size_x * 0.1)), x1)
	y_right = min(y1 + x0 - y_left + random.randint(-to_int(size_y * 0.1)-1, to_int(size_y * 0.1)), y1)

	pts = np.array(((x_top, y1), (x1, y_right), (x_down, y0), (x0,y_left)), dtype=int)

	return pts

def texture_generator(contour, fp):
	min_x, min_y, w, h = cv2.boundingRect(contour)
	mos_img = mosaics_texture_img(w, h, 25, 25, 5, 5, fp)
	# по идее бы сделать правильно преобразование (хотя бы поворот)


	return mos_img

def mosaics (img, label, fp, lo = 2, hi = 15):
	pic = np.copy(img)

	p_json_list = []

	pts = get_random_contoure(img)

	mos = texture_generator(pts, fp)

	pic = contuer_filler(pic, pts, mos)

	p_shapes = labelMe_class.Shapes(label, np_array_to_list_int_points(pts), None, "polygon", {})
	p_json_list.append(p_shapes)

	res = RetClass(pic, p_json_list)
	return res
