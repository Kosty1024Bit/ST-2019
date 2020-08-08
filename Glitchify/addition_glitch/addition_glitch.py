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


def white_square(picture, label, lo = 2, hi = 15):
	pic = picture.copy()
	height = pic.shape[0]
	width = pic.shape[1]
	number_of_patches = random.randint(lo,hi+1)

	f_json_list = []
	max_x = 0
	max_y = 0
	min_x = width
	min_y = height

	for i in range(number_of_patches):
		size = random.randint(20,50)
		#red = check_val(r + random.randint(-30,30))
		#green = check_val(g + random.randint(-30,30))
		#blue = check_val(b + random.randint(-30,30))
		color = [250,250,250]#[blue, green, red]

		first_y = random.randint(int(height*0.2), int(height*0.8))
		first_x = random.randint(int(width*0.2), int(width*0.8))

		last_y = first_y + size
		last_x = first_x + size

		min_x = min(min_x, first_x, last_x)
		max_x = max(max_x, first_x, last_x)

		min_y = min(min_y, first_y, last_y)
		max_y = max(max_y, first_y, last_y)

		pic[first_y:(last_y), first_x:(last_x)] = color

		f_shapes = labelMe_class.Shapes(label, [[first_x, first_y], [last_x, last_y]], None, "rectangle", {})

		f_json_list.append(f_shapes.to_string_form())

	r_shapes = labelMe_class.Shapes(label, [[min_x, min_y],[max_x, max_y]], None, "rectangle", {})

	res = TreeRet(pic, f_json_list, [r_shapes.to_string_form()])
	return res

