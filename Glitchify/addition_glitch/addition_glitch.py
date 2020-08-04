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

def check_val(value):
	if value > 255:
		value = 255
	if value < 0:
		value = 0
	return value

def white_square(picture, lo = 2, hi = 15):
	pic = picture.copy()
	height = pic.shape[0]
	width = pic.shape[1]
	number_of_patches = random.randint(lo,hi+1)

	first_y = -1
	first_x = -1

	#r = int(random.uniform(0, 1)*255)
	#g = int(random.uniform(0, 1)*255)
	#b = int(random.uniform(0, 1)*255)

	for i in range(number_of_patches):
		size = random.randint(20,50)
		#red = check_val(r + random.randint(-30,30))
		#green = check_val(g + random.randint(-30,30))
		#blue = check_val(b + random.randint(-30,30))
		color = [250,250,250]#[blue, green, red]
		if first_y < 0:
			first_y = random.randint(int(height*0.2), int(height*0.8))
			first_x = random.randint(int(width*0.2), int(width*0.8))
			pic[first_y:(first_y+size), first_x:(first_x+size)] = color
		else:
			y = first_y +  random.randint(-int(height*0.1), int(height*0.1))
			x = first_x +  random.randint(-int(width*0.1), int(width*0.1))
			pic[y:(y+size), x:(x+size)] = color


	res_list = namedtuple('res_list','img f_json r_json')
	res = res_list(img = pic, f_json = None, r_json = None)
	return res

