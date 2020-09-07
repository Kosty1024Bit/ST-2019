import cv2
import numpy as np
import random

from common_file.return_class import TreeRet
from common_file import labelMe_class

# in_name='bd-original-14.jpg'
# out_name='stuttering_test.png'

def divisors(n):
	divisors=[]
	for i in list(range(1,int(n/2)+1)):
		if (n%i==0):
			divisors.append(i)
	divisors.append(n)
	return divisors

def swap(image,pix1,pix2):
	dummyr=image[pix1[0],pix1[1],0]
	dummyg=image[pix1[0],pix1[1],1]
	dummyb=image[pix1[0],pix1[1],2]
	image[pix1[0],pix1[1],0]=image[pix2[0],pix2[1],0]
	image[pix1[0],pix1[1],1]=image[pix2[0],pix2[1],1]
	image[pix1[0],pix1[1],2]=image[pix2[0],pix2[1],2]
	image[pix2[0],pix2[1],0]=dummyr
	image[pix2[0],pix2[1],1]=dummyg
	image[pix2[0],pix2[1],2]=dummyb

def stutter(res, label, sizex,sizey):

	vstripes=int(res.img.shape[1]/sizex)
	hstripes=int(res.img.shape[0]/sizey)

	for k in list(range(0, res.img.shape[1], 2 * sizex)):
		for i in list(range(sizex)):
			for j in list(range(res.img.shape[0])):
				if(i + k + sizex < res.img.shape[1]):
					swap(res.img, [j, i + k],[j, sizex + i + k])

		first_y = 0
		first_x_1 = k
		first_x_2 = sizex + k

		is_stop_1_2 = False
		if(first_x_1 + sizex < res.img.shape[1]):
			last_x_1 = first_x_1 + sizex
		else:
			last_x_1 = res.img.shape[1] - 1
			is_stop_2 = True


		if(first_x_2 + sizex < res.img.shape[1]):
			last_x_2 = first_x_2 + sizex
		else:
			last_x_2 = res.img.shape[1] - 1

		last_y = res.img.shape[0] - 1

		f_shapes_1 = labelMe_class.Shapes(label, [[first_x_1, first_y],[last_x_1, last_y]], None, "rectangle", {})
		res.f_json.append(f_shapes_1.to_string_form())

		if not is_stop_1_2:
			f_shapes_2 = labelMe_class.Shapes(label, [[first_x_2, first_y],[last_x_2, last_y]], None, "rectangle", {})
			res.f_json.append(f_shapes_2.to_string_form())

	for k in list(range(0, res.img.shape[0], 2 * sizey)):
		for i in list(range(sizey)):
			for j in list(range(res.img.shape[1])):
				if(i + k + sizey < res.img.shape[0]):
					swap(res.img, [i + k, j],[i + k + sizey, j])

		first_y_1 = k
		first_y_2 = k + sizey
		first_x = 0

		is_stop_2_2 = False

		if(first_y_1 + sizey < res.img.shape[0]):
			last_y_1 = first_y_1 + sizey
		else:
			last_y_1 = res.img.shape[0] - 1
			is_stop_2 = True

		if(first_y_2 + sizey < res.img.shape[0]):
			last_y_2 = first_y_2 + sizey
		else:
			last_y_2 = res.img.shape[0] - 1

		last_x = res.img.shape[1] - 1

		f_shapes_1 = labelMe_class.Shapes(label, [[first_x, first_y_1],[last_x, last_y_1]], None, "rectangle", {})
		res.f_json.append(f_shapes_1.to_string_form())

		if not is_stop_2_2:
			f_shapes_2 = labelMe_class.Shapes(label, [[first_x, first_y_2],[last_x, last_y_2]], None, "rectangle", {})
			res.f_json.append(f_shapes_2.to_string_form())

	return res



def produce_stuttering(image, label):
	vdivisors=divisors(image.shape[0])
	hdivisors=divisors(image.shape[1])
	iterations=np.random.choice(np.arange(1,5),p=[0.65,0.2,0.1,0.05])

	res = TreeRet(image.copy(), [], [])

	for i in list(range(iterations)):
		sizex=random.choice(hdivisors)
		sizey=random.choice(vdivisors)
		res = stutter(res, label, sizex,sizey)

	r_shapes = labelMe_class.Shapes(label, [[0, 0],[image.shape[1] - 1, image.shape[0] - 1]], None, "rectangle", {})
	res.r_json.append(r_shapes.to_string_form())

	return res


# img=np.array(cv2.imread(in_name,1))
# vdivisors=divisors(img.shape[0])
# hdivisors=divisors(img.shape[1])
# iterations=np.random.choice(np.arange(1,5),p=[0.65,0.2,0.1,0.05])
# for i in list(range(iterations)):
# 	sizex=random.choice(hdivisors)
# 	sizey=random.choice(vdivisors)
# 	img=stutter(img,sizex,sizey)
# cv2.imwrite(out_name,img)