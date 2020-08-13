import os
import random
import cv2
import numpy as np
import math
from PIL import Image

from common_file import labelMe_class
from common_file.tree_return_class import TreeRet

def vrglow(image,pixel,color,radius):
	transparent=np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
	for i in list(range(-radius,radius+1)):
		for j in list(range(1,radius+1)):
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
	foreground=Image.fromarray(transparent,mode="RGBA")
	background=Image.fromarray(image,mode="RGBA")
	background.paste(foreground,(0,0),foreground)
	image=np.array(background)
	image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
	return image

def vlglow(image,pixel,color,radius):
	transparent=np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
	for i in list(range(-radius,radius+1)):
		for j in list(range(-radius,0)):
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
	foreground=Image.fromarray(transparent,mode="RGBA")
	background=Image.fromarray(image,mode="RGBA")
	background.paste(foreground,(0,0),foreground)
	image=np.array(background)
	image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
	return image

def hbglow(image,pixel,color,radius):
	transparent=np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
	for j in list(range(-radius,radius+1)):
		for i in list(range(1,radius+1)):
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
	foreground=Image.fromarray(transparent,mode="RGBA")
	background=Image.fromarray(image,mode="RGBA")
	background.paste(foreground,(0,0),foreground)
	image=np.array(background)
	image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
	return image

def htglow(image,pixel,color,radius):
	transparent=np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
	for j in list(range(-radius,radius+1)):
		for i in list(range(-radius,0)):
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
	foreground=Image.fromarray(transparent,mode="RGBA")
	background=Image.fromarray(image,mode="RGBA")
	background.paste(foreground,(0,0),foreground)
	image=np.array(background)
	image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
	return image

def line_pixelation(img, label, label2 = "2"):

	f_json_list = []
	max_x = 0
	max_y = 0
	min_x = img.shape[1]
	min_y =	img.shape[0]

	max_x_p = 0
	max_y_p = 0
	min_x_p = img.shape[1]
	min_y_p = img.shape[0]

	while(True):
		vertical=0
		horizontal=1

		# PARAMETERS
		skipstripe=random.randrange(0,2)
		orientation=random.randrange(0,2)
		brightness=random.randrange(0,2)
		monobias=random.randrange(0,3)
		biasval=random.randrange(0,11)
		glow=random.randrange(0,8)

		image=np.copy(img)
		height=max(abs(int(np.random.normal(int(image.shape[0]/200),2))),1)
		width=max(abs(int(np.random.normal(height,2))),1)
		if(orientation==vertical):
			if(width<image.shape[1]):
				indent=random.randrange(0,image.shape[1]-width)
			else:
				print("Error: 'width' is too large for input dimensions. ")
				continue
		if(orientation==horizontal):
			if(height<image.shape[0]):
				indent=random.randrange(0,image.shape[0]-height)
			else:
				print("Error: 'height' is too large for input dimensions.")
				continue
		stripes=random.randrange(1,max(1+abs(int(np.random.normal(20,20))), 2))
		ss=np.ones(stripes)
		if(skipstripe==1):
			ss=[1]
			for i in list(range(stripes-2)):
				ss.append(random.randrange(0,2))
			ss.append(1)
		if(monobias==1):
			monocolor=[0,0,0]
		if(monobias==2):
			monocolor=[255,255,255]

		if(orientation==vertical):
			for n in list(range(stripes)):
				if (ss[n]==1):

					temp_min_x = img.shape[1]-1
					temp_min_y = img.shape[0]-1
					temp_max_x = 0
					temp_max_y = 0

					temp_min_x_p = img.shape[1]-1
					temp_min_y_p = img.shape[0]-1
					temp_max_x_p = 0
					temp_max_y_p = 0

					for i in list(range(0,image.shape[0],height)):

						color=np.array([random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)])
						mono=0
						if(monobias>0):
							mono=random.randrange(1,11)
						if(glow==6 and n==0 and random.randrange(1,10)<random.randrange(1,3)):
							radius = random.randrange(5,4+4*height)
							y = i*height+int(height/2)
							if (y - radius < image.shape[0]- 10):
								image=vlglow(image,[y ,indent],color, radius)

								temp_min_x_p = min(max(indent - radius,0), img.shape[1]-1)
								temp_min_y_p = min(max(y - radius, 0), img.shape[0]-1)

								temp_max_x_p = min(indent, img.shape[1]-1)
								temp_max_y_p = min(y + radius, img.shape[0]-1)

								min_x_p = min(min_x_p, temp_min_x_p)
								max_x_p = max(max_x_p, temp_max_x_p)

								min_y_p = min(min_y_p, temp_min_y_p)
								max_y_p = max(max_y_p, temp_max_y_p)

								f_shapes = labelMe_class.Shapes(label2, [[temp_min_x_p ,temp_min_y_p ], [temp_max_x_p, temp_max_y_p]], None, "rectangle", {})
								f_json_list.append(f_shapes.to_string_form())


						if(glow==7 and n==(len(ss)-1) and random.randrange(1,10)<random.randrange(1,4)):
							radius = random.randrange(5,70)
							y = i*height+int(height/2)
							if (y - radius < image.shape[0]- 10):
								image=vrglow(image,[y, indent+n*width],color, radius)

								temp_min_x_p = min(max(indent+n*width, 0), img.shape[1]-1)
								temp_min_y_p = min(max(y - radius, 0), img.shape[0]-1)

								temp_max_x_p = min(indent+n*width + 1 + radius, img.shape[1]-1)
								temp_max_y_p = min(y + radius, img.shape[0]-1)

								min_x_p = min(min_x_p, temp_min_x_p)
								max_x_p = max(max_x_p, temp_max_x_p)

								min_y_p = min(min_y_p, temp_min_y_p)
								max_y_p = max(max_y_p, temp_max_y_p)

								f_shapes = labelMe_class.Shapes(label2, [[temp_min_x_p ,temp_min_y_p ], [temp_max_x_p, temp_max_y_p]], None, "rectangle", {})
								f_json_list.append(f_shapes.to_string_form())


						for j in list(range(height)):
							for k in list(range(width)):
								localcolor=np.array(color)
								if(((i+j)<image.shape[0]) and (indent+k+n*width<image.shape[1])):
									if(brightness==1 and mono<=biasval):
										seed=int(np.random.normal(0,10))
										localcolor[0]=max(min(color[0]+seed,255),0)
										localcolor[1]=max(min(color[1]+seed,255),0)
										localcolor[2]=max(min(color[2]+seed,255),0)
									elif(mono>biasval):
										localcolor=monocolor
									image[i+j,indent+(k+n*width)]=localcolor

									temp_min_x = min(indent+(k+n*width), temp_min_x)
									temp_min_y = min(i+j, temp_min_y)

									temp_max_x = min(max(indent+(k+n*width) + 1, temp_max_x), img.shape[1]-1)
									temp_max_y = min(max(i+j, temp_max_y) + 1,	img.shape[0]-1)



					if (temp_min_x != 0 or temp_min_y != 0 or temp_max_x != img.shape[1]-1 or temp_max_y != img.shape[0]-1):

						f_shapes = labelMe_class.Shapes(label, [[temp_min_x, temp_min_y], [temp_max_x, temp_max_y]], None, "rectangle", {})
						f_json_list.append(f_shapes.to_string_form())

						min_x = min(min_x, temp_min_x)
						max_x = max(max_x, temp_max_x)

						min_y = min(min_y, temp_min_y)
						max_y = max(max_y, temp_max_y)

		if(orientation==horizontal):
			for n in list(range(stripes)):
				if (ss[n]==1):
					temp_min_x = img.shape[1]-1
					temp_min_y = img.shape[0]-1
					temp_max_x = 0
					temp_max_y = 0

					temp_min_x_p = img.shape[1]-1
					temp_min_y_p = img.shape[0]-1
					temp_max_x_p = 0
					temp_max_y_p = 0

					for i in list(range(0,image.shape[1],width)):
						color=np.array([random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)])
						mono=0
						if(monobias>0):
							mono=random.randrange(1,11)
						if(glow==6 and n==0 and random.randrange(1,10)<random.randrange(1,3)):
							x = i*width+int(width/2)
							radius = random.randrange(5,4+4*width)
							if (x - radius < img.shape[1]- 10):
								image=htglow(image,[indent, x],color, radius)

								temp_min_y_p = min(max(indent - radius,0), img.shape[0]-1)
								temp_min_x_p = min(max(x - radius, 0), img.shape[1]-1)

								temp_max_y_p = min(indent,img.shape[0]-1)
								temp_max_x_p = min(x + radius,img.shape[1]-1)

								min_x_p = min(min_x_p, temp_min_x_p)
								max_x_p = max(max_x_p, temp_max_x_p)

								min_y_p = min(min_y_p, temp_min_y_p)
								max_y_p = max(max_y_p, temp_max_y_p)

								f_shapes = labelMe_class.Shapes(label2, [[temp_min_x_p ,temp_min_y_p ], [temp_max_x_p, temp_max_y_p]], None, "rectangle", {})
								f_json_list.append(f_shapes.to_string_form())

						if(glow==7 and n==(len(ss)-1) and random.randrange(1,10)<random.randrange(1,4)):
							radius = random.randrange(5,70)
							x = i*width+int(width/2)
							if (x - radius < img.shape[1] - 10):
								image=hbglow(image,[indent+height*n, x],color,radius)

								temp_min_y_p = min(max(indent+height*n ,0), img.shape[0]-1)
								temp_min_x_p = min(max(x - radius, 0), img.shape[1]-1)

								temp_max_y_p = min(indent+height*n + radius,img.shape[0]-1)
								temp_max_x_p = min(x + radius,img.shape[1]-1)

								min_x_p = min(min_x_p, temp_min_x_p)
								max_x_p = max(max_x_p, temp_max_x_p)

								min_y_p = min(min_y_p, temp_min_y_p)
								max_y_p = max(max_y_p, temp_max_y_p)

								f_shapes = labelMe_class.Shapes(label2, [[temp_min_x_p ,temp_min_y_p ], [temp_max_x_p, temp_max_y_p]], None, "rectangle", {})
								f_json_list.append(f_shapes.to_string_form())

						for j in list(range(width)):
							for k in list(range(height)):
								localcolor=np.array(color)
								if(((k+n*height+indent)<image.shape[0]) and (i+j<image.shape[1])):
									if(brightness==1 and mono<=biasval):
										seed=int(np.random.normal(0,10))
										localcolor[0]=max(min(color[0]+seed,255),0)
										localcolor[1]=max(min(color[1]+seed,255),0)
										localcolor[2]=max(min(color[2]+seed,255),0)
									elif(mono>biasval):
										localcolor=monocolor
									image[indent+k+(n*height),i+j]=localcolor

									temp_min_y = min(indent+k+(n*height), temp_min_y)
									temp_min_x = min(i+j, temp_min_x)

									temp_max_y = min(max(indent+k+(n*height) + 1, temp_max_y),	img.shape[0]-1)
									temp_max_x = min(max(i+j, temp_max_x),	img.shape[1]-1)


					if (temp_min_x != 0 or temp_min_y != 0 or temp_max_x != img.shape[1]-1 or temp_max_y != img.shape[0]-1):

						f_shapes = labelMe_class.Shapes(label, [[temp_min_x, temp_min_y], [temp_max_x, temp_max_y]], None, "rectangle", {})
						f_json_list.append(f_shapes.to_string_form())

						min_x = min(min_x, temp_min_x)
						max_x = max(max_x, temp_max_x)

						min_y = min(min_y, temp_min_y)
						max_y = max(max_y, temp_max_y)

		if(not np.array_equal(img,image)):
			r_shapes = labelMe_class.Shapes(label, [[min_x, min_y], [max_x, max_y]], None, "rectangle", {})
			r_json_list = [r_shapes.to_string_form()]
			if (max_x_p != 0 or max_y_p != 0 or min_x_p != img.shape[1]-1 or min_y_p != img.shape[0]-1):
				r_shapes = labelMe_class.Shapes(label2, [[min_x_p, min_y_p], [max_x_p, max_y_p]], None, "rectangle", {})
				r_json_list.append(r_shapes.to_string_form())
			res = TreeRet(image, f_json_list, r_json_list)
			return res



# img = cv2.imread("square_patch.jpg")
# img = noisy_line(img)
# cv2.imwrite("result.png", img)



