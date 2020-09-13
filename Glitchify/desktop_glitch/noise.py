import numpy as np

from common_file import labelMe_class
from common_file.return_class import TreeRet


def form_combined_pattern(img):
	h, w, channel = img.shape
    
	sub_w = int(w/8)
	if sub_w <= 0:
		return img
    
	cstep = 5

# vertical lines
	for i in range(0, sub_w*8, sub_w):
		c1 = np.random.randint(0,255)
		c2 = c1 + h*cstep
		if c1 > 127:
			c2 = c1 - h*cstep
		a = np.arange(c1,c2,cstep)
		if c1 > 127:
			a = np.arange(c2,c1,cstep)
		b = np.swapaxes(np.array(sub_w*[np.array([a,]*3).transpose()]), 1, 0)
		img[0:h, i: i+sub_w:] = b

# pixelizate 
	a = np.swapaxes(np.swapaxes(np.array([np.random.randint(-8, 8, size=(h, w))]*3), 1, 0), 2,1)
	img = img + a
   
# more pixelizate bottom half      
	h_2 = h//2
	if np.random.uniform() < 0.4: 
		a = np.swapaxes(np.swapaxes(np.array([np.random.randint(-30, 30, size=(h_2, w))]*3), 1, 0), 2,1)
		img[-h_2-1:-1, 0:w] = img[-h_2-1:-1, 0:w] + a
        
# more pixelizate right half
	w_2 = w//2
	if np.random.uniform() < 0.4: 
		a = np.swapaxes(np.swapaxes(np.array([np.random.randint(-30, 30, size=(h, w_2))]*3), 1, 0), 2,1)
		img[0:h, -w_2-1:-1] = img[0:h, -w_2-1:-1] + a
    
# darker
	img[1:h-1, 1:w-1] = img[1:h-1, 1:w-1] - np.random.randint(0,50)
	img[2:h-2, 2:w-2] = img[2:h-2, 2:w-2] - np.random.randint(0,100)
    
	img[img < 0] = 0
	img[img > 255] = 255  

	return img



def create_noise(img, label):
	height, width, channel = img.shape
	sub_h = 8
	sub_w = 8

	f_json_list = []

	layers = 5
	p = np.random.uniform(0.2, 1, size=layers)
	for i in range(0, height - sub_h,  sub_h//2):
		for j in range(0, width - sub_w, sub_w):
			if np.random.uniform() * p[int(i/height * layers)] < 0.02:
				img[i:i+sub_h, j:j+sub_w, :] = form_combined_pattern(img[i:i+sub_h, j:j+sub_w, :])
				f_shapes = labelMe_class.Shapes(label, [[j, i], [j + sub_w, i + sub_h]], None, "rectangle", {})
				f_json_list.append(f_shapes.to_string_form())

	res = TreeRet(img, f_json_list, None)
	return res




