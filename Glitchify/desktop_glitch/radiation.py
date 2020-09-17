import cv2
import numpy as np

from common_file import labelMe_class
from common_file.return_class import RetClass

def contur_to_list_int_points(contour):
	list_points = []
	for point in contour:
		(x,y) = point[0]
		list_points.append([int(x),int(y)])
	return list_points

def np_array_to_list_int_points(arr):
	list_points = []
	for (x,y) in arr:
		list_points.append([int(x),int(y)])
	return list_points

def cart2pol(x, y):
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(R, phi)

def get_away_points(width, height):
    if np.random.randint(0, 2) == 1:
        if np.random.randint(0, 2) == 1:
            x_v = np.random.randint(0, width * 0.2)
            x_t = np.random.randint(width * 0.6, width * 0.8)
        else:
            x_v = np.random.randint(width * 0.8, width)
            x_t = np.random.randint(width * 0.2, width * 0.4)
        y_v = np.random.randint(0, height)
        y_t = np.random.randint(height * 0.2, height * 0.8)
    else:
        if np.random.randint(0, 2) == 1:
            y_v = np.random.randint(0, height * 0.2)
            y_t = np.random.randint(height * 0.6, height * 0.8)
        else:
            y_v = np.random.randint(height * 0.8, height)
            y_t = np.random.randint(height * 0.2, height * 0.4)
        x_v = np.random.randint(0, width)
        x_t = np.random.randint(width * 0.2, width * 0.8)
    return np.array((x_v, y_v)), np.array((x_t, y_t))

def get_triangles(width, height):
    #width = image.shape[1]
    #height = image.shape[0]
    size_triangles = 20
    H_size_triangles = 20
    
    mask_triangles = np.zeros(shape = (height, width)).astype(np.uint8)
    num_triangle = np.random.randint(1, 5)
    vertex_triangles = np.empty(shape = (0, 3, 2)).astype(np.int32)
    
    common_vertex, common_triangles = get_away_points(width, height)
    R, phi = cart2pol(common_triangles[0] - common_vertex[0], common_triangles[1] - common_vertex[1])
    
    for i in range(num_triangle):
        #vertex_triangle_c = np.array((np.random.randint(width * 0.2, width * 0.8), np.random.randint(height * 0.2, height * 0.8)))
        deviation = np.random.rand() - 0.5
        vertex_triangle_c = common_vertex + np.array((R * np.cos(phi + deviation), R * np.sin(phi + deviation)))
        
        vector = np.float32(vertex_triangle_c - common_vertex)
        vector = vector / np.max(np.abs(vector))
        
        H = vertex_triangle_c + np.int32(vector * H_size_triangles)
        
        rand_size = np.random.random()
        vertex_triangle_a = H + np.int32(vector[::-1] * (size_triangles * rand_size, -size_triangles * rand_size))
        vertex_triangle_b = H - np.int32(vector[::-1] * (size_triangles * rand_size, -size_triangles * rand_size))
        
        p = np.array((vertex_triangle_a, vertex_triangle_b, vertex_triangle_c)).astype(np.int32)
        vertex_triangles = np.append(vertex_triangles, [p], axis = 0)
        
        mask_triangles = cv2.fillConvexPoly(mask_triangles, points = p, color = 255)
        mask_triangles[common_vertex[1], common_vertex[0]] = 255
        #mask_triangles[common_triangles[1], common_triangles[0]] = 255
        
    return mask_triangles.astype(np.uint8), vertex_triangles, common_vertex


def radiation(image, vertex_triangles, common_vertex):
	width = image.shape[1]
	height = image.shape[0]

	warp_image = image.copy()

	for i in range(vertex_triangles.shape[0]):
		#(widht, height)

		mask = np.full((image.shape[0], image.shape[1]), 0, dtype = np.uint8)
		mask = cv2.fillConvexPoly(mask, points = vertex_triangles[i], color = 255)
		triangle = cv2.bitwise_or(image, image, mask = mask)

		srcTri = vertex_triangles[i].astype(np.float32)
		dstTri = np.concatenate((vertex_triangles[i][:2], [common_vertex])).astype(np.float32)

		warp_mat = cv2.getAffineTransform(srcTri, dstTri)
		warp_triangle = cv2.warpAffine(triangle, warp_mat, (triangle.shape[1], triangle.shape[0]),
									   flags = cv2.INTER_NEAREST)#,
									   #borderMode = cv2.BORDER_REFLECT)
										#INTER_NEAREST    INTER_LINEAR    INTER_CUBIC    WARP_FILL_OUTLIERS
										#BORDER_CONSTANT    BORDER_REPLICATE

		warp_image[warp_triangle != 0] = 0
		warp_image = cv2.bitwise_or(warp_image, warp_triangle)

	return warp_image

def find_contours(image):
	contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return contours

def common_vertex_triangle(vertex_triangles, common_vertex):
	vertex_triangles_with_common = []

	for triangle in vertex_triangles:
		vertex_triangles_with_common.append([triangle[0], triangle[1], common_vertex])

	return vertex_triangles_with_common

def create_radiation(img, label):
	height, width, channel = img.shape

	mask_triangles, vertex_triangles, common_vertex = get_triangles(width, height)
	warp_img = radiation(img, vertex_triangles, common_vertex)
	warp_mask = radiation(mask_triangles, vertex_triangles, common_vertex)
	contours = find_contours(warp_mask)

	p_json_list = []

	#vertex_triangles_with_common = common_vertex_triangle(vertex_triangles, common_vertex)

	for triangle in contours:
		p_shapes = labelMe_class.Shapes(label, contur_to_list_int_points(triangle), None, "polygon", {})
		p_json_list.append(p_shapes)

	res = RetClass(warp_img, p_json_list)
	return res
