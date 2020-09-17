import cv2
import numpy as np

from common_file import labelMe_class
from common_file.return_class import RetClass

def get_snowflakes_image(image):
    width = image.shape[1]
    height = image.shape[0]

    snowflakes_image = np.random.rand(int(height / 13), int(width / 13))
    snowflakes_image = np.concatenate((snowflakes_image, snowflakes_image), axis = 0)
    snowflakes_image = np.concatenate((snowflakes_image, snowflakes_image), axis = 1)
    snowflakes_image = np.concatenate((snowflakes_image, snowflakes_image), axis = 0)
    snowflakes_image = np.concatenate((snowflakes_image, snowflakes_image), axis = 1)

    snowflakes_image = cv2.resize(snowflakes_image, (width, height))
    snowflakes_image = np.uint8(np.stack([snowflakes_image, snowflakes_image, snowflakes_image], axis = 2) * 255)
    x_rectangle_u_l = np.random.randint(width * 0.1, width * 0.9)
    y_rectangle_u_l = np.random.randint(height * 0.1, height * 0.9)
    x_rectangle_r_d = x_rectangle_u_l + np.random.randint(width * 0.1, width * 0.30)
    y_rectangle_r_d = y_rectangle_u_l + np.random.randint(width * 0.1, width * 0.30)
    if x_rectangle_r_d > width: x_rectangle_r_d = width
    if y_rectangle_r_d > height: y_rectangle_r_d = height

    snowflakes_image[y_rectangle_u_l:y_rectangle_r_d, x_rectangle_u_l:x_rectangle_r_d] = image[y_rectangle_u_l:y_rectangle_r_d, x_rectangle_u_l:x_rectangle_r_d]

    black_zone = np.random.randint(0, 6)
    if black_zone == 1:
        diff = x_rectangle_r_d - x_rectangle_u_l
        x_black_r_d = x_rectangle_u_l + np.random.randint(0, diff)
        snowflakes_image[y_rectangle_u_l:y_rectangle_r_d, x_rectangle_u_l:x_black_r_d] = 0

    if black_zone == 2:
        diff = x_rectangle_r_d - x_rectangle_u_l
        x_black_u_l = x_rectangle_r_d - np.random.randint(0, diff)
        snowflakes_image[y_rectangle_u_l:y_rectangle_r_d, x_black_u_l:x_rectangle_r_d] = 0

    if black_zone == 3:
        diff = y_rectangle_r_d - y_rectangle_u_l
        y_black_r_d = y_rectangle_u_l + np.random.randint(0, diff)
        snowflakes_image[y_rectangle_u_l:y_black_r_d, x_rectangle_u_l:x_rectangle_r_d] = 0

    if black_zone == 4:
        diff = y_rectangle_r_d - y_rectangle_u_l
        y_black_u_l = y_rectangle_r_d - np.random.randint(0, diff)
        snowflakes_image[y_black_u_l:y_rectangle_r_d, x_rectangle_u_l:x_rectangle_r_d] = 0

    return snowflakes_image


def create_snowflakes(img, label):
	snowflakes_image = get_snowflakes_image(img)

	height, width, channel = snowflakes_image.shape

	p_shapes = labelMe_class.Shapes(label, [[0, 0], [0, height - 1],[width - 1, height - 1], [width - 1, 0]], None, "polygon", {})

	res = RetClass(snowflakes_image, [p_shapes])
	return res
