import cv2
import os
import numpy as np
import argparse
from desktop_glitch.desktop_glitch_one import *
from desktop_glitch.desktop_glitch_two import create_desktop_glitch_two
from desktop_glitch.radiation import create_radiation
import ou_glitch.ou_glitch as og
from stuttering.stuttering import produce_stuttering
from line_pixelation.line_pixelation import line_pixelation
from addition_glitch import addition_glitch
from glitchify_modules import glitchify_modules
from addition_glitch import overlap
from addition_glitch import mosaics

import json
from common_file import labelMe_class

from common_file.return_class import RetClass

from common_file.return_class import TreeRet

import time



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



def new_write_poligon_json_files(p_json, original_name_file, filename, img_shape):
	string_p_json = []
	for el_p_json in p_json:
		string_p_json.append(el_p_json.to_string_form())

	#write json with adres filename
	write_p_json = labelMe_class.Json("0.0.0 version", {}, string_p_json, original_name_file, None, img_shape[0], img_shape[1])
	with open(filename + "_poligon.json", "w") as write_file:
		json.dump(write_p_json.to_string_form(), write_file, indent=4)


def poligon_to_full_json(p_json, img_shape):
	string_f_json = []
	for el_p_json in p_json:
		min_x = img_shape[1] - 1
		min_y = img_shape[0] - 1
		max_x = 0
		max_y = 0

		for (x,y) in el_p_json.points:
			min_x = min(min_x, x)
			min_y = min(min_y, y)
			max_x = max(max_x, x)
			max_y = max(max_y, y)

		f_shapes = labelMe_class.Shapes(el_p_json.label, [[min_x, min_y], [max_x, max_y]], None, "rectangle", {})
		string_f_json.append(f_shapes.to_string_form())

	return string_f_json

def new_write_full_json_files(p_json, original_name_file, filename, img_shape):
	#write json with adres filename
	string_f_json = poligon_to_full_json(p_json, img_shape)
	write_f_json = labelMe_class.Json("0.0.0 version", {}, string_f_json, original_name_file, None, img_shape[0], img_shape[1])
	with open(filename + "_full.json", "w") as write_file:
		json.dump(write_f_json.to_string_form(), write_file, indent=4)

def poligon_to_region_json(p_json, img_shape):
	min_x = img_shape[1] - 1
	min_y = img_shape[0] - 1
	max_x = 0
	max_y = 0

	for el_p_json in p_json:
		for (x,y) in el_p_json.points:
			min_x = min(min_x, x)
			min_y = min(min_y, y)
			max_x = max(max_x, x)
			max_y = max(max_y, y)

	r_shapes = labelMe_class.Shapes(el_p_json.label, [[min_x, min_y], [max_x, max_y]], None, "rectangle", {})

	return [r_shapes.to_string_form()]


def new_write_region_json_files(p_json, original_name_file, filename, img_shape):
	#write json with adres filename
	string_r_json = poligon_to_region_json(p_json, img_shape)
	write_r_json = labelMe_class.Json("0.0.0 version", {}, string_r_json, original_name_file, None, img_shape[0], img_shape[1])
	with open(filename + "_region.json", "w") as write_file:
		json.dump(write_r_json.to_string_form(), write_file, indent=4)

def new_all_writer(ret_class, outname, is_poligon_json, poligon_outname, is_full_json, full_outname, is_region_json, region_outname):

	output_filename = os.path.join(options.output_foldername, outname + ".png")
	cv2.imwrite(output_filename, ret_class.img)

	if is_poligon_json:
		new_write_poligon_json_files(ret_class.p_json, output_filename, poligon_outname, ret_class.img.shape)

	if is_full_json:
		new_write_full_json_files(ret_class.p_json, output_filename, full_outname, ret_class.img.shape)

	if is_region_json:
		new_write_region_json_files(ret_class.p_json, output_filename, region_outname, ret_class.img.shape)



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

	parser.add_argument('-fp', '--fill_percentage', dest='arg3', default = 0)

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
	parser.add_argument('-pj', '--poligon_json', dest = 'poligon_json')

	parser.add_argument('-ofj', '--output_full_json', dest = 'output_foldername_full_json')
	parser.add_argument('-orj', '--output_region_json', dest = 'output_foldername_region_json')
	parser.add_argument('-opj', '--output_poligon_json', dest = 'output_foldername_poligon_json')

	parser.add_argument('-bf', '--boolean_flag', dest = 'boolean_flag')
	parser.add_argument('-pwf', '--present_write_format', dest = 'present_write_format')


	options = parser.parse_args()
	global arg1, arg2, arg3, x0, y0, x1, y1, is_output_resized, new_height, new_width

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
	is_poligon_json = False

	if options.resize_output == 'True' or options.resize_output == 'true':
		is_output_resized = True
		new_height = int(options.new_height)
		new_width = int(options.new_width)

	if options.boolean_flag == 'True' or options.boolean_flag == 'true':
		bool_flag = True
	else:
		bool_flag = False

	if options.present_write_format == 'True' or options.present_write_format == 'true':
		is_present_write_format = True
	else:
		is_present_write_format = False


	# if options.output_type == 'video' or options.output_type == 'Video':
	# 	is_video = True
	is_video = False

	if options.arg1 is not None and options.arg2 is not None:
		is_bound_specified = True
		arg1 = int(options.arg1)
		arg2 = int(options.arg2)

	arg3 = float(options.arg3)


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

	if options.poligon_json is None or options.poligon_json == 'True' or options.poligon_json_json == 'true':
		is_poligon_json = True

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

	if options.output_foldername_poligon_json is None:
		options.output_foldername_poligon_json = options.output_foldername

	if not os.path.isdir(options.output_foldername_poligon_json):
		if is_poligon_json is True:
			os.mkdir(options.output_foldername_poligon_json)




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


			#для универсального типа вывода
			new_list = None
			output_name = None

############################################################################################

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

				new_list = None
				output_name = None

############################################################################################

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

				new_list = None
				output_name = None

############################################################################################

			if options.glitch_type == "radiation":
				new_list = create_radiation(img, "8")

				output_name = str(count) + "_" + str(time.time()) + "_radiation"

############################################################################################

			if options.glitch_type == "discoloration":
				# print(img.shape)
				if not bool_flag:
					new_list = addition_glitch.create_discoloration_new(img, "1")
				else:
					new_list = glitchify_modules.create_discoloration(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_discoloration"

############################################################################################

			if options.glitch_type == "random_patch":
				if not bool_flag:
					if is_bound_specified:
						new_list = addition_glitch.add_random_patches_mods(img, "1", arg1, arg2)
					else:
						new_list = addition_glitch.add_random_patches_mods(img, "1")
				else:
					if is_bound_specified:
						new_list = glitchify_modules.add_random_patches(img, "1", arg1, arg2)
					else:
						new_list = glitchify_modules.add_random_patches(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_random_patch"

############################################################################################

			if options.glitch_type == 'shape':
				if is_bound_specified:
					new_list = glitchify_modules.add_shapes(img, "1", arg1, arg2)
				else:
					new_list = glitchify_modules.add_shapes(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_shape"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

				new_list = None
				output_name = None

############################################################################################

			if options.glitch_type == 'triangle':
				if is_bound_specified:
					new_list = glitchify_modules.add_triangles(img, "1", arg1, arg2)
				else:
					new_list = glitchify_modules.add_triangles(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_triangle"

############################################################################################

			if options.glitch_type == 'shader':
				if is_bound_specified:
					new_list = glitchify_modules.add_shaders(img, "1", arg1, arg2)
				else:
					new_list = glitchify_modules.add_shaders(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_shader"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

				output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
				output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)

				write_full_json_files(new_list.f_json, is_full_json, output_filename, output_filename_f_json, new_list.img)
				write_region_json_files(new_list.r_json, is_region_json, output_filename, output_filename_r_json, new_list.img)

				write_files(original_img, new_list.img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

				new_list = None
				output_name = None

############################################################################################

			if options.glitch_type == 'dotted_line':
				if is_bound_specified:
					new_list = og.dotted_lines(img, "1", arg1, arg2)
				else:
					new_list = og.dotted_lines(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_dotted_line"

############################################################################################

			if options.glitch_type == 'radial_dotted_line':
				if is_bound_specified:
					new_list = og.dotted_lines_radial(img, "8", arg1, arg2)
				else:
					new_list = og.dotted_lines_radial(img, "8")

				output_name = str(count) + "_" + str(time.time()) + "_radial_dotted_line"

############################################################################################

			if options.glitch_type == 'parallel_line':
				if is_bound_specified:
					new_list = og.parallel_lines(img, "8", arg1, arg2)
				else:
					new_list = og.parallel_lines(img, "8")

				output_name = str(count) + "_" + str(time.time()) + "_parallel_line"

############################################################################################

			if options.glitch_type == 'square_patch':
				if is_bound_specified:
					new_list = og.square_patches(img, "1", arg1, arg2)
				else:
					new_list = og.square_patches(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_square_patch"

############################################################################################

			if options.glitch_type == 'texture_popin':
				new_list = glitchify_modules.blurring(img, "9")

				output_name = str(count) + "_" + str(time.time()) + "_texture_popin"

############################################################################################

			if options.glitch_type == 'random_triangulation':
				print("Random Triangulation is removed from the list of glitches")

############################################################################################

			if options.glitch_type == 'regular_triangulation':
				new_list = glitchify_modules.triangulation(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_regular_triangulation"
				output_filename = os.path.join(options.output_foldername, output_name + ".png")

############################################################################################

			if options.glitch_type == 'morse_code':
				new_list = glitchify_modules.add_vertical_pattern(img, "1")

				output_name = str(count) + "_" + str(time.time()) + "_morse_code"

############################################################################################

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

				new_list = None
				output_name = None


############################################################################################

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

				new_list = None
				output_name = None


############################################################################################

			if options.glitch_type == 'white_square':
				if is_bound_specified:
					new_list = addition_glitch.white_square(img, "1", bool_flag, arg3, arg1, arg2)
				else:
					new_list = addition_glitch.white_square(img, "1", bool_flag, arg3)

				output_name = str(count) + "_" + str(time.time()) + "_white_square"

############################################################################################

			if options.glitch_type == 'black_tree':
				if is_bound_specified:
					new_list = addition_glitch.black_tree(img, "1", bool_flag, arg1, arg2)
				else:
					new_list = addition_glitch.black_tree(img, "1", bool_flag)

				output_name = str(count) + "_" + str(time.time()) + "_black_tree"

############################################################################################

			if options.glitch_type == 'color_cast':
				if is_bound_specified:
					new_list = addition_glitch.color_cast(img, "2", bool_flag, arg1, arg2)
				else:
					new_list = addition_glitch.color_cast(img, "2", bool_flag)

				output_name = str(count) + "_" + str(time.time()) + "_color_cast"

############################################################################################

			if options.glitch_type == 'overlap':
				if is_bound_specified:
					new_list = overlap.overlap(img, "4", arg1, arg2)
				else:
					new_list = overlap.overlap(img, "4")

				output_name = str(count) + "_" + str(time.time()) + "_overlap"

############################################################################################

			if options.glitch_type == 'mosaics':
				if is_bound_specified:
					new_list = mosaics.mosaics(img, "6", arg3, arg1, arg2)
				else:
					new_list = mosaics.mosaics(img, "6", arg3)

				output_name = str(count) + "_" + str(time.time()) + "_mosaics"
############################################################################################
############################################################################################
############################################################################################

			if new_list is not None and output_name is not None:
				if not is_present_write_format:
					output_filename_f_json = os.path.join(options.output_foldername_full_json, output_name)
					output_filename_r_json = os.path.join(options.output_foldername_region_json, output_name)
					output_filename_p_json = os.path.join(options.output_foldername_poligon_json, output_name)

					new_all_writer(new_list, output_name, is_poligon_json, 	output_filename_p_json,\
													      is_full_json,     output_filename_f_json,\
														  is_region_json,   output_filename_r_json)
				else:
					output_filename = os.path.join(options.output_foldername, output_name + ".png")
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





