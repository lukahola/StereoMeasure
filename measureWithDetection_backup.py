__author__ = "Erhui Sun"
__credits__ = ["Erhui Sun", "Dragos Axinte", "Xin Dong"]
__version__ = "1.0.1"
__maintainer__ = "Erhui Sun"
__email__ = "erhui.sun1@nottingham.ac.uk"
__status__ = "Test"


import numpy as np
import glob
import cv2
from time import  ctime
from time import sleep
import os

import time



def loadCameraCalibration(filename):

	cameraParameter = np.loadtxt(filename)
	return cameraParameter


def pixel2Cam(p, K):

	x = np.float32((p[0]-K[0][2])/K[0][0])
	y = np.float32((p[1]-K[1][2])/K[1][1])  #

	return [x, y]


def triangulation(relativePose, mtxL, mtxR, match_kp_1, match_kp_2):
	P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
	P2 = relativePose

	pts_1 = []
	pts_2 = []

	[pts_1.append(pixel2Cam(match_kp_1[i], mtxL)) for i in np.arange(len(match_kp_1))]
	[pts_2.append(pixel2Cam(match_kp_2[i], mtxR)) for i in np.arange(len(match_kp_2))]

	pts_1 = np.array(pts_1).T
	pts_2 = np.array(pts_2).T

	pts_4d = cv2.triangulatePoints(P1, P2, pts_1, pts_2)
	pts_4d = np.transpose(pts_4d)
	r, c = pts_4d.shape
	pts_3d = []

	[pts_3d.append(pts_4d[i, 0:3] / pts_4d[i][3]) for i in np.arange(r)]

	b = pts_3d[0]
	d = pts_3d[1]

	# print('the coordination is :', pts_3d)

	dimension = np.sqrt(np.power(b[0] - d[0], 2) + np.power(b[1] - d[1], 2) + np.power(b[2] - d[2], 2))

	return pts_3d, dimension


def min_distance(point, kps):

	max_distance = 100000000000.0
	min_point = ()
	for kp in kps:
		pt = kp[0]
		distance = np.power(pt[0] - point[0], 2) + np.power(pt[1] - point[1], 2)
		if distance < max_distance:
			max_distance = distance
			min_point = kp

	return min_point


def mouse_choose(event, x, y, flags, params):

	global click_point1, click_point2, drawing, mouse_pos

	if event == cv2.EVENT_MOUSEMOVE:
		mouse_pos = (x, y)

	elif event == cv2.EVENT_LBUTTONDOWN:
		if drawing is False:
			click_point1 = (x, y)
			drawing = True
		else:
			drawing = False
			click_point2 = (x, y)


def mouse_choose2(event, x, y, flags, params):
	global click_point1, click_point2, click_point3, click_point4, drawing, drawing2, mouse_pos

	if event == cv2.EVENT_MOUSEMOVE:
		mouse_pos = (x, y)
	elif event == cv2.EVENT_LBUTTONDOWN:
		if drawing is False and drawing2 is False:
			click_point1 = (x, y)
			drawing = True
		elif drawing is True and drawing2 is False:
			drawing2 = True
			click_point2 = (x, y)

		elif drawing is True and drawing2 is True:
			click_point3 = (x, y)
			drawing = False

		elif drawing is False and drawing2 is True:
			click_point4 = (x, y)
			drawing2 = False


def inv_transform(se3):

	r = se3[0:3, 0:3]
	r = r.T

	t = se3[0:3, 3].reshape(3, 1)

	t_new = -np.dot(r, t)

	inv_se3 = np.concatenate((r, t_new), axis=1)

	return inv_se3


def px2cam(px):

	fx = mtxL[0, 0]
	fy = mtxL[1, 1]
	cx = mtxL[0, 2]
	cy = mtxL[1, 2]

	return np.array([(px[0]-cx)/fx, (px[1]-cy)/fy, 1])


def cam2px(p_cam):

	fx = mtxR[0, 0]
	fy = mtxR[1, 1]
	cx = mtxR[0, 2]
	cy = mtxR[1, 2]

	return np.array([p_cam[0]*fx/p_cam[2]+cx, p_cam[1]*fy/p_cam[2]+cy])


def inside(pt):

	flag = (boarder <= pt[0] <= width - boarder) and (boarder <= pt[1] <= height - boarder)

	return flag


def get_bilinear_interpolated_value(img, pt):

	x = int(pt[0])
	y = int(pt[1])
	xx = pt[0] - x
	yy = pt[1] - y

	result = ((1-xx)*(1-yy)*img[y, x] + xx*(1-yy)*img[y, x+1] + (1-xx)*yy*img[y+1, x] + xx*yy*img[y+1, x+1])/255.0

	return result


def normalized_cross_Correlation(ref, curr, pt_ref, pt_curr):

	mean_ref = 0
	mean_curr = 0

	values_ref = []
	values_curr = []

	for x in np.arange(-ncc_window_size, ncc_window_size+1):
		for y in np.arange(-ncc_window_size, ncc_window_size+1):
			value_ref = ref[int(y+pt_ref[1]), int(x+pt_ref[0])]/255.0

			mean_ref += value_ref

			value_curr = get_bilinear_interpolated_value(curr, pt_curr+[x, y])
			mean_curr += value_curr

			values_ref.append(value_ref)
			values_curr.append(value_curr)

	mean_ref /= ncc_area
	mean_curr /= ncc_area

	numerator = 0
	denominator1 = 0
	denominator2 = 0

	for i in np.arange(len(values_ref)):
		numerator += (values_ref[i]-mean_ref) * (values_curr[i]-mean_curr)
		denominator1 += np.square(values_ref[i]-mean_ref)
		denominator2 += np.square(values_curr[i] - mean_curr)

	result = numerator / np.sqrt(denominator1*denominator2+1e-10)

	return result


def epi_polar_search(ref, curr, tcr, pt_ref, depth_mu, depth_cov):

	f_ref = px2cam(pt_ref)

	f_ref = f_ref / np.linalg.norm(f_ref)
	px_max_curr = ()
	px_min_curr = ()
	d_min = 30
	inside_min = False
	while not inside_min:

		P_min_ref = np.hstack((f_ref * d_min, 1))

		px_min_curr = cam2px(tcr.dot(P_min_ref))

		inside_min = inside(px_min_curr)

		d_min += 20
	d_max = 5000
	inside_max = False
	while not inside_max:

		P_max_ref = np.hstack((f_ref * d_max, 1))
		px_max_curr = cam2px(tcr.dot(P_max_ref))
		inside_max = inside(px_max_curr)

		d_max -= 100

	epi_polar_line = px_max_curr - px_min_curr
	epi_polar_line_norm = np.linalg.norm(epi_polar_line)

	epi_polar_direction = epi_polar_line / epi_polar_line_norm

	best_ncc = -1.0

	best_px_curr = np.zeros((2, 1))

	for l in np.arange(0, epi_polar_line_norm, 0.707/4):
		px_curr = px_min_curr + l * epi_polar_direction

		if not inside(px_curr):
			continue

		ncc = normalized_cross_Correlation(ref, curr, pt_ref, px_curr)

		if ncc > best_ncc:
			best_ncc = ncc
			best_px_curr = px_curr

	px_curr = best_px_curr

	return px_curr, px_max_curr, px_min_curr


def clipped(frame, mouse_pos):

	roi = frame[mouse_pos[1] - 10:mouse_pos[1] + 10, mouse_pos[0] - 10:mouse_pos[0] + 10]
	blank_frame = cv2.resize(roi, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
	rows, cols, _ = blank_frame.shape
	rows = int(rows / 2)
	cols = int(cols / 2)

	cv2.line(blank_frame, (rows-20, cols), (rows+20, cols), (0, 0, 255), 2)
	cv2.line(blank_frame, (rows, cols-20), (rows, cols+20), (0, 0, 255), 2)

	return blank_frame


def measure():
	print(ctime())
	# **************load camera parameters****************

	pose_tcr = loadCameraCalibration(trialname+'camMatrix/relativePose.txt')

	# ******************************

	global click_point1, click_point2, drawing, click_point3, click_point4, drawing2
	global mouse_pos
	global pt_curr_1, pt_curr_2
	cv2.namedWindow('frame')
	cv2.namedWindow('floating window', cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('frame', 100, 250)
	cv2.moveWindow('floating window', 100, 250)
	cv2.setMouseCallback('frame', mouse_choose2)

	blank_frame = np.zeros((100, 100, 3), np.uint8)
	# ******************************
	filenames = sorted(glob.glob('measure/*.jpg'))

	img = cv2.imread(filenames[0])
	img1 = img[:, 0:640]
	img2 = img[:, 640:1280]

	# ***********************************
	undist1 = cv2.undistort(img1, mtxL, distL)
	undist2 = cv2.undistort(img2, mtxR, distR)

	img1 = cv2.cvtColor(undist1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(undist2, cv2.COLOR_BGR2GRAY)
	img_undist = np.hstack((undist1, undist2))

	flag = True

	while True:
		frame = img_undist.copy()
		if frame is not None:
			key = cv2.waitKey(int(1000 / 60)) & 0xFF

			y_lim, x_lim, _ = frame.shape

			if key == ord('f'):
				flag = False
				click_point1 = ()
				click_point2 = ()
				pt_curr_1 = np.array([[]])
				pt_curr_2 = np.array([[]])
				drawing = False
				click_point3 = ()
				click_point4 = ()
				drawing2 = False

			if key == ord('t'):
				flag = True
				click_point1 = ()
				click_point2 = ()
				pt_curr_1 = np.array([[]])
				pt_curr_2 = np.array([[]])
				drawing = False
				click_point3 = ()
				click_point4 = ()
				drawing2 = False

			if flag:
				
				if click_point1:

					cv2.circle(frame, click_point1, 3, (0, 0, 255), -1)
					(x, y) = click_point1
					if not pt_curr_1.any():
						pt_curr_1, px_min_curr, px_max_curr = epi_polar_search(
							img1, img2, pose_tcr, [x, y], 500, 500)
						
					if pt_curr_1.any():
						cv2.circle(frame, (int(pt_curr_1[0])+640, int(pt_curr_1[1])), 3, (0, 255, 0), -1)

				if click_point1 and click_point2:

					cv2.circle(frame, click_point2, 3, (0, 0, 255), -1)
					cv2.line(frame, click_point1, click_point2, (0, 0, 255), 2)
					(x, y) = click_point2
					if not pt_curr_2.any():
						pt_curr_2, _, _ = epi_polar_search(
							img1, img2, pose_tcr, [x, y], 500, 500)

					if pt_curr_2.any():
						cv2.circle(frame, (int(pt_curr_2[0])+640, int(pt_curr_2[1])), 3, (0, 255, 0), -1)

					min_point1 = [click_point1, click_point2]
					min_point2 = [(pt_curr_1[0], pt_curr_1[1]), (pt_curr_2[0], pt_curr_2[1])]
					pts_3d, dimension = triangulation(pose_tcr, mtxL, mtxR, min_point1, min_point2)

					dimension_show = round(dimension, 2)

					circle_center = click_point1
					if click_point1[0] < click_point2[0]:
						circle_center = click_point2

					cv2.putText(frame, str(dimension_show), (int(circle_center[0] + 10), int(circle_center[1] + 10)),
								cv2.FONT_HERSHEY_COMPLEX, .8, (0, 0, 255))
					cv2.putText(frame, 'dimension: ' + str(dimension_show), (25, 25),
								cv2.FONT_HERSHEY_COMPLEX, .8, (0, 0, 255))

			if not flag:
				if click_point1:
					cv2.circle(frame, click_point1, 3, (0, 0, 255), -1)

				if click_point1 and click_point2:
					cv2.line(frame, click_point1, click_point2, (0, 0, 255), 2)

				if click_point3:
					cv2.circle(frame, click_point3, 3, (0, 255, 0), -1)

				if click_point3 and click_point4:
					cv2.line(frame, click_point3, click_point4, (0, 255, 0), 2)


				if click_point1 and click_point2 and click_point3 and click_point4:
					min_point1 = [click_point1, click_point2]
					min_point2 = [(click_point3[0]-640, click_point3[1]),
								  (click_point4[0]-640, click_point4[1])]
					pts_3d, dimension = triangulation(pose_tcr, mtxL, mtxR, min_point1, min_point2)

					dimension_show = round(dimension, 2)

					circle_center = click_point1
					if click_point1[0] < click_point2[0]:
						circle_center = click_point2

					cv2.putText(frame, str(dimension_show), (int(circle_center[0] + 10), int(circle_center[1] + 10)),
								cv2.FONT_HERSHEY_COMPLEX, .6, (0, 0, 255))

					cv2.putText(frame, 'dimension: ' + str(dimension_show), (25, 25),
								cv2.FONT_HERSHEY_COMPLEX, .6, (0, 0, 255))

			if mouse_pos:
				if 10 <= mouse_pos[0] <= x_lim - 10 and 10 <= mouse_pos[1] <= y_lim - 10:
					blank_frame = clipped(frame, mouse_pos)

			cv2.moveWindow('frame', 0, 0)
			cv2.moveWindow('floating window', 0, y_lim+0)
			cv2.imshow('frame', frame)
			cv2.imshow('floating window', blank_frame)

			if key == ord('q'):
				click_point1 = ()
				click_point2 = ()
				pt_curr_1 = np.array([[]])
				pt_curr_2 = np.array([[]])
				drawing = False
				click_point3 = ()
				click_point4 = ()
				drawing2 = False
				break

			elif key == ord('d'):
				click_point1 = ()
				click_point2 = ()
				pt_curr_1 = np.array([[]])
				pt_curr_2 = np.array([[]])
				drawing = False
				click_point3 = ()
				click_point4 = ()
				drawing2 = False

	cv2.destroyWindow('frame')
	cv2.destroyWindow('floating window')


def main():


	cv2.namedWindow('detection')

	cv2.moveWindow('detection', 0, 0)

	cap1 = cv2.VideoCapture(0)
	cap2 = cv2.VideoCapture(2)

	try:
		while True:
			
			ret1, imgL_Origin = cap1.read()
			ret2, imgR_Origin = cap2.read()

			if ret1 and ret2:

				img = np.hstack((imgL_Origin, imgR_Origin))
				cv2.moveWindow('detection', 0, 0)
				cv2.imshow('detection', img)

				key = cv2.waitKey(1)

				if key == ord('s'):
					filename = 'measure/img.jpg'
					cv2.imwrite(filename, img)
					sleep(2)
					measure()

				elif key == ord('m'):
					curr = img.copy()
					cv2.imshow('curr', curr)
					cv2.waitKey(0)
					cv2.destroyWindow('curr')

	finally:
		cv2.destroyAllWindows()
 

if __name__ == "__main__":
	boarder = 10
	width = 640
	height = 480

	ncc_window_size = 2  # NCC half width
	ncc_area = (2*ncc_window_size+1)*(2*ncc_window_size+1)  # NCC area
	min_cov = 0.1  
	max_cov = 10    
	trialname = './'

	try:
		os.mkdir('./measure')
	except:
		print('file is existing!')

	#############################################
	mtxL = loadCameraCalibration(trialname+'camMatrix/cameraMatrixL.txt') # intrinsic
	distL = loadCameraCalibration(trialname+'camMatrix/distortionL.txt')
	mtxR = loadCameraCalibration(trialname+'camMatrix/cameraMatrixR.txt')
	distR = loadCameraCalibration(trialname+'camMatrix/distortionR.txt')

	mouse_pos = ()

	click_point1 = ()
	click_point2 = ()
	click_point3 = ()
	click_point4 = ()

	pt_curr_1 = np.array([[]])
	pt_curr_2 = np.array([[]])

	drawing = False
	drawing2 = False

	main()