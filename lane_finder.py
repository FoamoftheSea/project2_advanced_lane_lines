import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from moviepy.editor import VideoFileClip

class Lane_Finder:
	"""
	This class can find the lanes on the road, as well as their curvature, and locate the
	position of the car relative to the lane center. The process_image() method will take in
	an image (from the same camera used for calibration) and return an annotated image.
	
	NOTE: You may pass mtx, dist, src, and dst into the initialization explicitly or not.
	If mtx and dst are not passed, images in the calib_dir will automatically be used to
	calibrate the camera upon instantiation.
	If src/dst not provided, default parameters for perspective transformation will be used.
	"""
	def __init__(self,
				 mtx=None, 
				 dist=None, 
				 calib_dir='camera_cal/',
				 nx=9,
				 ny=6,
				 objp=None,
				 src=None,
				 dst=None,
				 h_thresh=None,
				 l_thresh=(50, 255),
				 s_thresh=(120, 255), 
				 sx_thresh=(20, 100),
				 frame_buffer=5,
				 margin=100,
				 padding=200,
				 lane_length=49.0,
				 lane_width=3.7,
				 nwindows=9,
				 minpix=50,
				 output_dir='output_images/',
				 always_windows=False
				):
		"""
		Params:
		mtx - (array) intrinsic matrix for camera. Automatically generated if None
		dist - (array) distortion coefficients for camera. Auto generated if None
		calib_dir - (str) directory to camera calibration images. Only used if above are None
					NOTE: only calibration images should be stored in this folder!
		nx - (int) number of chessboard corners in calib images, x direction
		ny - (int) number of chessboard corners in calib images, y direction
		objp - (int) 3D coordinates of chessboard corners. Leave as None to use arbitrary scale
		src - (array) source points for perspective transform
		dst - (array) destination points for perspective transform
		h_thresh - (tuple) lower and upper threshold for hue
		l_thresh - (tuple) lower and upper threshold for lightness
		s_thresh - (tuple) lower and upper threshold for saturation
		sx_thresh - (tuple) lower and upper threshold for x gradient
		frame_buffer - (int) number of frames to average polynomial curve fits for videos
		padding - (int) number of pixels to add on sides of warped image to catch lane curvature
		lane_length - (float) number of meters (est) of lane in front of car being considered
		lane_width - (float) width of lane in meters
		nwindows - (int) number of sliding window positions to locate lanes
		minpix - (int) minimum number of pixels to recenter sliding window over
		output_dir - (str) path to folder to store output videos
		always_windows - (bool) whether to use sliding window search in every frame
		"""
		self.nx = nx
		self.ny = ny
		self.h_thresh = h_thresh
		self.l_thresh = l_thresh
		self.s_thresh = s_thresh
		self.sx_thresh = sx_thresh
		self.frame_buffer = frame_buffer
		self.margin = margin
		self.padding = padding
		self.lane_length = lane_length
		self.lane_width = lane_width
		self.nwindows = nwindows
		self.minpix = minpix
		self.recent_fit = False
		self.output_dir = output_dir
		self.processing_video = False
		self.first = True
		self.always_windows = always_windows

		# Empty lists for polynomial fits in buffer
		self.left_fits = []
		self.right_fits = []
		self.left_fits_metric = []
		self.right_fits_metric = []

		# Generate 3D object points
		if objp is None:
			objp = np.zeros((ny*nx, 3), np.float32)
			objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
			self.objp = objp
		# Try to calibrate camera if calibration information not provided
		if mtx is None:
			print('Calibrating camera with images in {}'.format(calib_dir))
			#try:
			self.calibrate(calib_dir)
			self.calibrated = True
			#except:
			#    print('Failed to calibrate camera, using no image correction.')
			#    print('Consider retrying calibration with appropriate calib_dir')
			#    self.calibrated = False
		
		# Load default src and dst points if none provided:
		if src == None:
			src = np.array([[200, 720], [595, 450],
							[686, 450], [1100, 720]],
						   np.float32)
		if dst == None:
			dst = np.array([[200 + padding, 720], [200 + padding, 0],
							[1110 + padding, 0], [1110 + padding, 720]],
						   np.float32)
	
		# Set up transformation matrix and inverse transform matrix
		self.M = cv2.getPerspectiveTransform(src, dst)
		self.Minv = cv2.getPerspectiveTransform(dst, src)
				
	def calibrate(self, calib_dir):
		"""
		Calibrates camera based on chessboard images in provided directory.
		"""
		cal_images = os.listdir(calib_dir)
		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.

		# termination criteria for corner refinement
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		for cal_image in cal_images:
			# Note that we're using plt.imread() so we use RGB2GRAY color conversion
			img = plt.imread(calib_dir + cal_image)
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			# Find the chess board corners
			ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
			# If found, add object points, image points (after refining them)
			if ret == True:
				# Stores the list we created earlier for every successful return
				objpoints.append(self.objp)
				corners = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
				imgpoints.append(corners)
				# Draw and display the corners
				cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
				cv2.imshow('img', img)
				cv2.waitKey(200)
			# Report when corners are not found
			else:
				print('Corners not found for ', cal_image)
		cv2.destroyAllWindows()
		
		# Now to extract the camera matrix from our corner detections
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
														   imgpoints, 
														   gray.shape[::-1], None, None)
		# Apply the calibration
		self.mtx = mtx
		self.dist = dist
		print('Calibration successful!')
		
		pass
	
	
	def binary_lane_img(self, 
						image, 
						h_thresh=None,
						l_thresh=None,
						s_thresh=None, 
						sx_thresh=None, 
						diagnostic=False):
		'''
		This function takes in an image, then returns a binary image based on the color space
		and x derivative thresholds passed. Diagnostic mode returns a colored image so the user
		can see the effects of their threshold choices.

		Params:
		image - (array) image to be processed
		s_thresh - (tuple) saturation thresholds to apply to HLS color space (lower, upper)
		sx_thresh - (tuple) x derivative thresholds to apply to sobel gradient (lower, upper)
		diagnostic - (bool) whether to return diagnostic image for threshold tuning
		'''
		if h_thresh == None:
			h_thresh = self.h_thresh
		if l_thresh == None:
			l_thresh = self.l_thresh
		if s_thresh == None:
			s_thresh = self.s_thresh
		if sx_thresh == None:
			sx_thresh = self.sx_thresh

		# Copy image to preserve input
		img = np.copy(image)
		# Convert to HLS color space and separate the V channel
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		h_channel = hls[:,:,0]
		l_channel = hls[:,:,1]
		s_channel = hls[:,:,2]
		# Sobel x
		sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
		abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
		scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

		# Threshold x gradient
		sxbinary = np.zeros_like(scaled_sobel)
		sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

		# Threshold color channels
		# Hue
		if h_thresh is not None:
			h_binary = np.zeros_like(h_channel)
			h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
		# Saturation
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
		# Lightness
		l_binary = np.zeros_like(l_channel)
		if l_thresh is not None:
			l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

		if diagnostic:
			# Stack each channel (does not include hue)
			color_binary = np.dstack((l_binary, sxbinary, s_binary)) * 255
			return color_binary
		else:
			binary = np.zeros_like(img[:, :, 0])
			if l_thresh is None:
				if h_thresh is None:
					binary[(sxbinary == 1) | (s_binary == 1)] = 1
				else:
					binary[((h_binary == 1) & (s_binary == 1)) | (sxbinary == 1)] = 1
			else:
				if h_thresh is None:
					binary[((l_binary == 1) & (s_binary == 1)) | (sxbinary == 1)] = 1
				else:
					binary[((h_binary == 1) & (l_binary == 1) & (s_binary == 1)) & (sxbinary == 1)] = 1
			return binary
		
		
	# On a first iteration, we need to use a sliding window search
	def find_lane_pixels(self, binary_warped):
		# Take a histogram of the bottom half of the image
		histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
		# Create an output image to draw on and visualize the result
		out_img = np.zeros((*binary_warped.shape, 3), dtype=np.uint8)
		#out_img = np.dstack((binary_warped, binary_warped, binary_warped))
		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]//2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Set height of windows - based on nwindows above and image shape
		window_height = np.int(binary_warped.shape[0]//self.nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated later for each window in nwindows
		leftx_current = leftx_base
		rightx_current = rightx_base

		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(self.nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
			### Find the four below boundaries of the window ###
			win_xleft_low = leftx_current - self.margin
			win_xleft_high = leftx_current + self.margin
			win_xright_low = rightx_current - self.margin
			win_xright_high = rightx_current + self.margin

			## Draw the windows on the visualization image
			#cv2.rectangle(out_img,(win_xleft_low,win_y_low),
			#(win_xleft_high,win_y_high),(0,255,0), 2) 
			#cv2.rectangle(out_img,(win_xright_low,win_y_low),
			#(win_xright_high,win_y_high),(0,255,0), 2) 

			### Identify the nonzero pixels in x and y within the window ###
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
			(nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)

			### If you found > minpix pixels, recenter next window ###
			if len(good_left_inds) > self.minpix:
				leftx_current = np.mean(nonzerox[good_left_inds]).astype('int')
			if len(good_right_inds) > self.minpix:
				rightx_current = np.mean(nonzerox[good_right_inds]).astype('int')
			### (`right` or `leftx_current`) on their mean position ###
			#pass # Remove this when you add your function

		# Concatenate the arrays of indices (previously was a list of lists of pixels)
		try:
			left_lane_inds = np.concatenate(left_lane_inds)
			right_lane_inds = np.concatenate(right_lane_inds)
		except ValueError:
			# Avoids an error if the above is not implemented fully
			pass

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		return leftx, lefty, rightx, righty, out_img
	
	
	def search_around_poly(self, binary_warped, visualize=False):

		# Grab activated pixels
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		# Set the area of search based on activated x-values
		# within the +/- margin of our polynomial function
		prev_line_left = self.left_fit[0]*nonzeroy**2 + \
						 self.left_fit[1]*nonzeroy + self.left_fit[2]
		prev_line_right = self.right_fit[0]*nonzeroy**2 + \
						  self.right_fit[1]*nonzeroy + self.right_fit[2]
		left_lane_inds = ((nonzerox >= prev_line_left - self.margin) &
						  (nonzerox < prev_line_left + self.margin)).nonzero()[0]
		right_lane_inds = ((nonzerox >= prev_line_right - self.margin) &
						   (nonzerox < prev_line_right + self.margin)).nonzero()[0]

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		
		# Generate an output image
		out_img = np.zeros((*binary_warped.shape, 3), np.uint8)
		#out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		
		if visualize:
			## Visualization ##
			# Create an image to draw on and an image to show the selection window
			window_img = np.zeros_like(out_img)
			# Color in left and right line pixels
			out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
			out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

			# Generate a polygon to illustrate the search window area
			# And recast the x and y points into usable format for cv2.fillPoly()
			left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
			left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, 
									  ploty])))])
			left_line_pts = np.hstack((left_line_window1, left_line_window2))
			right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
			right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, 
									  ploty])))])
			right_line_pts = np.hstack((right_line_window1, right_line_window2))

			# Draw the lane onto the warped blank image
			cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
			cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
			result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

			# Plot the polynomial lines onto the image
			plt.plot(left_fitx, ploty, color='yellow')
			plt.plot(right_fitx, ploty, color='yellow')
			plt.imshow(result)
			plt.show()
			## End visualization steps ##

		return leftx, lefty, rightx, righty, out_img


	def fit_polynomial(self, leftx, lefty, rightx, righty, metric=False):
		# Fit a second order polynomial to each using np.polyfit()
		if not metric:

			try:
				self.left_fit = np.polyfit(lefty, leftx, 2)
				self.right_fit = np.polyfit(righty, rightx, 2)
				self.recent_fit = True
			except:
				self.left_fit = self.left_fit
				self.right_fit = self.right_fit
				self.recent_fit = False

		else:

			try:
				self.left_fit_metric = np.polyfit(lefty * self.ym_per_pixel,
												  leftx * self.xm_per_pixel,
												  2
												 )
				self.right_fit_metric = np.polyfit(righty * self.ym_per_pixel,
												   rightx * self.xm_per_pixel,
												   2
												  )
			except:
				self.left_fit_metric = self.left_fit_metric
				self.right_fit_metric = self.right_fit_metric

		# If processing video, add current fit to list, and average curves
		if self.processing_video:

			if not metric:
				# Do left polynomial curve
				self.left_fits.append(self.left_fit)
				if len(self.left_fits) >= self.frame_buffer:
					self.left_fits = self.left_fits[-self.frame_buffer:]
				self.left_fit = np.mean(self.left_fits, axis=0)

				# Do right polynomial curve
				self.right_fits.append(self.right_fit)
				if len(self.right_fits) >= self.frame_buffer:
					self.right_fits = self.right_fits[-self.frame_buffer:]
				self.right_fit = np.mean(self.right_fits, axis=0)

			else:
				# Do left metric polynomial curve
				self.left_fits_metric.append(self.left_fit_metric)
				if len(self.left_fits_metric) >= self.frame_buffer:
					self.left_fits_metric = self.left_fits_metric[-self.frame_buffer:]
				self.left_fit_metric = np.mean(self.left_fits_metric, axis=0)

				# Do right metric polynomial curve
				self.right_fits_metric.append(self.right_fit_metric)
				if len(self.right_fits_metric) >= self.frame_buffer:
					self.right_fits_metric = self.right_fits_metric[-self.frame_buffer:]
				self.right_fit_metric = np.mean(self.right_fits_metric, axis=0)

		
		pass
	
	def calc_curvature(self, y_eval):
		left_curverad = (1 + (2*self.left_fit_metric[0]*y_eval + \
							  self.left_fit_metric[1])**2)**(3/2) / \
							np.abs(2*self.left_fit_metric[0])
		right_curverad = (1 + (2*self.right_fit_metric[0]*y_eval + \
							   self.right_fit_metric[1])**2)**(3/2) / \
							np.abs(2*self.right_fit_metric[0])
		# Return curve radii
		return left_curverad, right_curverad
	
	
	def process_image(self, image, first=True):
		"""
		Takes in an image and returns an annotated copy with the lanes marked, and with the
		lane curvature radius and vehicle distance from lane center written on.
		
		Params:
		image - (array) image to be processed
		first - (bool) whether this is a single image or the first image of a sequence
		"""
		if not self.processing_video:
			self.first = first
		
		self.imheight = image.shape[0]
		self.imwidth = image.shape[1]
		
		# If calibration info is available, undistort image
		if self.calibrated:
			image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
		
		# Get binary image
		binary = self.binary_lane_img(image)
		# Warp binary image
		binary_warped = cv2.warpPerspective(binary.copy(), 
											self.M, 
											(self.imwidth + self.padding * 2, self.imheight), 
											flags=cv2.INTER_LINEAR)
		# Establish meters per pixel in y direction
		self.ym_per_pixel = self.lane_length / binary_warped.shape[0]
		# Find our lane pixels. Use sliding window if on first/single frame
		if self.first or self.always_windows:
			leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)
		else:
			leftx, lefty, rightx, righty, out_img = self.search_around_poly(binary_warped)
		# Fit polynomials
		self.fit_polynomial(leftx, lefty, rightx, righty)
		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		
		try:
			left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
			right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
		except TypeError:
			# Avoids an error if `left` and `right_fit` are still none or incorrect
			print('The function failed to fit a line!')
			left_fitx = 1*ploty**2 + 1*ploty
			right_fitx = 1*ploty**2 + 1*ploty
			
		# If on first/single frame, use calculated x values at bottom of image to calculate
		# the xm_per_pixel
		if self.first:
			self.xm_per_pixel = self.lane_width / (right_fitx[-1] - left_fitx[-1])
		# Calculate curvature
		self.fit_polynomial(leftx, lefty, rightx, righty, metric=True)
		y_eval = np.max(ploty) * self.ym_per_pixel
		left_curverad, right_curverad = self.calc_curvature(y_eval)
		# Get average curve radius
		curverad = (left_curverad + right_curverad) / 2
		# Find center of lane in pixels, remember to remove padding
		lane_center_px = ((left_fitx[-1] + right_fitx[-1]) / 2) - self.padding
		img_center = self.imwidth / 2
		offset_px = img_center - lane_center_px
		offset_meters = self.xm_per_pixel * offset_px

		## Visualization ##    
		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))
		# Colors in the left and right lane regions
		out_img[lefty, leftx] = [255, 0, 0]
		out_img[righty, rightx] = [0, 0, 255]
		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(out_img, self.Minv, (image.shape[1], image.shape[0])) 
		# Combine the result with the original image
		out_img = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
		# Write lane curvature on image
		font_face = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 1
		curve_string = 'Lane Curvature Radius: {}m'.format(round(curverad, 2))
		offset_string = 'Vehicle offset from lane center: {}m'.format(round(offset_meters, 2))
		out_img = cv2.putText(out_img, 
							  curve_string, 
							  (25,50),
							  font_face,
							  font_scale,
							  (255, 255, 255),
							  thickness=2
							 )
		out_img = cv2.putText(out_img, 
							  offset_string, 
							  (25,100),
							  font_face,
							  font_scale,
							  (255, 255, 255),
							  thickness=2
							 )
		# Plots the left and right polynomials on the lane lines
		#plt.plot(left_fitx, ploty, color='yellow')
		#plt.plot(right_fitx, ploty, color='yellow')
		if self.processing_video:
			self.first = False

		return out_img
	
	def process_video(self, video_path, write_path=None, subclip=None):
		'''
		This function takes in the path of a video file to process, and writes an annotated
		video to the output_dir with the name write_path, or the original name of the video
		if write_path=None. Set subclip to a number of seconds if testing a short portion.
		'''
		self.processing_video = True
		self.first = True
		if write_path is None:
			outpath = self.output_dir + video_path.split('/')[-1]
		else:
			outpath = self.output_dir + write_path
		clip = VideoFileClip(video_path)
		if subclip is not None:
			clip = clip.subclip(0, subclip)
		mod_clip = clip.fl_image(self.process_image)
		mod_clip.write_videofile(outpath, audio=False)

		# Reset values after completion
		self.processing_video = False
		self.right_fits = []
		self.right_fits_metric = []
		self.left_fits = []
		self.left_fits_metric = []
		
		pass