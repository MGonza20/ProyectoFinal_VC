import cv2
import numpy as np
from src.utils import convert_meters_distance_to_pixels, convert_pixels_distance_to_meters, get_box_height, get_box_center, euclidean_distance
from src.constants import HALF_COURT_HEIGHT, COURT_WIDTH,DOUBLES_ALLEY, NO_MANS_LAND, SINGLE_LINE_WIDTH, BALL_HEIGHT
class MiniCourt:	
	def __init__(self, frame):
		self.drawing_rectangle_width =	250
		self.drawing_rectangle_height = 500
		self.padding = 20
		self.buffer = 50
		self.set_canvas_background_box_position(frame)
		self.set_mini_court_position()
		self.set_court_drawing_key_points()
		self.set_court_drawing_line_points()
		self.set_net_drawing_points()

	def set_canvas_background_box_position(self, frame):
		frame = frame.copy()
		self.end_x = frame.shape[1] - self.buffer
		self.end_y = self.buffer + self.drawing_rectangle_height
		self.start_x = self.end_x - self.drawing_rectangle_width
		self.start_y = self.end_y - self.drawing_rectangle_height

	def set_mini_court_position(self):
		self.court_start_x = self.start_x + self.padding
		self.court_end_x = self.end_x - self.padding
		self.court_start_y = self.start_y + self.padding
		self.court_end_y = self.end_y - self.padding
		self.court_drawing_width = self.court_end_x - self.court_start_x
		self.court_drawing_height = self.court_end_y - self.court_start_y

	def set_court_drawing_key_points(self):
		drawing_key_points = [0]*28 # List of 14 points 
		# Point 0
		drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
		# Point 1
		drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
		# Point 2
		drawing_key_points[4] = int(self.court_start_x)
		drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(HALF_COURT_HEIGHT*2)
		# Point 3
		drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
		drawing_key_points[7] = drawing_key_points[5]
	  # Point 4
		drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(DOUBLES_ALLEY)
		drawing_key_points[9] = drawing_key_points[1]
		# Point 5
		drawing_key_points[10] = drawing_key_points[8]
		drawing_key_points[11] = drawing_key_points[5]
		# Point 6
		drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(DOUBLES_ALLEY)
		drawing_key_points[13] = drawing_key_points[1]
		# Point 7
		drawing_key_points[14] = drawing_key_points[12]
		drawing_key_points[15] = drawing_key_points[5]
		# Point 8
		drawing_key_points[16] = drawing_key_points[8]
		drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(NO_MANS_LAND)
		# Point 9
		drawing_key_points[18] = drawing_key_points[12]
		drawing_key_points[19] = drawing_key_points[17]
		# Point 10
		drawing_key_points[20] = drawing_key_points[16]
		drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(NO_MANS_LAND)
		# Point 11
		drawing_key_points[22] = drawing_key_points[18]
		drawing_key_points[23] = drawing_key_points[21]
		# Point 12
		drawing_key_points[24] = drawing_key_points[16] + self.convert_meters_to_pixels(SINGLE_LINE_WIDTH/2)
		drawing_key_points[25] = drawing_key_points[17]
		# Point 13
		drawing_key_points[26] = drawing_key_points[18] - self.convert_meters_to_pixels(SINGLE_LINE_WIDTH/2)
		drawing_key_points[27] = drawing_key_points[21]
		self.court_drawing_key_points = drawing_key_points
	
	def set_court_drawing_line_points(self):
		drawing_line_points = [0]*36
		# Point 0
		drawing_line_points[0], drawing_line_points[1] = self.court_drawing_key_points[0], self.court_drawing_key_points[1]
		# Point 1
		drawing_line_points[2], drawing_line_points[3] = self.court_drawing_key_points[4], self.court_drawing_key_points[5]
		# Point 2 (point 4)
		drawing_line_points[4], drawing_line_points[5] = self.court_drawing_key_points[8], self.court_drawing_key_points[9]
		# Point 3 (point 5)
		drawing_line_points[6], drawing_line_points[7] = self.court_drawing_key_points[10], self.court_drawing_key_points[11]
		# Point 4 (point 1)
		drawing_line_points[8], drawing_line_points[9] = self.court_drawing_key_points[2], self.court_drawing_key_points[3]
		# Point 5 (point 3)
		drawing_line_points[10], drawing_line_points[11] = self.court_drawing_key_points[6], self.court_drawing_key_points[7]
		# Point 6 (point 2)
		drawing_line_points[12], drawing_line_points[13] = self.court_drawing_key_points[4], self.court_drawing_key_points[5]
		# Point 7 (point 3)
		drawing_line_points[14], drawing_line_points[15] = self.court_drawing_key_points[6], self.court_drawing_key_points[7]
		# Point 8 (point 0)
		drawing_line_points[16], drawing_line_points[17] = self.court_drawing_key_points[0], self.court_drawing_key_points[1]
		# Point 9 (point 1)
		drawing_line_points[18], drawing_line_points[19] = self.court_drawing_key_points[2], self.court_drawing_key_points[3]
		# Point 10 (point 12)
		drawing_line_points[20], drawing_line_points[21] = self.court_drawing_key_points[24], self.court_drawing_key_points[25]
		# Point 11 (point 13)
		drawing_line_points[22], drawing_line_points[23] = self.court_drawing_key_points[26], self.court_drawing_key_points[27]
		# Point 12 (point 8)
		drawing_line_points[24], drawing_line_points[25] = self.court_drawing_key_points[16], self.court_drawing_key_points[17]
		# Point 13 (point 9)
		drawing_line_points[26], drawing_line_points[27] = self.court_drawing_key_points[18], self.court_drawing_key_points[19]
		# Point 14 (point 7)
		drawing_line_points[28], drawing_line_points[29] = self.court_drawing_key_points[14], self.court_drawing_key_points[15]
		# Point 15 (point 6)
		drawing_line_points[30], drawing_line_points[31] = self.court_drawing_key_points[12], self.court_drawing_key_points[13]
		# Point 16 (point 10)
		drawing_line_points[32], drawing_line_points[33] = self.court_drawing_key_points[20], self.court_drawing_key_points[21]
		# Point 17 (point 11)
		drawing_line_points[34], drawing_line_points[35] = self.court_drawing_key_points[22], self.court_drawing_key_points[23]
		
			
		self.court_drawing_line_points = drawing_line_points

	def set_net_drawing_points(self):
		self.net_drawing_points = [0]*4
		self.net_drawing_points[0], self.net_drawing_points[1] = self.start_x, self.start_y + self.convert_meters_to_pixels(HALF_COURT_HEIGHT)
		self.net_drawing_points[2], self.net_drawing_points[3] = self.end_x, self.start_y + self.convert_meters_to_pixels(HALF_COURT_HEIGHT)
	
	def convert_meters_to_pixels(self, meters):
		return convert_meters_distance_to_pixels(meters, COURT_WIDTH, self.court_drawing_width)
	
	def convert_pixels_to_meters(self, pixels):
		return convert_pixels_distance_to_meters(pixels, COURT_WIDTH, self.end_x - self.start_x)

	

	def draw_background(self, frame):
		shapes = np.zeros_like(frame, np.uint8)
		cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
		out = frame.copy()
		alpha = 0.5
		mask = shapes.astype(bool)
		out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
		return out
	
	def draw_court(self, frame):
		for i in range(0, len(self.court_drawing_key_points), 2):
			x = int(self.court_drawing_key_points[i])
			y = int(self.court_drawing_key_points[i+1])
			cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
		for i in range(0, len(self.court_drawing_line_points), 4):
			x1 = int(self.court_drawing_line_points[i])
			y1 = int(self.court_drawing_line_points[i+1])
			x2 = int(self.court_drawing_line_points[i+2])
			y2 = int(self.court_drawing_line_points[i+3])
			cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
		cv2.line(frame, (int(self.net_drawing_points[0]), int(self.net_drawing_points[1])), (int(self.net_drawing_points[2]), int(self.net_drawing_points[3])), (255, 0, 0), 2)
		return frame
	def draw_ball_in_mini_court(self, frame, ball_detection, keypoint_detection):
		if 0 in ball_detection:
			ball_height_pixels = get_box_height(ball_detection[0])
			ball_center = get_box_center(ball_detection[0])
			# Find closest keypoint index
			closest_keypoint = None
			closest_distance = 100000
			for i in range(0, len(keypoint_detection), 2):
				x = keypoint_detection[i]
				y = keypoint_detection[i+1]
				distance = euclidean_distance((x, y), ball_center)
				if distance < closest_distance:
					closest_distance = distance
					closest_keypoint = i
			key_point_index = closest_keypoint // 2
			drawing_key_point_x = self.court_drawing_key_points[key_point_index]
			drawing_key_point_y = self.court_drawing_key_points[key_point_index+1]

			# Draw a blue circle on the keypoint
			cv2.circle(frame, (int(drawing_key_point_x), int(drawing_key_point_y)), 5, (255, 0, 0), -1)
			
		return frame

	
	def draw_mini_court(self, frames, ball_detections, key_points):
		output_frames = []
		for i, frame in enumerate(frames):
			frame = self.draw_background(frame)
			frame = self.draw_court(frame)
			frame = self.draw_ball_in_mini_court(frame, ball_detections[i], key_points[i])
			output_frames.append(frame)
		return output_frames
