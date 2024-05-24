import cv2

def process_video(match_name='match01', ball_detector=None,court_detector=None,people_detector=None):
	'''
	Process the video of a match and return the processed frames
	Parameters:
		match_name: The name of the match to process
		ball_detector: The BallDetector object to use
		court_detector: The CourtDetector object to use
		people_detector: The PeopleDetector object to use
	'''
	# 1. Open Video To Process
	video_path = f'data/raw/videos/{match_name}.mp4'
	output_video_path = f'output/videos/{match_name}.mp4'
	cap = cv2.VideoCapture(video_path)
	# Out Video Configuration
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
	count_frame = 0
	# 2. Process the video into frames
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		
		# ===== FRAME PROCESSING =====
		
		# Detect the ball
		ball_results = ball_detector.results(frame)
		
		# Detect the court
		# Detect the people
		results = people_detector.detect(frame)
		# Result is a dict of track_id: box
		for track_id, box in results.items():
			frame = draw_player_box(frame, track_id, box)
		if 0 in ball_results:
			frame = draw_ball_box(frame, ball_results[0])
		# Out the frame
		out.write(frame)
		count_frame += 1
		if count_frame == 3000:
			print(f'Processed {count_frame} frames')
			break

	cap.release()
	out.release()
	cv2.destroyAllWindows()

def draw_player_box(frame, id, box):
	# box is a list of x,y,x,y
	# Id is the id of the player
	x1, y1, x2, y2 = box
	cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
	cv2.putText(frame, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	return frame

def draw_ball_box(frame, box):
	x1, y1, x2, y2 = box
	cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
	return frame