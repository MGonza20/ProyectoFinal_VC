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
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

	# 2. Process the video into frames
	while True:
		ret, frame = cap.read()
		if not ret: break
		
		# ===== FRAME PROCESSING =====
		
		# Detect the ball

		# Detect the court
		if court_detector is not None:
			court_lines = court_detector.detect(frame)
			for line in court_lines: cv2.line(frame, line[0], line[1], (255, 0, 0), 2)

		# Detect the people

		# Display the frame
		cv2.imshow('Processed Frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'): break

		# Out the frame
		out.write(frame)

	cap.release()
	out.release()
	cv2.destroyAllWindows()