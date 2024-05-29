import cv2
from src.minicourt import MiniCourt

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
	# Frames
	frames = read_frames(cap) 

	# Detect ball
	ball_detections = ball_detector.detect_frames(frames, stub_path=f'output/stubs/balls-{match_name}.pkl')
	# Detect the court
	if court_detector is not None:
		court_lines = court_detector.detect_frames(frames, stub_path=f'output/stubs/court-{match_name}.pkl')
		# ===== AQUI CORRECCION ======
		# ===== AQUI CORRECCION ======
		# Detect the players
	player_detections = people_detector.detect_frames(frames, stub_path=f'output/stubs/players-{match_name}.pkl')
	player_detections = people_detector.filter_players(player_detections, court_lines)

	# Court drawing
	mini_court = MiniCourt(frame=frames[0])
	
	frames = mini_court.draw_mini_court(frames, ball_detections, court_lines)

	# Draw frame number on top left
	for i, frame in enumerate(frames):
		cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	# Draw the court
	for i, frame in enumerate(frames):
		if court_detector is not None:
			for indx in range(0, len(court_lines[i]), 2):
				x = int(court_lines[i][indx])
				y = int(court_lines[i][indx+1])
				cv2.putText(frame, str(indx//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.circle(frame, (x,y), 5, (0, 0, 255), -1)
	# Draw the boxes
	for i, frame in enumerate(frames):
		# Draw the players
		for track_id, box in player_detections[i].items():
			frame = draw_player_box(frame, track_id, box)
		# Draw the ball
		if 0 in ball_detections[i]:
			frame = draw_ball_box(frame, ball_detections[i][0])
		# Display the frame
		cv2.imshow('Processed Frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'): break
		# Out the frame
		out.write(frame)

	cap.release()
	out.release()


def read_frames(cap):
	'''
	Read the frames of a video
	Parameters:
		cap: The video capture object
	Returns:
		frames: A list of frames
	'''
	frames = []
	while True:
		ret, frame = cap.read()
		if not ret: break
		frames.append(frame)
	return frames

def draw_player_box(frame, id, box):
	# box is a list of x,y,x,y
	# Id is the id of the player
	x1, y1, x2, y2 = box
	cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
	cv2.putText(frame, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	return frame

def draw_ball_box(frame, box):
	x1, y1, x2, y2 = box
	cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
	return frame