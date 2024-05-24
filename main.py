from src.video_processor import process_video
from src.player_detector import PlayerDetector
from src.tennis_ball_detector import TennisBallDetector

if __name__ == '__main__':
	# 1. Open Video To Process
	match_name = 'match02'
	# 2. Process the video into frames
	player_detector = PlayerDetector('yolov8x')
	ball_detector = TennisBallDetector('models/tennis_ball/last.pt')
	process_video(match_name, people_detector=player_detector, ball_detector=ball_detector)
	print('processed')