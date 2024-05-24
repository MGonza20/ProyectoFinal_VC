from src.video_processor import process_video
from src.player_detector import PlayerDetector
from src.tennis_ball_detector import TennisBallDetector
from src.utils import CourtDetector


if __name__ == '__main__':
	# 1. Open Video To Process
	match_name = 'match01'
	# 2. Process the video into frames
	player_detector = PlayerDetector('yolov8x')
	ball_detector = TennisBallDetector('models/tennis_ball/last.pt')
	model_path = 'models/keypoints_model.pth'
	process_video(match_name, people_detector=player_detector, ball_detector=ball_detector, court_detector=CourtDetector(model_path=model_path))
	
	print('processed')
