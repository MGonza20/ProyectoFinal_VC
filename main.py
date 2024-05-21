from src.video_processor import process_video
from src.utils import CourtDetector


if __name__ == '__main__':
	# 1. Open Video To Process
	match_name = 'match01'
	# 2. Process the video into frames
	process_video(match_name=match_name, court_detector=CourtDetector())
	print('processed')
