from src.video_processor import process_video


if __name__ == '__main__':
	# 1. Open Video To Process
	match_name = 'match01'
	# 2. Process the video into frames
	process_video(match_name)
	print('processed')