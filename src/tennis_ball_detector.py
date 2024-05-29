from ultralytics import YOLO
import pickle

class TennisBallDetector:
		def __init__(self, model_path):
				self.model = YOLO(model_path)


		def detect_frames(self, frames, output_path=None, stub_path=None):
			'''
			Process the frames and return the detected balls.
			'''
			ball_positions = []
			if stub_path is not None:
				with open(stub_path, 'rb') as f:
					ball_positions = pickle.load(f)
					return ball_positions
			for i, frame in enumerate(frames):
				results = self.results(frame)
				ball_positions.append(results)
			if output_path is not None:
				with open(output_path, 'wb') as f:
					pickle.dump(ball_positions, f)
			return ball_positions

		def results(self, frame):
			'''
			Process the frame and return the detected balls.
			'''
			results = self.model.predict(frame, conf=0.15)[0]
			id_name_dict = results.names
			ball_dict = {}
			for box in results.boxes:
				result = box.xyxy.tolist()[0]
				ball_dict[0] = result
			return ball_dict
			
			