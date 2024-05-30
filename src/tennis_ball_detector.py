from ultralytics import YOLO
import pickle
import pandas as pd

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
		
		def interpolate(self, ball_detections):
			# Replace empty detections with None
			for i, ball in enumerate(ball_detections):
				if not ball:
					ball_detections[i] = None

			data = {
				'frame': list(range(len(ball_detections))),
				'ball_position': ball_detections
			}
			df = pd.DataFrame(data)
			df.set_index('frame', inplace=True)
			df[['x1', 'y1', 'x2', 'y2']] = df['ball_position'].apply(self.extract_coordinates)
			df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].interpolate()
			df['ball_position'] = df.apply(self.assemble_coordinates, axis=1)
			ball_detections = df['ball_position'].tolist()
			# Convert None to empty dict
			for i, ball in enumerate(ball_detections):
				if ball is None:
					ball_detections[i] = {}
			return ball_detections
			
		def extract_coordinates(self, row):
			if row and 0 in row:
					return pd.Series(row[0])
			else:
					return pd.Series([None, None, None, None],dtype=float)
			
		def assemble_coordinates(self, row):
			if not pd.isna(row['x1']):
					return {0: [row['x1'], row['y1'], row['x2'], row['y2']]}
			else:
					return None