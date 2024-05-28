from ultralytics import YOLO

class TennisBallDetector:
		def __init__(self, model_path):
				self.model = YOLO(model_path)

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
			
			