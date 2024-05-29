import pickle
from ultralytics import YOLO

class PlayerDetector:
		def __init__(self, model_path):
				self.model = YOLO(model_path)

		def detect_frames(self, frames, output_path=None, stub_path=None):
			'''
			Process the frames and return the detected players.
			'''
			player_positions = []

			if stub_path is not None:
				with open(stub_path, 'rb') as f:
					player_positions = pickle.load(f)
					return player_positions
				
			for i, frame in enumerate(frames):
				results = self.detect(frame)
				player_positions.append(results)
			
			# Save the results
			if output_path is not None:
				with open(output_path, 'wb') as f:
					pickle.dump(player_positions, f)

			return player_positions
		
		def detect(self, frame):
			results = self.model.track(frame, persist=True)[0]
			id_name_dict = results.names
			player_dict = {}
			for box in results.boxes:
				if box.id is None:
					continue
				track_id = int(box.id.tolist()[0])
				result = box.xyxy.tolist()[0]
				object_cls_id = box.cls.tolist()[0]
				object_cls_name = id_name_dict[object_cls_id]
				if object_cls_name == 'person':
					player_dict[track_id] = result
			return player_dict
			