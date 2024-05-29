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
		
		def filter_players(self, player_positions, court_detections):
			filtered_players = []
			first_frame_court_detections = court_detections[0]
			first_frame_players = player_positions[0]
			players_dict = {} # key: track_id, value: shortest distance to the court
			for player_id, player_box in first_frame_players.items():
				x1, y1, x2, y2 = player_box
				player_center = ((x2 + x1) / 2, (y2 + y1) / 2)
				min_distance = float('inf')
				for i in range(0, len(first_frame_court_detections), 2):
					court_point = (first_frame_court_detections[i], first_frame_court_detections[i+1])
					distance = self.euclidean_distance(player_center, court_point)
					if distance < min_distance:
						min_distance = distance
			
				players_dict[player_id] = min_distance
			
			# Return the first two players
			players_dict = dict(sorted(players_dict.items(), key=lambda item: item[1]))
			first_two_players = list(players_dict.keys())[:2]
			for player_dict in player_positions:
				filtered_players.append({player_id: player_dict[player_id] for player_id in first_two_players})
			return filtered_players

		def euclidean_distance(self, point1, point2):
			x1, y1 = point1
			x2, y2 = point2
			return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5