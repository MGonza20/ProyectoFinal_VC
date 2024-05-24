from ultralytics import YOLO

class PlayerDetector:
		def __init__(self, model_path):
				self.model = YOLO(model_path)
		
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
			