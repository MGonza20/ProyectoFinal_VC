
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
from src.kps_postprocessing import fix_keypoints

class CourtDetector:
		def __init__(self, model_path):
				self.model = models.resnet50(pretrained=False)
				self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
				self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

				self.transform = transforms.Compose([
						transforms.ToPILImage(),
						transforms.Resize((224, 224)),
						transforms.ToTensor(),
						transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])
				
		def detect_frames(self, frames, output_path=None, stub_path=None):
				detections = []
				if stub_path is not None:
						with open(stub_path, 'rb') as f:
								detections = pickle.load(f)
								return detections
				for frame in frames:
						detections.append(self.detect(frame))
				if output_path is not None:
						with open(output_path, 'wb') as f:
								pickle.dump(detections, f)
				return detections		 
					

		def detect(self, frame): 

				img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				img_tensor = self.transform(img_rgb).unsqueeze(0)

				with torch.no_grad(): outputs = self.model(img_tensor)

				keypoints = outputs.squeeze().cpu().numpy()
				original_h, original_w = img_rgb.shape[:2]

				keypoints[::2] *= original_w/224.0
				keypoints[1::2] *= original_h/224.0
				keypoints = fix_keypoints(keypoints, frame)
                
				return keypoints
	