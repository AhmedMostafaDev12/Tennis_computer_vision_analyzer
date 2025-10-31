import torch 
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineDetector:
    def __init__(self,model_path):
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)  # 14 keypoints with (x,y) coordinates
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def predict(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #unsqueeze to add batch dimension if the image is single 
        # you must put it in a list otherwise it will throw an error
        # the model expects a batch of images
        input_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        key_points = outputs.squeeze().cpu().numpy()

        # the key points are normalized to [0,1], we need to denormalize them to the original image size
        original_h, original_w, _ = img_rgb.shape
        key_points[0::2] *= original_w/224.0  # x coordinates
        key_points[1::2] *= original_h/224.0  # y coordinates

        return key_points
    
    def draw_keypoints(self, frame, key_points):
        for i in range(0, len(key_points), 2):
            x = int(key_points[i])
            y = int(key_points[i+1])
            cv2.putText(frame, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        return frame
    
    def draw_keypoints_on_video(self, video_frames, keypoints_list):
        output_frames = []
        for frame in  video_frames:
            output_frame = self.draw_keypoints(frame, keypoints_list)
            output_frames.append(output_frame)
        return output_frames