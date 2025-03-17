### detect_lanes.py
import cv2
from moviepy.editor import VideoFileClip
import torch
from models import LaneNet
from utils import Lanes
from utils import road_lines

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LaneNet().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()
    lanes = Lanes()
    clip_input = VideoFileClip("mountain-view-ca.mp4")
    vid_output = 'output_video.mp4'
    def process_frame(frame):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_bgr = road_lines(frame_bgr, model, lanes)
        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    vid_clip = clip_input.fl_image(process_frame)
    vid_clip.write_videofile(vid_output, audio=False)

if __name__ == '__main__':
    main()