from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch

class HumanDetector:
    def __init__(self, model_path: str = "/app/models/yolo11n.pt",
                 conf_threshold: float = 0.5,
                 device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        
        # 只检测"person"类别 (COCO index 0)
        self.classes = [0]
        
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        检测人体，返回人体框列表
        """
        results = self.model(frame, 
                           classes=self.classes, 
                           conf=self.conf_threshold,
                           device=self.device,
                           verbose=False)
        
        humans = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                humans.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "roi": frame[int(y1):int(y2), int(x1):int(x2)]
                })
        
        return humans
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        批量检测，利用双4090并行处理
        """
        results = self.model(frames, 
                           classes=self.classes,
                           conf=self.conf_threshold,
                           device=self.device,
                           batch=len(frames),
                           verbose=False)
        
        batch_humans = []
        for result in results:
            humans = []
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                frame_idx = int(box.orig_shape[0])  # 获取原始帧索引
                
                humans.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf)
                })
            batch_humans.append(humans)
        
        return batch_humans
