from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from yolo_detector import HumanDetector
from gesture_recognizer import GestureRecognizer

app = FastAPI(title="无人零售车视觉识别API", version="1.0.0")

# 添加 CORS 支持 - 允许所有来源（生产环境请限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型
detector = HumanDetector()
gesture_recognizer = GestureRecognizer()

@app.post("/detect")
async def detect_gesture(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image"}
    
    gesture_recognizer.reset_history()
    humans = detector.detect(frame)
    results = []
    
    for i, human in enumerate(humans):
        gesture_result = gesture_recognizer.recognize(
            human["roi"], human["bbox"], track_index=i
        )
        results.append({
            "bbox": human["bbox"],
            "human_confidence": human["confidence"],
            "gesture": {
                "is_waving": gesture_result.is_waving,
                "confidence": gesture_result.confidence,
                "type": gesture_result.gesture_type,
                "wave_style": gesture_result.wave_style,
            }
        })
    
    return {
        "humans_detected": len(humans),
        "waving_detected": len([r for r in results if r["gesture"]["is_waving"]]),
        "targets": results
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 明确接受所有 WebSocket 连接
    await websocket.accept()
    gesture_recognizer.reset_history()
    
    try:
        while True:
            # 接收二进制帧数据
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # 双阶段检测
            humans = detector.detect(frame)
            response = {
                "timestamp": asyncio.get_event_loop().time(),
                "targets": []
            }
            
            for i, human in enumerate(humans):
                gesture = gesture_recognizer.recognize(
                    human["roi"], human["bbox"], track_index=i
                )
                response["targets"].append({
                    "bbox": human["bbox"],
                    "gesture": gesture.gesture_type,
                    "confidence": gesture.confidence,
                    "is_target": gesture.is_waving,
                    "wave_style": gesture.wave_style,
                })
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print(f"客户端断开: {websocket.client}")
    except Exception as e:
        print(f"WebSocket错误: {e}")
    finally:
        await websocket.close()

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "gpu_available": detector.device == "cuda",
        "models": ["yolov8n", "mediapipe_hands"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
