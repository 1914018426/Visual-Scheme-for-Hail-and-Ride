// ========== 基础类型 ==========

export type CameraId = 'front' | 'back' | 'left' | 'right';

export type Protocol = 'rtsp' | 'rtmp' | 'http' | 'webrtc' | 'local' | 'file';

export type Gesture = 'hand_up' | 'wave' | 'none';

export type Direction = 'forward' | 'backward' | 'left' | 'right' | 'none';

// ========== 关键点类型 ==========

export interface Keypoint {
  x: number;
  y: number;
  score: number;
}

export interface Pose {
  keypoints: Keypoint[];
  score: number;
}

// ========== 检测结果 ==========

export interface DetectionResult {
  camera_id: CameraId;
  person_count: number;
  gesture: Gesture;
  gesture_confidence: number;
  poses: Pose[];
  timestamp: number;
}

// ========== 消息类型 ==========

export interface FrameMessage {
  type: 'frame';
  camera_id: CameraId;
  data: string; // base64 encoded image
  timestamp: number;
}

export interface DirectionMessage {
  type: 'direction';
  direction: Direction;
  confidence: number;
  source_camera: CameraId;
  timestamp: number;
}

export interface StatusMessage {
  type: 'status';
  camera_id: CameraId;
  online: boolean;
  fps: number;
}

export type WebSocketMessage = FrameMessage | DirectionMessage | StatusMessage;

// ========== 摄像头配置 ==========

export interface CameraConfig {
  id: CameraId;
  protocol: Protocol;
  source: string;
  enabled: boolean;
  label: string;
}

export interface CameraConfigs {
  front: CameraConfig;
  back: CameraConfig;
  left: CameraConfig;
  right: CameraConfig;
}

export type PullMethod = 'webrtc' | 'rtmp';

export interface CameraNameMapping {
  front: string;
  back: string;
  left: string;
  right: string;
}

export interface VehicleCameraProfile {
  id: string;
  name: string;
  vehicleCount: number;
  cameras: CameraNameMapping;
}

export interface CameraProfileBundle {
  id: string;
  name: string;
  defaultPullMethod: PullMethod;
  webrtcPlayerBaseUrl: string;
  webrtcApiBaseUrl: string;
  rtmpBaseUrl: string;
  profiles: VehicleCameraProfile[];
}

export interface CameraProfileDocument {
  version: string;
  bundles: CameraProfileBundle[];
}

// ========== WebSocket Hook 返回类型 ==========

export interface UseWebSocketReturn {
  connected: boolean;
  connecting: boolean;
  lastError: string;
  frames: Record<CameraId, string>;
  detections: Record<CameraId, DetectionResult>;
  direction: Direction;
  directionConfidence: number;
  directionTimestamp: number;
  fps: number;
  sendMessage: (msg: object) => void;
  reconnect: () => void;
}

// ========== 常量 ==========

export const CAMERA_LABELS: Record<CameraId, string> = {
  front: '前视摄像头',
  back: '后视摄像头',
  left: '左视摄像头',
  right: '右视摄像头',
};

export const CAMERA_SHORT_LABELS: Record<CameraId, string> = {
  front: '前视',
  back: '后视',
  left: '左视',
  right: '右视',
};

export const DIRECTION_LABELS: Record<Direction, string> = {
  forward: '前进',
  backward: '后退',
  left: '左转',
  right: '右转',
  none: '停止',
};

export const GESTURE_LABELS: Record<Gesture, string> = {
  hand_up: '举手',
  wave: '挥手',
  none: '无手势',
};

export const PROTOCOL_LABELS: Record<Protocol, string> = {
  rtsp: 'RTSP',
  rtmp: 'RTMP',
  http: 'HTTP',
  webrtc: 'WebRTC',
  local: '本地摄像头',
  file: '视频文件',
};

export const DEFAULT_CAMERA_CONFIGS: CameraConfigs = {
  front: {
    id: 'front',
    protocol: 'rtsp',
    source: '',
    enabled: true,
    label: '前视摄像头',
  },
  back: {
    id: 'back',
    protocol: 'rtsp',
    source: '',
    enabled: true,
    label: '后视摄像头',
  },
  left: {
    id: 'left',
    protocol: 'rtsp',
    source: '',
    enabled: true,
    label: '左视摄像头',
  },
  right: {
    id: 'right',
    protocol: 'rtsp',
    source: '',
    enabled: true,
    label: '右视摄像头',
  },
};

export const DEFAULT_CAMERA_PROFILE_DOCUMENT: CameraProfileDocument = {
  version: '1.0.0',
  bundles: [
    {
      id: 'lingubot-default',
      name: 'Lingubot 默认场景',
      defaultPullMethod: 'webrtc',
      webrtcPlayerBaseUrl: 'https://sztu-video.lingubot.cn/webrtc/',
      webrtcApiBaseUrl: 'https://sztu-video.lingubot.cn/index/api/webrtc',
      rtmpBaseUrl: 'rtmp://sztu-video.lingubot.cn/live',
      profiles: [
        {
          id: 'hasun-test',
          name: 'hasun-test',
          vehicleCount: 1,
          cameras: {
            front: 'KRIPC_93002871_100',
            back: 'KRIPC_93003115_60',
            left: 'KRIPC_93002892_28',
            right: 'KRIPC_93003115_86',
          },
        },
        {
          id: 'lingu_test2',
          name: 'lingu_test2',
          vehicleCount: 2,
          cameras: {
            front: 'KRIPC_93002892_45',
            back: 'KRIPC_93003115_91',
            left: 'KRIPC_93003115_96',
            right: 'KRIPC_93003115_75',
          },
        },
      ],
    },
  ],
};

// ========== 骨骼连接定义 ==========

export const SKELETON_CONNECTIONS: [number, number][] = [
  [5, 7],   // left_shoulder -> left_elbow
  [7, 9],   // left_elbow -> left_wrist
  [6, 8],   // right_shoulder -> right_elbow
  [8, 10],  // right_elbow -> right_wrist
  [5, 6],   // left_shoulder -> right_shoulder
  [5, 11],  // left_shoulder -> left_hip
  [6, 12],  // right_shoulder -> right_hip
  [11, 12], // left_hip -> right_hip
];

export const KEYPOINT_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
];
