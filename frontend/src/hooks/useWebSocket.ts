import { useState, useRef, useCallback, useEffect } from 'react';
import type {
  Direction,
  DetectionResult,
  FrameMessage,
  DirectionMessage,
  StatusMessage,
  WebSocketMessage,
  UseWebSocketReturn,
} from '@/types';

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_INTERVAL = 3000;
const HEARTBEAT_INTERVAL = 15000;
const FRAME_TIMEOUT_MS = 10000;  // 10秒未收到帧则主动重连

export function useWebSocket(): UseWebSocketReturn {
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [lastError, setLastError] = useState('');
  const [frames, setFrames] = useState<Record<string, string>>({});
  const [detections, setDetections] = useState<Record<string, DetectionResult>>({});
  const [direction, setDirection] = useState<Direction>('none');
  const [directionConfidence, setDirectionConfidence] = useState(0);
  const [directionTimestamp, setDirectionTimestamp] = useState(0);
  const [fps, setFps] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameTimestampsRef = useRef<number[]>([]);
  const connectingRef = useRef(false);
  const lastFrameTimeRef = useRef(0);

  // Calculate FPS from frame timestamps
  const updateFps = useCallback(() => {
    const now = Date.now();
    frameTimestampsRef.current = frameTimestampsRef.current.filter(
      (t) => now - t < 1000
    );
    setFps(frameTimestampsRef.current.length);
  }, []);

  // Get WebSocket URL based on current protocol
  const getWebSocketUrl = useCallback((): string => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/video`;
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (connectingRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    connectingRef.current = true;
    setConnecting(true);

    try {
      const url = getWebSocketUrl();
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setConnecting(false);
        setLastError('');
        reconnectAttemptsRef.current = 0;
        connectingRef.current = false;
        lastFrameTimeRef.current = Date.now();

        // Start heartbeat
        heartbeatTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'ping' }));
          }
        }, HEARTBEAT_INTERVAL);
      };

      ws.onmessage = (event) => {
        // 更新最后收到帧的时间
        lastFrameTimeRef.current = Date.now();

        try {
          const msg: WebSocketMessage = JSON.parse(event.data);

          // New backend payload (without type) compatibility
          if ('camera_id' in msg && 'frame' in msg) {
            const frameMsg = msg as unknown as FrameMessage & { frame?: string };
            const frameData = frameMsg.data || frameMsg.frame || '';
            if (frameData) {
              setFrames((prev) => ({
                ...prev,
                [frameMsg.camera_id]: frameData,
              }));
              frameTimestampsRef.current.push(Date.now());
              updateFps();
            }
            const raw = msg as unknown as {
              camera_id?: string;
              person_count?: number;
              detections?: Array<{
                gesture?: 'greeting' | 'hailing' | 'hand_up' | 'none';
                gesture_conf?: number;
                confidence?: number;
              }>;
              timestamp?: number;
            };
            if (raw.camera_id) {
              const dets = Array.isArray(raw.detections) ? raw.detections : [];
              let bestGesture: 'greeting' | 'hailing' | 'hand_up' | 'none' = 'none';
              let bestGestureConf = 0;
              for (const d of dets) {
                const g = d.gesture ?? 'none';
                const c = d.gesture_conf ?? d.confidence ?? 0;
                if (g !== 'none' && c > bestGestureConf) {
                  bestGesture = g;
                  bestGestureConf = c;
                }
              }
              setDetections((prev) => ({
                ...prev,
                [raw.camera_id as string]: {
                  camera_id: raw.camera_id as string,
                  person_count:
                    typeof raw.person_count === 'number'
                      ? raw.person_count
                      : dets.length,
                  gesture: bestGesture,
                  gesture_confidence: bestGestureConf,
                  poses: [],
                  timestamp: raw.timestamp ?? Date.now(),
                },
              }));
            }
            if ('direction' in (msg as unknown as Record<string, unknown>)) {
              const dirPayload = msg as unknown as {
                direction?: Direction;
                confidence?: number;
                timestamp?: number;
              };
              setDirection(dirPayload.direction ?? 'none');
              setDirectionConfidence(dirPayload.confidence ?? 0);
              setDirectionTimestamp(dirPayload.timestamp ?? Date.now());
            }
            return;
          }

          switch (msg.type) {
            case 'frame': {
              const frameMsg = msg as FrameMessage;
              setFrames((prev) => ({
                ...prev,
                [frameMsg.camera_id]: frameMsg.data,
              }));
              frameTimestampsRef.current.push(Date.now());
              updateFps();
              break;
            }
            case 'direction': {
              const dirMsg = msg as DirectionMessage;
              setDirection(dirMsg.direction);
              setDirectionConfidence(dirMsg.confidence);
              setDirectionTimestamp(dirMsg.timestamp);
              break;
            }
            case 'status': {
              const statusMsg = msg as StatusMessage;
              if (statusMsg.fps) {
                setFps(statusMsg.fps);
              }
              break;
            }
          }
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onclose = () => {
        setConnected(false);
        setConnecting(false);
        connectingRef.current = false;

        if (heartbeatTimerRef.current) {
          clearInterval(heartbeatTimerRef.current);
          heartbeatTimerRef.current = null;
        }

        // Attempt reconnection
        if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttemptsRef.current += 1;
          setLastError(
            `连接中断，正在重连 (${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})`
          );
          reconnectTimerRef.current = setTimeout(() => {
            connect();
          }, RECONNECT_INTERVAL);
        } else {
          setLastError('WebSocket 连接失败，请检查服务状态与网络。');
        }
      };

      ws.onerror = () => {
        setLastError('WebSocket 连接错误。');
        ws.close();
      };
    } catch {
      setConnecting(false);
      connectingRef.current = false;
      setLastError('初始化 WebSocket 失败。');
    }
  }, [getWebSocketUrl, updateFps]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // Send message through WebSocket
  const sendMessage = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  // Manual reconnect
  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttemptsRef.current = 0;
    setTimeout(() => connect(), 100);
  }, [disconnect, connect]);

  // Connect on mount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // FPS decay when no frames arrive
  useEffect(() => {
    const interval = setInterval(() => {
      updateFps();
    }, 500);
    return () => clearInterval(interval);
  }, [updateFps]);

  // 帧超时检测：连接打开但长时间未收到帧，主动重连
  // 这是防御后端推流任务挂起或丢失的关键机制
  useEffect(() => {
    const interval = setInterval(() => {
      if (
        wsRef.current?.readyState === WebSocket.OPEN &&
        lastFrameTimeRef.current > 0 &&
        Date.now() - lastFrameTimeRef.current > FRAME_TIMEOUT_MS
      ) {
        console.warn(
          `[WebSocket] ${FRAME_TIMEOUT_MS}ms 未收到帧，主动断开重连`
        );
        reconnect();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [reconnect]);

  return {
    connected,
    connecting,
    lastError,
    frames,
    detections,
    direction,
    directionConfidence,
    directionTimestamp,
    fps,
    sendMessage,
    reconnect,
  };
}
