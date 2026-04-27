import { useState, useRef, useCallback, useEffect } from 'react';

export interface LogEntry {
  timestamp: number;
  level: string;
  logger: string;
  message: string;
}

export interface UseLogWebSocketReturn {
  connected: boolean;
  logs: LogEntry[];
  clearLogs: () => void;
}

const MAX_LOGS = 500;

export function useLogWebSocket(): UseLogWebSocketReturn {
  const [connected, setConnected] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const getLogWebSocketUrl = useCallback((): string => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/logs`;
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const url = getLogWebSocketUrl();
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.timestamp && msg.level && msg.message) {
            setLogs((prev) => {
              const next = [...prev, msg as LogEntry];
              if (next.length > MAX_LOGS) {
                return next.slice(next.length - MAX_LOGS);
              }
              return next;
            });
          }
        } catch {
          // ignore
        }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, 5000);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      // ignore
    }
  }, [getLogWebSocketUrl]);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return { connected, logs, clearLogs };
}
