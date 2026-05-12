import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  RecordingSession,
  AblationExperiment,
  AnalysisReport,
  DataLabStatus,
  RecordingTriggerMode,
} from './types';

const API_BASE = '/api/datalab';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ------------------------------------------------------------------
// Recordings
// ------------------------------------------------------------------

export function useRecordings() {
  const [recordings, setRecordings] = useState<RecordingSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchJson<RecordingSession[]>(`${API_BASE}/recordings`);
      setRecordings(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { recordings, loading, error, refresh };
}

export function useRecordingControl() {
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);

  const startRecording = useCallback(
    async (cameraId: string, triggerMode: RecordingTriggerMode, saveVideo = true) => {
      setStarting(true);
      setLastError(null);
      try {
        const data = await fetchJson<RecordingSession>(`${API_BASE}/recordings/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ camera_id: cameraId, trigger_mode: triggerMode, save_video: saveVideo }),
        });
        return data;
      } catch (e: any) {
        const msg = e?.message || '启动录制失败';
        setLastError(msg);
        // eslint-disable-next-line no-alert
        alert(`录制启动失败: ${msg}`);
        throw e;
      } finally {
        setStarting(false);
      }
    },
    []
  );

  const stopRecording = useCallback(async (sessionId: string) => {
    setStopping(true);
    setLastError(null);
    try {
      const data = await fetchJson<RecordingSession>(`${API_BASE}/recordings/${sessionId}/stop`, {
        method: 'POST',
      });
      return data;
    } catch (e: any) {
      const msg = e?.message || '停止录制失败';
      setLastError(msg);
      // eslint-disable-next-line no-alert
      alert(`录制停止失败: ${msg}`);
      throw e;
    } finally {
      setStopping(false);
    }
  }, []);

  const labelRecording = useCallback(async (sessionId: string, label: string, notes = '') => {
    const data = await fetchJson<RecordingSession>(`${API_BASE}/recordings/${sessionId}/label`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, notes }),
    });
    return data;
  }, []);

  const deleteRecording = useCallback(async (sessionId: string) => {
    const res = await fetch(`${API_BASE}/recordings/${sessionId}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
  }, []);

  const importVideo = useCallback(async (videoPath: string, label: string, notes: string) => {
    const data = await fetchJson<RecordingSession>(`${API_BASE}/recordings/import-video`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_path: videoPath, label, notes }),
    });
    return data;
  }, []);

  const uploadVideo = useCallback(async (file: File, label: string, notes: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', label);
    formData.append('notes', notes);
    const res = await fetch(`${API_BASE}/recordings/upload-video`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
    return res.json() as Promise<RecordingSession>;
  }, []);

  return { startRecording, stopRecording, labelRecording, deleteRecording, importVideo, uploadVideo, starting, stopping, lastError };
}

// ------------------------------------------------------------------
// Experiments
// ------------------------------------------------------------------

export function useExperiments() {
  const [experiments, setExperiments] = useState<AblationExperiment[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchJson<AblationExperiment[]>(`${API_BASE}/experiments`);
      setExperiments(data);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { experiments, loading, refresh };
}

export function useExperimentControl() {
  const [starting, setStarting] = useState(false);

  const startExperiment = useCallback(async (
    recordingId: string,
    experimentType: string = 'engine_comparison',
    engineNames?: string[],
    thresholdRange?: number[],
  ) => {
    setStarting(true);
    try {
      const body: Record<string, any> = {
        recording_id: recordingId,
        experiment_type: experimentType,
      };
      if (engineNames) body.engine_names = engineNames;
      if (thresholdRange) body.threshold_range = thresholdRange;
      const data = await fetchJson<AblationExperiment>(`${API_BASE}/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      return data;
    } finally {
      setStarting(false);
    }
  }, []);

  const stopExperiment = useCallback(async (expId: string) => {
    const data = await fetchJson<AblationExperiment>(`${API_BASE}/experiments/${expId}/stop`, {
      method: 'POST',
    });
    return data;
  }, []);

  const deleteExperiment = useCallback(async (expId: string) => {
    const res = await fetch(`${API_BASE}/experiments/${expId}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
  }, []);

  const startFullSuite = useCallback(async (positiveRecordingIds: string[], negativeRecordingIds: string[]) => {
    setStarting(true);
    try {
      const data = await fetchJson<AblationExperiment>(`${API_BASE}/experiments/start-full-suite`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          positive_recording_ids: positiveRecordingIds,
          negative_recording_ids: negativeRecordingIds,
        }),
      });
      return data;
    } finally {
      setStarting(false);
    }
  }, []);

  const exportMarkdown = useCallback(async (expId: string) => {
    const res = await fetch(`${API_BASE}/experiments/${expId}/export/md`);
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report_${expId}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, []);

  const exportCharts = useCallback(async (expId: string) => {
    const res = await fetch(`${API_BASE}/experiments/${expId}/export/charts`);
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `charts_${expId}.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, []);

  const exportPdf = useCallback(async (expId: string) => {
    const res = await fetch(`${API_BASE}/experiments/${expId}/export/pdf`);
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report_${expId}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, []);

  return { startExperiment, stopExperiment, deleteExperiment, startFullSuite, exportMarkdown, exportCharts, exportPdf, starting };
}

// ------------------------------------------------------------------
// Analysis Report
// ------------------------------------------------------------------

export function useAnalysisReport(expId?: string) {
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!expId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchJson<AnalysisReport>(`${API_BASE}/experiments/${expId}/report`);
      setReport(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [expId]);

  useEffect(() => {
    load();
  }, [load]);

  return { report, loading, error, refresh: load };
}

// ------------------------------------------------------------------
// WebSocket
// ------------------------------------------------------------------

export function useDataLabWebSocket() {
  const [status, setStatus] = useState<DataLabStatus | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/datalab/ws/datalab`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setStatus(data as DataLabStatus);
      } catch {
        // ignore
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
    };
  }, []);

  return status;
}
