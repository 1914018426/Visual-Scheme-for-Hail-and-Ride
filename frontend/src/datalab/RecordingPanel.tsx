import { useState } from 'react';
import {
  Circle,
  Square,
  Video,
  Trash2,
  RefreshCw,
  Clock,
  Film,
  Upload,
  FileVideo,
  Loader2,
} from 'lucide-react';
import { useRecordings, useRecordingControl, useDataLabWebSocket } from './useDataLab';
import type { RecordingSession, RecordingTriggerMode } from './types';

interface RecordingPanelProps {
  cameraId?: string;
}

const TRIGGER_LABELS: Record<RecordingTriggerMode, string> = {
  manual: '手动',
  auto_gesture: '自动检测',
  auto_continuous: '连续',
};

const LABEL_BADGES: Record<string, { text: string; className: string }> = {
  positive: { text: '正样本', className: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' },
  negative: { text: '负样本', className: 'bg-rose-500/20 text-rose-400 border-rose-500/30' },
  unlabeled: { text: '未标注', className: 'bg-slate-500/20 text-slate-400 border-slate-500/30' },
};

export function RecordingPanel({ cameraId = 'front' }: RecordingPanelProps) {
  const { recordings, loading, error, refresh } = useRecordings();
  const { startRecording, stopRecording, labelRecording, deleteRecording, uploadVideo, starting, stopping, lastError } =
    useRecordingControl();
  const wsStatus = useDataLabWebSocket();

  const [triggerMode, setTriggerMode] = useState<RecordingTriggerMode>('manual');
  const [saveVideo, setSaveVideo] = useState(true);

  // 视频导入状态
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [importLabel, setImportLabel] = useState('unlabeled');
  const [importNotes, setImportNotes] = useState('');
  const [importLoading, setImportLoading] = useState(false);

  const recorderStatus = wsStatus?.recording;
  const isRecording = recorderStatus?.is_recording ?? false;
  const activeSessions = recorderStatus?.sessions ?? [];

  // 当前选中的相机是否已在录制中
  const selectedCameraRecording = activeSessions.find((s) => s.camera_id === cameraId);

  const handleStart = async () => {
    await startRecording(cameraId, triggerMode, saveVideo);
    refresh();
  };

  const handleStop = async (sessionId: string) => {
    await stopRecording(sessionId);
    refresh();
  };

  const handleLabel = async (session: RecordingSession, label: string) => {
    await labelRecording(session.id, label);
    refresh();
  };

  const handleDelete = async (session: RecordingSession) => {
    if (!confirm(`确定删除录制 ${session.id}？`)) return;
    await deleteRecording(session.id);
    refresh();
  };

  const handleImportVideo = async () => {
    if (!selectedFile) return;
    setImportLoading(true);
    try {
      await uploadVideo(selectedFile, importLabel, importNotes);
      setSelectedFile(null);
      setImportNotes('');
      setImportLabel('unlabeled');
      refresh();
    } catch (e: any) {
      alert('导入失败: ' + (e.message || '未知错误'));
    } finally {
      setImportLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* 控制栏 */}
      <div className="flex flex-wrap items-center gap-4 rounded-xl border border-slate-800 bg-slate-900/50 p-4">
        <div className="flex items-center gap-2">
          <Video className="h-5 w-5 text-slate-400" />
          <span className="text-sm font-medium text-slate-300">录制控制</span>
        </div>

        <select
          value={triggerMode}
          onChange={(e) => setTriggerMode(e.target.value as RecordingTriggerMode)}
          disabled={!!selectedCameraRecording}
          className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200 outline-none focus:border-sky-500"
        >
          <option value="manual">手动触发</option>
          <option value="auto_gesture">自动检测 waving</option>
          <option value="auto_continuous">连续录制</option>
        </select>

        <label className="flex items-center gap-2 text-sm text-slate-400">
          <input
            type="checkbox"
            checked={saveVideo}
            onChange={(e) => setSaveVideo(e.target.checked)}
            disabled={!!selectedCameraRecording}
            className="h-4 w-4 rounded border-slate-600 bg-slate-700"
          />
          保存视频
        </label>

        {!selectedCameraRecording ? (
          <button
            onClick={handleStart}
            disabled={starting}
            className="flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-500 disabled:opacity-50"
          >
            <Circle className="h-4 w-4 fill-current" />
            {starting ? '启动中...' : '开始录制'}
          </button>
        ) : (
          <button
            onClick={() => handleStop(selectedCameraRecording.session_id)}
            disabled={stopping}
            className="flex items-center gap-2 rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white hover:bg-slate-600 disabled:opacity-50"
          >
            <Square className="h-4 w-4 fill-current" />
            {stopping ? '停止中...' : '停止录制'}
          </button>
        )}

        <button
          onClick={refresh}
          disabled={loading}
          className="ml-auto flex items-center gap-1 rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-400 hover:bg-slate-800"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          刷新
        </button>
      </div>

      {/* 视频导入栏 */}
      <div className="flex flex-wrap items-start gap-4 rounded-xl border border-slate-800 bg-slate-900/50 p-4">
        <div className="flex items-center gap-2">
          <FileVideo className="h-5 w-5 text-slate-400" />
          <span className="text-sm font-medium text-slate-300">视频导入</span>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-slate-500">选择视频文件</label>
          <input
            id="video-upload-input"
            type="file"
            accept="video/*"
            onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
            disabled={importLoading}
            className="hidden"
          />
          <div className="flex items-center gap-2">
            <button
              onClick={() => document.getElementById('video-upload-input')?.click()}
              disabled={importLoading}
              className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-700"
            >
              <Upload className="h-4 w-4" />
              浏览...
            </button>
            <span className="max-w-[240px] truncate text-sm text-slate-400">
              {selectedFile?.name || '未选择文件'}
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-slate-500">标签</label>
          <select
            value={importLabel}
            onChange={(e) => setImportLabel(e.target.value)}
            disabled={importLoading}
            className="min-w-[120px] rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200 outline-none focus:border-sky-500"
          >
            <option value="unlabeled">未标注</option>
            <option value="positive">正样本（招手）</option>
            <option value="negative">负样本（非招手）</option>
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-slate-500">备注</label>
          <input
            type="text"
            value={importNotes}
            onChange={(e) => setImportNotes(e.target.value)}
            disabled={importLoading}
            placeholder="可选备注"
            className="min-w-[180px] rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200 outline-none focus:border-sky-500"
          />
        </div>

        <button
          onClick={handleImportVideo}
          disabled={!selectedFile || importLoading}
          className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
        >
          {importLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          {importLoading ? '导入中...' : '导入视频'}
        </button>
      </div>

      {/* 活跃录制列表（多摄像头） */}
      {isRecording && activeSessions.length > 0 && (
        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-red-900/30 bg-red-950/20 p-4">
          <span className="text-sm font-medium text-red-300">正在录制:</span>
          {activeSessions.map((s) => (
            <div
              key={s.session_id}
              className="flex items-center gap-2 rounded-lg border border-red-800/40 bg-red-900/30 px-3 py-1.5"
            >
              <span className="relative flex h-2.5 w-2.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-red-500" />
              </span>
              <span className="text-xs font-medium text-red-200">
                {s.camera_id}
              </span>
              <span className="text-xs text-red-300/70">
                {s.frame_count} 帧 / {s.duration_s.toFixed(1)}s
              </span>
              {s.camera_id !== cameraId && (
                <button
                  onClick={() => handleStop(s.session_id)}
                  className="ml-1 rounded px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-800/40"
                  title="停止该录制"
                >
                  停止
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {(error || lastError) && (
        <div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-3 text-sm text-red-300">
          错误: {error || lastError}
        </div>
      )}

      {/* 录制列表 */}
      <div className="overflow-hidden rounded-xl border border-slate-800">
        <table className="w-full text-sm">
          <thead className="bg-slate-900/80 text-left text-slate-400">
            <tr>
              <th className="px-4 py-3 font-medium">ID</th>
              <th className="px-4 py-3 font-medium">时间</th>
              <th className="px-4 py-3 font-medium">相机</th>
              <th className="px-4 py-3 font-medium">模式</th>
              <th className="px-4 py-3 font-medium">时长</th>
              <th className="px-4 py-3 font-medium">帧数</th>
              <th className="px-4 py-3 font-medium">标签</th>
              <th className="px-4 py-3 font-medium text-right">操作</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {recordings.length === 0 && (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                  暂无录制记录
                </td>
              </tr>
            )}
            {recordings.map((r) => (
              <tr key={r.id} className="hover:bg-slate-800/50">
                <td className="px-4 py-3 font-mono text-slate-300">{r.id}</td>
                <td className="px-4 py-3 text-slate-400">
                  {new Date(r.start_time * 1000).toLocaleString()}
                </td>
                <td className="px-4 py-3 text-slate-400">{r.camera_id}</td>
                <td className="px-4 py-3 text-slate-400">
                  {TRIGGER_LABELS[r.trigger_mode as RecordingTriggerMode] ?? r.trigger_mode}
                </td>
                <td className="px-4 py-3 text-slate-400">
                  <span className="flex items-center gap-1">
                    <Clock className="h-3.5 w-3.5" />
                    {r.duration_s ? `${r.duration_s.toFixed(1)}s` : '-'}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-400">
                  <span className="flex items-center gap-1">
                    <Film className="h-3.5 w-3.5" />
                    {r.frame_count}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <select
                    value={r.manual_label}
                    onChange={(e) => handleLabel(r, e.target.value)}
                    className={`rounded border px-2 py-0.5 text-xs outline-none ${
                      LABEL_BADGES[r.manual_label]?.className ?? LABEL_BADGES.unlabeled.className
                    }`}
                  >
                    <option value="unlabeled">未标注</option>
                    <option value="positive">正样本</option>
                    <option value="negative">负样本</option>
                  </select>
                </td>
                <td className="px-4 py-3 text-right">
                  <button
                    onClick={() => handleDelete(r)}
                    className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-rose-400 hover:bg-rose-500/10"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    删除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
