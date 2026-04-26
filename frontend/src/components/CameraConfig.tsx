import * as Dialog from '@radix-ui/react-dialog';
import * as Tabs from '@radix-ui/react-tabs';
import * as Switch from '@radix-ui/react-switch';
import { useRef, useState, type ChangeEvent } from 'react';
import {
  X,
  Save,
  RotateCcw,
  TestTube,
  Camera,
  FileUp,
  Wand2,
  Braces,
  Database,
  Trash2,
  AlertTriangle,
  Plus,
  Minus,
  Monitor,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  CAMERA_LABELS,
  PROTOCOL_LABELS,
  type PullMethod,
  type VehicleCameraProfile,
  type CameraProfileBundle,
  type CameraConfig as CameraConfigType,
  type DisplayConfig,
} from '@/types';

interface CameraConfigProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  activeTab: string;
  onTabChange: (tab: string) => void;
  configs: Record<string, CameraConfigType>;
  bundles: CameraProfileBundle[];
  selectedBundleId: string;
  selectedProfileId: string;
  pullMethod: PullMethod;
  selectedProfile: VehicleCameraProfile | null;
  jsonEditorText: string;
  displayConfig: DisplayConfig;
  onUpdateConfig: (id: string, updates: Partial<CameraConfigType>) => void;
  onUpdateSelectedProfileCameraName: (id: string, cameraName: string) => void;
  onBundleChange: (bundleId: string) => void;
  onProfileChange: (profileId: string) => void;
  onPullMethodChange: (method: PullMethod) => void;
  onJsonEditorChange: (text: string) => void;
  onImportConfigDocument: (jsonText: string) => { ok: boolean; message: string };
  onApplyJsonEditor: () => { ok: boolean; message: string };
  onApplySelectedProfile: () => void;
  onTestConnection: () => Promise<{ ok: boolean; message: string }>;
  onSave: () => Promise<{ ok: boolean; message: string }>;
  onReset: () => void;
  onDeleteBundle: () => { ok: boolean; message: string };
  onDeleteProfile: () => { ok: boolean; message: string };
  onClearAll: () => void;
  onAddCamera: (id: string, label?: string) => void;
  onRemoveCamera: (id: string) => void;
  onUpdateDisplayConfig: (updates: Partial<DisplayConfig>) => void;
}

const PROTOCOLS: Array<CameraConfigType['protocol']> = ['rtsp', 'rtmp', 'http', 'webrtc', 'local', 'file'];

export function CameraConfig({
  open,
  onOpenChange,
  activeTab,
  onTabChange,
  configs,
  bundles,
  selectedBundleId,
  selectedProfileId,
  pullMethod,
  selectedProfile,
  jsonEditorText,
  displayConfig,
  onUpdateConfig,
  onUpdateSelectedProfileCameraName,
  onBundleChange,
  onProfileChange,
  onPullMethodChange,
  onJsonEditorChange,
  onImportConfigDocument,
  onApplyJsonEditor,
  onApplySelectedProfile,
  onTestConnection,
  onSave,
  onReset,
  onDeleteBundle,
  onDeleteProfile,
  onClearAll,
  onAddCamera,
  onRemoveCamera,
  onUpdateDisplayConfig,
}: CameraConfigProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [importMessage, setImportMessage] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [newCameraId, setNewCameraId] = useState('');
  const [newCameraLabel, setNewCameraLabel] = useState('');
  const [showDisplayJson, setShowDisplayJson] = useState(false);
  const [displayJsonText, setDisplayJsonText] = useState('');

  const cameraTabs = Object.keys(configs);
  const selectedBundle =
    bundles.find((bundle) => bundle.id === selectedBundleId) ?? null;

  const handleImportFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const text = await file.text();
    const result = onImportConfigDocument(text);
    setImportMessage(result.message);
    event.target.value = '';
  };

  const handleApplyJsonEditor = () => {
    const result = onApplyJsonEditor();
    setImportMessage(result.message);
  };

  const handleAddCamera = () => {
    const id = newCameraId.trim();
    if (!id) return;
    if (!/^[a-zA-Z0-9_-]{1,32}$/.test(id)) {
      setImportMessage('摄像头ID只能包含字母、数字、下划线和连字符。');
      return;
    }
    if (configs[id]) {
      setImportMessage(`摄像头 "${id}" 已存在。`);
      return;
    }
    onAddCamera(id, newCameraLabel.trim() || undefined);
    setNewCameraId('');
    setNewCameraLabel('');
    setImportMessage(`摄像头 "${id}" 已添加。`);
    onTabChange(id);
  };

  const handleRemoveCamera = (id: string) => {
    onRemoveCamera(id);
    setImportMessage(`摄像头 "${id}" 已删除。`);
  };

  const handleApplyDisplayJson = () => {
    try {
      const parsed = JSON.parse(displayJsonText) as DisplayConfig;
      if (!Array.isArray(parsed.order)) {
        setImportMessage('显示配置 JSON 无效：order 必须是数组。');
        return;
      }
      onUpdateDisplayConfig({
        order: parsed.order,
        labels: parsed.labels || {},
      });
      setImportMessage('显示配置已应用。');
      setShowDisplayJson(false);
    } catch {
      setImportMessage('显示配置 JSON 解析失败。');
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay
          className={cn(
            'fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm',
            'data-[state=open]:animate-fade-in'
          )}
        />
        <Dialog.Content
          className={cn(
            'fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-[101]',
            'w-full max-w-xl mx-4',
            'max-h-[90vh] flex flex-col overflow-hidden',
            'bg-slate-900 border border-slate-700/60 rounded-2xl shadow-2xl',
            'data-[state=open]:animate-slide-up',
            'focus:outline-none'
          )}
        >
          {/* Header */}
          <div className="shrink-0 flex items-center justify-between px-5 py-4 border-b border-slate-800/60">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-teal-500/15 border border-teal-500/20">
                <Camera className="w-4 h-4 text-teal-400" />
              </div>
              <div>
                <Dialog.Title className="text-sm font-semibold text-slate-100">
                  摄像头配置
                </Dialog.Title>
                <Dialog.Description className="text-[11px] text-slate-500 mt-0.5">
                  配置视频流连接参数与显示设置
                </Dialog.Description>
              </div>
            </div>
            <Dialog.Close asChild>
              <button
                className={cn(
                  'flex items-center justify-center w-8 h-8 rounded-lg',
                  'text-slate-400 hover:text-slate-200 hover:bg-slate-800',
                  'transition-colors duration-200',
                  'focus:outline-none focus:ring-2 focus:ring-teal-500/30'
                )}
              >
                <X className="w-4 h-4" />
              </button>
            </Dialog.Close>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin">
            {/* Tabs */}
            <Tabs.Root
              value={activeTab}
              onValueChange={(v) => onTabChange(v)}
            >
              <Tabs.List
                className="sticky top-0 z-10 flex gap-1 px-5 pt-4 pb-2 bg-slate-900/95 backdrop-blur supports-[backdrop-filter]:bg-slate-900/80 flex-wrap"
                aria-label="选择摄像头"
              >
                {cameraTabs.map((camId) => (
                  <Tabs.Trigger
                    key={camId}
                    value={camId}
                    className={cn(
                      'flex items-center justify-center gap-2 px-3 py-2 rounded-lg',
                      'text-xs font-medium transition-all duration-200',
                      'border focus:outline-none focus:ring-2 focus:ring-teal-500/30',
                      'data-[state=active]:bg-teal-500/15 data-[state=active]:text-teal-400 data-[state=active]:border-teal-500/30',
                      'data-[state=inactive]:bg-slate-800/40 data-[state=inactive]:text-slate-400 data-[state=inactive]:border-slate-700/40',
                      'data-[state=inactive]:hover:bg-slate-800 data-[state=inactive]:hover:text-slate-300'
                    )}
                  >
                    <div
                      className={cn(
                        'w-1.5 h-1.5 rounded-full',
                        configs[camId]?.enabled ? 'bg-teal-400' : 'bg-slate-600'
                      )}
                    />
                    {configs[camId]?.label || CAMERA_LABELS[camId] || camId}
                  </Tabs.Trigger>
                ))}
                {/* Add Camera button in tab bar */}
                <div className="flex items-center gap-1 ml-1">
                  <input
                    type="text"
                    value={newCameraId}
                    onChange={(e) => setNewCameraId(e.target.value)}
                    placeholder="新ID"
                    className="w-20 px-2 py-1.5 rounded-lg text-xs bg-slate-800/60 border border-slate-700/50 text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-teal-500/30"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleAddCamera();
                    }}
                  />
                  <input
                    type="text"
                    value={newCameraLabel}
                    onChange={(e) => setNewCameraLabel(e.target.value)}
                    placeholder="名称"
                    className="w-24 px-2 py-1.5 rounded-lg text-xs bg-slate-800/60 border border-slate-700/50 text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-teal-500/30"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleAddCamera();
                    }}
                  />
                  <button
                    onClick={handleAddCamera}
                    className={cn(
                      'flex items-center justify-center w-7 h-7 rounded-lg',
                      'bg-teal-500/15 text-teal-400 border border-teal-500/30',
                      'hover:bg-teal-500/25 transition-colors'
                    )}
                    title="添加摄像头"
                  >
                    <Plus className="w-3.5 h-3.5" />
                  </button>
                </div>
              </Tabs.List>

              {/* Tab Content */}
              {cameraTabs.map((camId) => (
                <Tabs.Content
                  key={camId}
                  value={camId}
                  className="px-5 py-4 focus:outline-none"
                >
                  <ConfigForm
                    config={configs[camId]}
                    onUpdate={(updates) => onUpdateConfig(camId, updates)}
                  />
                  <div className="mt-4 flex items-center justify-end">
                    <button
                      onClick={() => handleRemoveCamera(camId)}
                      className={cn(
                        'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs',
                        'text-rose-400 bg-rose-500/10 border border-rose-500/20',
                        'hover:bg-rose-500/20 transition-colors'
                      )}
                    >
                      <Minus className="w-3 h-3" />
                      删除此摄像头
                    </button>
                  </div>
                </Tabs.Content>
              ))}
            </Tabs.Root>

            <div className="px-5 pb-4 space-y-4">
              <div className="h-px bg-slate-800/70" />

              {/* Display Settings */}
              <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Monitor className="w-4 h-4 text-teal-300" />
                    <h3 className="text-xs font-semibold text-slate-200">
                      显示设置
                    </h3>
                  </div>
                  <button
                    onClick={() => {
                      setDisplayJsonText(JSON.stringify(displayConfig, null, 2));
                      setShowDisplayJson((v) => !v);
                    }}
                    className="text-[10px] text-teal-400 hover:text-teal-300"
                  >
                    {showDisplayJson ? '收起 JSON' : '编辑 JSON'}
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  {displayConfig.order.map((camId) => (
                    <div key={camId} className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-500 w-5 text-right">
                        {displayConfig.order.indexOf(camId) + 1}
                      </span>
                      <input
                        type="text"
                        value={displayConfig.labels[camId] || camId}
                        onChange={(e) =>
                          onUpdateDisplayConfig({
                            labels: { [camId]: e.target.value },
                          })
                        }
                        className="flex-1 px-2 py-1 rounded-md text-xs bg-slate-800/60 border border-slate-700/50 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500/30"
                      />
                    </div>
                  ))}
                </div>

                {showDisplayJson && (
                  <div className="space-y-2">
                    <textarea
                      value={displayJsonText}
                      onChange={(e) => setDisplayJsonText(e.target.value)}
                      className="w-full min-h-[100px] rounded-lg border border-slate-700/60 bg-slate-950/80 px-3 py-2 text-[11px] text-slate-200 font-mono leading-5 focus:outline-none focus:ring-2 focus:ring-teal-500/30"
                    />
                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() =>
                          setDisplayJsonText(
                            JSON.stringify(displayConfig, null, 2)
                          )
                        }
                        className="px-2.5 py-1 rounded-md text-[10px] text-slate-400 bg-slate-800 border border-slate-700 hover:text-slate-200"
                      >
                        重置为当前
                      </button>
                      <button
                        onClick={handleApplyDisplayJson}
                        className="px-3 py-1.5 rounded-md text-xs border border-teal-500/40 bg-teal-500/20 text-teal-200 hover:bg-teal-500/30"
                      >
                        应用 JSON
                      </button>
                    </div>
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4 space-y-4">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-indigo-300" />
                  <h3 className="text-xs font-semibold text-slate-200">
                    配置集管理
                  </h3>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-[11px] text-slate-400">配置集</label>
                      <button
                        onClick={() => {
                          const result = onDeleteBundle();
                          setImportMessage(result.message);
                        }}
                        disabled={bundles.length <= 1}
                        className="text-[10px] text-rose-400 hover:text-rose-300 disabled:text-slate-600 disabled:cursor-not-allowed"
                        title="删除当前配置集"
                      >
                        删除
                      </button>
                    </div>
                    <select
                      value={selectedBundleId}
                      onChange={(e) => onBundleChange(e.target.value)}
                      className="w-full rounded-lg border border-slate-700/60 bg-slate-800/70 px-2.5 py-2 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/30"
                    >
                      {bundles.map((bundle) => (
                        <option key={bundle.id} value={bundle.id}>
                          {bundle.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-[11px] text-slate-400">场景配置</label>
                      <button
                        onClick={() => {
                          const result = onDeleteProfile();
                          setImportMessage(result.message);
                        }}
                        disabled={(selectedBundle?.profiles.length ?? 0) <= 1}
                        className="text-[10px] text-rose-400 hover:text-rose-300 disabled:text-slate-600 disabled:cursor-not-allowed"
                        title="删除当前场景"
                      >
                        删除
                      </button>
                    </div>
                    <select
                      value={selectedProfileId}
                      onChange={(e) => onProfileChange(e.target.value)}
                      className="w-full rounded-lg border border-slate-700/60 bg-slate-800/70 px-2.5 py-2 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/30"
                    >
                      {(selectedBundle?.profiles ?? []).map((profile) => (
                        <option key={profile.id} value={profile.id}>
                          {profile.name}（{profile.vehicleCount}车）
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[11px] text-slate-400">拉流方式</label>
                  <div className="flex gap-2">
                    {(['webrtc', 'rtmp'] as PullMethod[]).map((method) => (
                      <button
                        key={method}
                        onClick={() => onPullMethodChange(method)}
                        className={cn(
                          'px-3 py-1.5 rounded-md text-xs border transition-colors',
                          pullMethod === method
                            ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40'
                            : 'bg-slate-800/60 text-slate-400 border-slate-700/50 hover:text-slate-200'
                        )}
                      >
                        {method.toUpperCase()}
                      </button>
                    ))}
                    <button
                      onClick={onApplySelectedProfile}
                      className="ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs border border-indigo-500/40 bg-indigo-500/20 text-indigo-200 hover:bg-indigo-500/30"
                    >
                      <Wand2 className="w-3.5 h-3.5" />
                      应用到配置
                    </button>
                  </div>
                  <p className="text-[10px] text-slate-500">
                    默认 WebRTC 播放页：{selectedBundle?.webrtcPlayerBaseUrl ?? '-'}
                  </p>
                </div>

                {selectedProfile ? (
                  <div className="space-y-2">
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-300">
                      <Camera className="w-3.5 h-3.5" />
                      可视化自定义配置（摄像机名称）
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.keys(selectedProfile.cameras).map((camId) => (
                        <label key={camId} className="space-y-1">
                          <span className="text-[10px] text-slate-500">
                            {configs[camId]?.label || CAMERA_LABELS[camId] || camId}
                          </span>
                          <input
                            type="text"
                            value={
                              (selectedProfile.cameras as unknown as Record<string, string>)[camId] || ''
                            }
                            onChange={(e) =>
                              onUpdateSelectedProfileCameraName(
                                camId,
                                e.target.value
                              )
                            }
                            className="w-full rounded-md border border-slate-700/60 bg-slate-800/70 px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/30"
                          />
                        </label>
                      ))}
                    </div>
                  </div>
                ) : null}

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-300">
                      <Braces className="w-3.5 h-3.5" />
                      在线编辑 JSON 配置
                    </div>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] border border-slate-600/60 bg-slate-800/60 text-slate-300 hover:text-slate-100"
                    >
                      <FileUp className="w-3 h-3" />
                      导入 JSON
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".json,application/json"
                      onChange={handleImportFile}
                      className="hidden"
                    />
                  </div>
                  <textarea
                    value={jsonEditorText}
                    onChange={(e) => onJsonEditorChange(e.target.value)}
                    className="w-full min-h-[120px] rounded-lg border border-slate-700/60 bg-slate-950/80 px-3 py-2 text-[11px] text-slate-200 font-mono leading-5 focus:outline-none focus:ring-2 focus:ring-indigo-500/30"
                  />
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-500">
                      支持导入后切换、在线编辑并立即生效
                    </span>
                    <button
                      onClick={handleApplyJsonEditor}
                      className="px-3 py-1.5 rounded-md text-xs border border-indigo-500/40 bg-indigo-500/20 text-indigo-200 hover:bg-indigo-500/30"
                    >
                      应用 JSON
                    </button>
                  </div>
                  {importMessage ? (
                    <p className="text-[10px] text-slate-400">{importMessage}</p>
                  ) : null}
                </div>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="shrink-0 flex items-center justify-between px-5 py-4 border-t border-slate-800/60 bg-slate-900">
            <div className="flex items-center gap-2 relative">
              {showClearConfirm ? (
                <div className="absolute bottom-full left-0 mb-2 w-64 p-3 rounded-lg border border-rose-500/30 bg-slate-800 shadow-xl z-20">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                    <div className="space-y-2">
                      <p className="text-[11px] text-slate-200">
                        确定要清除所有配置吗？此操作将删除所有自定义配置集、场景和摄像头设置，且不可恢复。
                      </p>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => {
                            onClearAll();
                            setShowClearConfirm(false);
                            setImportMessage('所有配置已清除。');
                          }}
                          className="px-2.5 py-1 rounded-md text-[10px] bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30"
                        >
                          确认清除
                        </button>
                        <button
                          onClick={() => setShowClearConfirm(false)}
                          className="px-2.5 py-1 rounded-md text-[10px] bg-slate-700 text-slate-300 border border-slate-600 hover:bg-slate-600"
                        >
                          取消
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ) : null}
              <button
                onClick={() => setShowClearConfirm(true)}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg',
                  'text-xs font-medium text-rose-400',
                  'bg-rose-500/10 border border-rose-500/20',
                  'hover:bg-rose-500/20 hover:border-rose-500/30',
                  'transition-all duration-200',
                  'focus:outline-none focus:ring-2 focus:ring-rose-500/30'
                )}
              >
                <Trash2 className="w-3.5 h-3.5" />
                清除全部
              </button>
              <button
                onClick={onReset}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg',
                  'text-xs font-medium text-slate-400',
                  'bg-slate-800/40 border border-slate-700/40',
                  'hover:text-slate-200 hover:bg-slate-800 hover:border-slate-600/40',
                  'transition-all duration-200',
                  'focus:outline-none focus:ring-2 focus:ring-slate-500/30'
                )}
              >
                <RotateCcw className="w-3.5 h-3.5" />
                重置
              </button>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={async () => {
                  setIsTesting(true);
                  const result = await onTestConnection();
                  setImportMessage(result.message);
                  setIsTesting(false);
                }}
                disabled={isTesting || isSaving}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg',
                  'text-xs font-medium text-teal-400',
                  'bg-teal-500/10 border border-teal-500/20',
                  'hover:bg-teal-500/20 hover:border-teal-500/30',
                  'transition-all duration-200',
                  'focus:outline-none focus:ring-2 focus:ring-teal-500/30',
                  (isTesting || isSaving) && 'opacity-60 cursor-not-allowed'
                )}
              >
                <TestTube className="w-3.5 h-3.5" />
                {isTesting ? '测试中...' : '测试连接'}
              </button>

              <button
                onClick={async () => {
                  setIsSaving(true);
                  const result = await onSave();
                  setImportMessage(result.message);
                  setIsSaving(false);
                  if (result.ok) {
                    onOpenChange(false);
                  }
                }}
                disabled={isSaving || isTesting}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg',
                  'text-xs font-medium text-slate-900',
                  'bg-teal-400 border border-teal-400',
                  'hover:bg-teal-300 hover:border-teal-300',
                  'transition-all duration-200 shadow-glow',
                  'focus:outline-none focus:ring-2 focus:ring-teal-500/50',
                  (isSaving || isTesting) && 'opacity-60 cursor-not-allowed'
                )}
              >
                <Save className="w-3.5 h-3.5" />
                {isSaving ? '保存中...' : '保存'}
              </button>
            </div>
          </div>
          {importMessage ? (
            <div className="px-5 pb-3">
              <p className="text-[11px] text-slate-400">{importMessage}</p>
            </div>
          ) : null}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

// ========== ConfigForm sub-component ==========

interface ConfigFormProps {
  config: CameraConfigType;
  onUpdate: (updates: Partial<CameraConfigType>) => void;
}

function ConfigForm({ config, onUpdate }: ConfigFormProps) {
  return (
    <div className="space-y-4">
      {/* Label */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-slate-300">显示名称</label>
        <input
          type="text"
          value={config.label}
          onChange={(e) => onUpdate({ label: e.target.value })}
          className={cn(
            'w-full px-3 py-2 rounded-lg bg-slate-800/60 border border-slate-700/50',
            'text-xs text-slate-200 placeholder:text-slate-600',
            'focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500/30',
            'transition-all duration-200'
          )}
        />
      </div>

      {/* Enable Toggle */}
      <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/30 border border-slate-700/40">
        <div className="flex flex-col gap-0.5">
          <span className="text-xs font-medium text-slate-300">启用摄像头</span>
          <span className="text-[10px] text-slate-500">
            关闭后将停止该路视频流的接收和处理
          </span>
        </div>
        <Switch.Root
          checked={config.enabled}
          onCheckedChange={(checked) => onUpdate({ enabled: checked })}
          className={cn(
            'relative w-10 h-6 rounded-full transition-colors duration-200',
            'focus:outline-none focus:ring-2 focus:ring-teal-500/30',
            config.enabled ? 'bg-teal-500' : 'bg-slate-700'
          )}
        >
          <Switch.Thumb
            className={cn(
              'block w-4 h-4 rounded-full bg-white shadow-md transition-transform duration-200',
              config.enabled ? 'translate-x-5' : 'translate-x-1'
            )}
          />
        </Switch.Root>
      </div>

      {/* Protocol Select */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-slate-300">传输协议</label>
        <div className="grid grid-cols-3 gap-2">
          {PROTOCOLS.map((protocol) => (
            <button
              key={protocol}
              onClick={() => onUpdate({ protocol })}
              className={cn(
                'px-3 py-2 rounded-lg text-xs font-medium border transition-all duration-200',
                config.protocol === protocol
                  ? 'bg-teal-500/15 text-teal-400 border-teal-500/30'
                  : 'bg-slate-800/40 text-slate-400 border-slate-700/40 hover:bg-slate-800 hover:text-slate-300'
              )}
            >
              {PROTOCOL_LABELS[protocol]}
            </button>
          ))}
        </div>
      </div>

      {/* Source URL */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-slate-300">源地址</label>
        <input
          type="text"
          value={config.source}
          onChange={(e) => onUpdate({ source: e.target.value })}
          placeholder={getPlaceholder(config.protocol)}
          className={cn(
            'w-full px-3 py-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50',
            'text-xs text-slate-200 placeholder:text-slate-600',
            'focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500/30',
            'transition-all duration-200'
          )}
        />
        <p className="text-[10px] text-slate-600">
          {getProtocolHint(config.protocol)}
        </p>
      </div>

      {/* Info */}
      <div className="flex items-start gap-2 p-3 rounded-lg bg-slate-800/30 border border-slate-700/40">
        <Camera className="w-3.5 h-3.5 text-slate-500 mt-0.5 shrink-0" />
        <div className="flex flex-col gap-0.5">
          <span className="text-[11px] font-medium text-slate-400">
            {config.label}
          </span>
          <span className="text-[10px] text-slate-600">
            ID: {config.id} | 协议: {PROTOCOL_LABELS[config.protocol]} |
            状态: {config.enabled ? '已启用' : '已禁用'}
          </span>
        </div>
      </div>
    </div>
  );
}

function getPlaceholder(protocol: CameraConfigType['protocol']): string {
  switch (protocol) {
    case 'rtsp':
      return 'rtsp://192.168.1.100:554/stream';
    case 'rtmp':
      return 'rtmp://192.168.1.100/live/stream';
    case 'http':
      return 'http://192.168.1.100:8080/video';
    case 'webrtc':
      return 'ws://192.168.1.100:8080/webrtc';
    case 'local':
      return '/dev/video0 或 0';
    case 'file':
      return '/path/to/video.mp4';
  }
}

function getProtocolHint(protocol: CameraConfigType['protocol']): string {
  switch (protocol) {
    case 'rtsp':
      return 'RTSP 流地址，通常用于 IP 摄像头';
    case 'rtmp':
      return 'RTMP 推流地址，通常用于直播平台';
    case 'http':
      return 'HTTP 视频流地址，如 MJPEG 或 HLS';
    case 'webrtc':
      return 'WebRTC 信令服务器地址';
    case 'local':
      return '本地摄像头设备路径或索引号';
    case 'file':
      return '本地视频文件的绝对路径';
  }
}
