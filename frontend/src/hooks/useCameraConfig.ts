import { useState, useCallback, useMemo, useEffect } from 'react';
import type {
  CameraId,
  CameraConfig,
  CameraConfigs,
  PullMethod,
  CameraNameMapping,
  CameraProfileBundle,
  CameraProfileDocument,
  VehicleCameraProfile,
} from '@/types';
import {
  DEFAULT_CAMERA_CONFIGS,
  DEFAULT_CAMERA_PROFILE_DOCUMENT,
} from '@/types';

export interface UseCameraConfigReturn {
  configs: CameraConfigs;
  bundles: CameraProfileBundle[];
  selectedBundleId: string;
  selectedProfileId: string;
  pullMethod: PullMethod;
  selectedBundle: CameraProfileBundle | null;
  selectedProfile: VehicleCameraProfile | null;
  jsonEditorText: string;
  isOpen: boolean;
  activeTab: CameraId;
  setIsOpen: (open: boolean) => void;
  setActiveTab: (tab: CameraId) => void;
  updateConfig: (id: CameraId, updates: Partial<CameraConfig>) => void;
  updateSelectedProfileCameraName: (id: CameraId, cameraName: string) => void;
  setSelectedBundleId: (bundleId: string) => void;
  setSelectedProfileId: (profileId: string) => void;
  setPullMethod: (method: PullMethod) => void;
  setJsonEditorText: (text: string) => void;
  importConfigDocument: (jsonText: string) => { ok: boolean; message: string };
  applyJsonEditorToSelectedBundle: () => { ok: boolean; message: string };
  applySelectedProfileToConfigs: () => void;
  testConnection: () => Promise<{ ok: boolean; message: string }>;
  saveConfigs: () => Promise<{ ok: boolean; message: string }>;
  resetConfigs: () => void;
}

const STORAGE_KEY = 'hailuo_camera_configs';
const STORAGE_MANAGER_KEY = 'hailuo_camera_profile_manager';
const EXTRA_CONFIG_PATH = '/extra-camera-configs.json';

function loadConfigs(): CameraConfigs {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as CameraConfigs;
      return { ...DEFAULT_CAMERA_CONFIGS, ...parsed };
    }
  } catch {
    // Ignore parse errors
  }
  return { ...DEFAULT_CAMERA_CONFIGS };
}

export function useCameraConfig(): UseCameraConfigReturn {
  const syncConfigsToBackend = useCallback(async (nextConfigs: CameraConfigs) => {
    const request = async (path: string, init?: RequestInit) => {
      const response = await fetch(path, {
        headers: {
          'Content-Type': 'application/json',
        },
        ...init,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`${response.status} ${text}`);
      }
      if (response.status === 204) {
        return null;
      }
      return response.json();
    };

    const listResult = (await request('/api/cameras')) as
      | { cameras?: Array<{ camera_id: string }> }
      | null;
    const existingIds = new Set(
      (listResult?.cameras ?? []).map((item) => item.camera_id)
    );

    const orderedIds: CameraId[] = ['front', 'back', 'left', 'right'];
    for (const cameraId of orderedIds) {
      if (existingIds.has(cameraId)) {
        try {
          await request(`/api/cameras/${cameraId}`, { method: 'DELETE' });
        } catch {
          // Ignore remove failures and continue trying apply
        }
      }
    }

    for (const cameraId of orderedIds) {
      const cfg = nextConfigs[cameraId];
      if (!cfg.enabled || !cfg.source.trim()) {
        continue;
      }
      await request('/api/cameras', {
        method: 'POST',
        body: JSON.stringify({
          camera_id: cameraId,
          source: cfg.source.trim(),
        }),
      });
    }
  }, []);

  const [configs, setConfigs] = useState<CameraConfigs>(loadConfigs);
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<CameraId>('front');
  const [bundles, setBundles] = useState<CameraProfileBundle[]>(
    DEFAULT_CAMERA_PROFILE_DOCUMENT.bundles
  );
  const [selectedBundleId, setSelectedBundleId] = useState(
    DEFAULT_CAMERA_PROFILE_DOCUMENT.bundles[0]?.id ?? ''
  );
  const [selectedProfileId, setSelectedProfileId] = useState(
    DEFAULT_CAMERA_PROFILE_DOCUMENT.bundles[0]?.profiles[0]?.id ?? ''
  );
  const [pullMethod, setPullMethod] = useState<PullMethod>(
    DEFAULT_CAMERA_PROFILE_DOCUMENT.bundles[0]?.defaultPullMethod ?? 'webrtc'
  );
  const [jsonEditorText, setJsonEditorText] = useState('');

  const selectedBundle = useMemo(
    () => bundles.find((item) => item.id === selectedBundleId) ?? null,
    [bundles, selectedBundleId]
  );

  const selectedProfile = useMemo(
    () =>
      selectedBundle?.profiles.find((item) => item.id === selectedProfileId) ??
      null,
    [selectedBundle, selectedProfileId]
  );

  const buildSourceByMethod = useCallback(
    (bundle: CameraProfileBundle, method: PullMethod, cameraName: string) => {
      if (method === 'rtmp') {
        return `${bundle.rtmpBaseUrl.replace(/\/$/, '')}/camera/${cameraName}`;
      }
      const endpoint = new URL(bundle.webrtcApiBaseUrl);
      endpoint.searchParams.set('app', 'live');
      endpoint.searchParams.set('stream', `camera/${cameraName}`);
      endpoint.searchParams.set('type', 'play');
      return endpoint.toString();
    },
    []
  );

  const applyProfileToConfigs = useCallback(
    (bundle: CameraProfileBundle, profile: VehicleCameraProfile, method: PullMethod) => {
      setConfigs({
        front: {
          ...configs.front,
          protocol: method,
          source: buildSourceByMethod(bundle, method, profile.cameras.front),
        },
        back: {
          ...configs.back,
          protocol: method,
          source: buildSourceByMethod(bundle, method, profile.cameras.back),
        },
        left: {
          ...configs.left,
          protocol: method,
          source: buildSourceByMethod(bundle, method, profile.cameras.left),
        },
        right: {
          ...configs.right,
          protocol: method,
          source: buildSourceByMethod(bundle, method, profile.cameras.right),
        },
      });
    },
    [buildSourceByMethod, configs]
  );

  const normalizeDocument = useCallback((data: unknown): CameraProfileDocument | null => {
    if (!data || typeof data !== 'object' || !('bundles' in data)) {
      return null;
    }
    const doc = data as CameraProfileDocument;
    if (!Array.isArray(doc.bundles) || doc.bundles.length === 0) {
      return null;
    }
    return doc;
  }, []);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_MANAGER_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as {
          bundles: CameraProfileBundle[];
          selectedBundleId: string;
          selectedProfileId: string;
          pullMethod: PullMethod;
        };
        if (Array.isArray(parsed.bundles) && parsed.bundles.length > 0) {
          setBundles(parsed.bundles);
          setSelectedBundleId(parsed.selectedBundleId || parsed.bundles[0].id);
          setSelectedProfileId(
            parsed.selectedProfileId || parsed.bundles[0].profiles[0]?.id || ''
          );
          setPullMethod(parsed.pullMethod || parsed.bundles[0].defaultPullMethod);
        }
      }
    } catch {
      // Ignore parse errors
    }
  }, []);

  useEffect(() => {
    const loadExtraConfig = async () => {
      try {
        const response = await fetch(EXTRA_CONFIG_PATH, { cache: 'no-cache' });
        if (!response.ok) {
          return;
        }
        const parsed = (await response.json()) as unknown;
        const doc = normalizeDocument(parsed);
        if (!doc) {
          return;
        }
        setBundles((prev) => {
          const nextMap = new Map(prev.map((item) => [item.id, item]));
          doc.bundles.forEach((bundle) => nextMap.set(bundle.id, bundle));
          return Array.from(nextMap.values());
        });
      } catch {
        // Ignore fetch errors
      }
    };
    void loadExtraConfig();
  }, [normalizeDocument]);

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_MANAGER_KEY,
        JSON.stringify({
          bundles,
          selectedBundleId,
          selectedProfileId,
          pullMethod,
        })
      );
    } catch {
      // Ignore storage errors
    }
  }, [bundles, selectedBundleId, selectedProfileId, pullMethod]);

  useEffect(() => {
    if (selectedBundle) {
      setJsonEditorText(JSON.stringify(selectedBundle, null, 2));
    }
  }, [selectedBundle]);

  const updateConfig = useCallback((id: CameraId, updates: Partial<CameraConfig>) => {
    setConfigs((prev) => ({
      ...prev,
      [id]: { ...prev[id], ...updates },
    }));
  }, []);

  const updateSelectedProfileCameraName = useCallback(
    (id: CameraId, cameraName: string) => {
      setBundles((prev) =>
        prev.map((bundle) => {
          if (bundle.id !== selectedBundleId) {
            return bundle;
          }
          return {
            ...bundle,
            profiles: bundle.profiles.map((profile) => {
              if (profile.id !== selectedProfileId) {
                return profile;
              }
              return {
                ...profile,
                cameras: { ...profile.cameras, [id]: cameraName } as CameraNameMapping,
              };
            }),
          };
        })
      );
    },
    [selectedBundleId, selectedProfileId]
  );

  const importConfigDocument = useCallback(
    (jsonText: string) => {
      try {
        const parsed = JSON.parse(jsonText) as unknown;
        const doc = normalizeDocument(parsed);
        if (!doc) {
          return { ok: false, message: 'JSON 结构无效，缺少 bundles 数组。' };
        }
        setBundles((prev) => {
          const nextMap = new Map(prev.map((item) => [item.id, item]));
          doc.bundles.forEach((bundle) => nextMap.set(bundle.id, bundle));
          return Array.from(nextMap.values());
        });
        const first = doc.bundles[0];
        setSelectedBundleId(first.id);
        setSelectedProfileId(first.profiles[0]?.id ?? '');
        setPullMethod(first.defaultPullMethod);
        return { ok: true, message: `已导入 ${doc.bundles.length} 套配置。` };
      } catch {
        return { ok: false, message: 'JSON 解析失败，请检查格式。' };
      }
    },
    [normalizeDocument]
  );

  const applyJsonEditorToSelectedBundle = useCallback(() => {
    try {
      const parsed = JSON.parse(jsonEditorText) as CameraProfileBundle;
      if (!parsed.id || !Array.isArray(parsed.profiles)) {
        return { ok: false, message: '配置集内容无效，请检查 id/profiles。' };
      }
      setBundles((prev) => prev.map((item) => (item.id === selectedBundleId ? parsed : item)));
      setSelectedProfileId(parsed.profiles[0]?.id ?? '');
      setPullMethod(parsed.defaultPullMethod);
      return { ok: true, message: '配置集已更新。' };
    } catch {
      return { ok: false, message: 'JSON 解析失败，未应用。' };
    }
  }, [jsonEditorText, selectedBundleId]);

  const applySelectedProfileToConfigs = useCallback(() => {
    if (!selectedBundle || !selectedProfile) {
      return;
    }
    applyProfileToConfigs(selectedBundle, selectedProfile, pullMethod);
  }, [applyProfileToConfigs, pullMethod, selectedBundle, selectedProfile]);

  const saveConfigs = useCallback(async () => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(configs));
      await syncConfigsToBackend(configs);
      return { ok: true, message: '配置已保存并下发后端。' };
    } catch {
      return { ok: false, message: '配置保存成功，但下发后端失败。' };
    }
  }, [configs, syncConfigsToBackend]);

  const testConnection = useCallback(async () => {
    const configured = (['front', 'back', 'left', 'right'] as CameraId[]).filter(
      (id) => configs[id].enabled && configs[id].source.trim()
    );
    if (configured.length === 0) {
      return { ok: false, message: '至少需要配置并启用一路视频流。' };
    }
    try {
      // 先将当前配置下发到后端，再进行真实拉流状态检查
      try {
        await syncConfigsToBackend(configs);
      } catch {
        return { ok: false, message: '配置下发后端失败，无法进行拉流测试。' };
      }

      const fetchCameraStatus = async () => {
        const response = await fetch('/api/cameras');
        if (!response.ok) {
          throw new Error(`获取摄像头状态失败（${response.status}）`);
        }
        return (await response.json()) as {
          cameras?: Array<{
            camera_id: string;
            status: string;
            fps?: number;
            frame_count?: number;
            last_error?: string;
          }>;
        };
      };

      const isOnline = (camera: {
        status: string;
        fps?: number;
        frame_count?: number;
      }) => {
        const status = (camera.status || '').toLowerCase();
        return (
          (camera.fps ?? 0) > 0 ||
          (camera.frame_count ?? 0) > 0 ||
          status === 'running' ||
          status === 'online' ||
          status === 'active'
        );
      };

      const timeoutMs = 15000;
      const intervalMs = 2000;
      const start = Date.now();
      let latest:
        | {
            cameras?: Array<{
              camera_id: string;
              status: string;
              fps?: number;
              frame_count?: number;
              last_error?: string;
            }>;
          }
        | undefined;

      while (Date.now() - start < timeoutMs) {
        latest = await fetchCameraStatus();
        const map = new Map((latest.cameras ?? []).map((item) => [item.camera_id, item]));
        const allOnline = configured.every((id) => {
          const camera = map.get(id);
          return camera ? isOnline(camera) : false;
        });
        if (allOnline) {
          return {
            ok: true,
            message: `拉流测试通过，${configured.length} 路摄像头已连通。`,
          };
        }
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
      }

      const cameraMap = new Map((latest?.cameras ?? []).map((item) => [item.camera_id, item]));
      const detail = configured
        .map((id) => {
          const item = cameraMap.get(id);
          if (!item) {
            return `${id}: 未注册`;
          }
          const err = item.last_error ? `(${item.last_error})` : '';
          return `${id}: ${item.status}${err}`;
        })
        .join('；');
      return {
        ok: false,
        message: `拉流测试未通过（15秒超时）。${detail}`,
      };
    } catch {
      return { ok: false, message: '拉流测试失败，请检查后端状态与流地址可达性。' };
    }
  }, [configs, syncConfigsToBackend]);

  const resetConfigs = useCallback(() => {
    setConfigs({ ...DEFAULT_CAMERA_CONFIGS });
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore storage errors
    }
  }, []);

  return {
    configs,
    bundles,
    selectedBundleId,
    selectedProfileId,
    pullMethod,
    selectedBundle,
    selectedProfile,
    jsonEditorText,
    isOpen,
    activeTab,
    setIsOpen,
    setActiveTab,
    updateConfig,
    updateSelectedProfileCameraName,
    setSelectedBundleId,
    setSelectedProfileId,
    setPullMethod,
    setJsonEditorText,
    importConfigDocument,
    applyJsonEditorToSelectedBundle,
    applySelectedProfileToConfigs,
    testConnection,
    saveConfigs,
    resetConfigs,
  };
}
