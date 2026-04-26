import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useCameraConfig } from '@/hooks/useCameraConfig';
import { StatusBar } from '@/components/StatusBar';
import { VideoGrid } from '@/components/VideoGrid';
import { DirectionPanel } from '@/components/DirectionPanel';
import { CameraConfig } from '@/components/CameraConfig';

function App() {
  const {
    connected,
    lastError,
    frames,
    detections,
    direction,
    directionConfidence,
    directionTimestamp,
    fps,
  } = useWebSocket();

  const {
    configs,
    bundles,
    selectedBundleId,
    selectedProfileId,
    pullMethod,
    selectedProfile,
    jsonEditorText,
    isOpen,
    activeTab,
    displayConfig,
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
    deleteBundle,
    deleteProfile,
    clearAllConfigs,
    addCamera,
    removeCamera,
    updateDisplayConfig,
    moveCameraOrder,
  } = useCameraConfig();

  // Derive camera online status from frames
  const cameraStatuses = useMemo<Record<string, boolean>>(
    () => {
      const result: Record<string, boolean> = {};
      for (const key of Object.keys(frames)) {
        result[key] = !!frames[key];
      }
      return result;
    },
    [frames]
  );

  return (
    <div
      className={cn(
        'min-h-screen flex flex-col bg-slate-950 text-slate-100',
        'bg-grid-pattern'
      )}
    >
      {/* Status Bar */}
      <StatusBar
        connected={connected}
        fps={fps}
        lastError={lastError}
        onSettingsClick={() => setIsOpen(true)}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Video Grid */}
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          <VideoGrid
            frames={frames}
            detections={detections}
            displayConfig={displayConfig}
            onReorder={moveCameraOrder}
          />
        </div>

        {/* Direction Panel */}
        <DirectionPanel
          direction={direction}
          confidence={directionConfidence}
          timestamp={directionTimestamp}
          cameraStatuses={cameraStatuses}
          displayLabels={displayConfig.labels}
        />
      </main>

      {/* Camera Config Dialog */}
      <CameraConfig
        open={isOpen}
        onOpenChange={setIsOpen}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        configs={configs}
        bundles={bundles}
        selectedBundleId={selectedBundleId}
        selectedProfileId={selectedProfileId}
        pullMethod={pullMethod}
        selectedProfile={selectedProfile}
        jsonEditorText={jsonEditorText}
        displayConfig={displayConfig}
        onUpdateConfig={updateConfig}
        onUpdateSelectedProfileCameraName={updateSelectedProfileCameraName}
        onBundleChange={setSelectedBundleId}
        onProfileChange={setSelectedProfileId}
        onPullMethodChange={setPullMethod}
        onJsonEditorChange={setJsonEditorText}
        onImportConfigDocument={importConfigDocument}
        onApplyJsonEditor={applyJsonEditorToSelectedBundle}
        onApplySelectedProfile={applySelectedProfileToConfigs}
        onTestConnection={testConnection}
        onSave={saveConfigs}
        onReset={resetConfigs}
        onDeleteBundle={deleteBundle}
        onDeleteProfile={deleteProfile}
        onClearAll={clearAllConfigs}
        onAddCamera={addCamera}
        onRemoveCamera={removeCamera}
        onUpdateDisplayConfig={updateDisplayConfig}
      />
    </div>
  );
}

export default App;
