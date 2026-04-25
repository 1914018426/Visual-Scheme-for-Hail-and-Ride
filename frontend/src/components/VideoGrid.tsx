import { VideoPanel } from './VideoPanel';
import type { CameraId, DetectionResult } from '@/types';

interface VideoGridProps {
  frames: Record<CameraId, string>;
  detections: Record<CameraId, DetectionResult>;
}

const CAMERA_ORDER: CameraId[] = ['front', 'left', 'right', 'back'];

export function VideoGrid({ frames, detections }: VideoGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-3 md:gap-4 md:p-4">
      {CAMERA_ORDER.map((cameraId) => (
        <VideoPanel
          key={cameraId}
          cameraId={cameraId}
          frameImage={frames[cameraId]}
          detection={detections[cameraId]}
          isOnline={!!frames[cameraId]}
        />
      ))}
    </div>
  );
}
