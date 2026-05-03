import { cn } from '@/lib/utils';
import { GESTURE_LABELS } from '@/types';
import type { DetectionResult, Gesture } from '@/types';

interface GestureOverlayProps {
  detection: DetectionResult;
}

function getGestureBadgeColor(gesture: Gesture): string {
  switch (gesture) {
    case 'hand_up':
      return 'bg-slate-500/80 text-white';
    case 'waving':
      return 'bg-red-500/80 text-white';
    default:
      return 'bg-slate-600/60 text-slate-300';
  }
}

export function GestureOverlay({ detection }: GestureOverlayProps) {
  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none z-10"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      {/* Gesture label badge */}
      {detection.best_gesture !== 'none' && (
        <g>
          <rect
            x="72"
            y="4"
            width="24"
            height="8"
            rx="2"
            className={cn('animate-fade-in', getGestureBadgeColor(detection.best_gesture))}
            style={{ fillOpacity: 0.9 }}
          />
          <text
            x="84"
            y="9.5"
            textAnchor="middle"
            fontSize="3.5"
            fontWeight="600"
            fill="white"
            style={{ pointerEvents: 'none' }}
          >
            {GESTURE_LABELS[detection.best_gesture]}
          </text>
        </g>
      )}
    </svg>
  );
}
