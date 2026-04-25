import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import { GESTURE_LABELS } from '@/types';
import { SKELETON_CONNECTIONS } from '@/types';
import type { DetectionResult, Gesture } from '@/types';

interface GestureOverlayProps {
  detection: DetectionResult;
}

function getGestureBadgeColor(gesture: Gesture): string {
  switch (gesture) {
    case 'hand_up':
      return 'bg-amber-500/80 text-white';
    case 'wave':
      return 'bg-teal-500/80 text-white';
    default:
      return 'bg-slate-600/60 text-slate-300';
  }
}

export function GestureOverlay({ detection }: GestureOverlayProps) {
  const keypoints = useMemo(() => {
    if (!detection.poses || detection.poses.length === 0) return [];
    return detection.poses[0]?.keypoints || [];
  }, [detection.poses]);

  const hasKeypoints = keypoints.length > 0;

  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none z-10"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      {/* Skeleton connections */}
      {hasKeypoints &&
        SKELETON_CONNECTIONS.map(([start, end], i) => {
          const kp1 = keypoints[start];
          const kp2 = keypoints[end];
          if (!kp1 || !kp2 || kp1.score < 0.3 || kp2.score < 0.3) return null;

          return (
            <line
              key={`conn-${i}`}
              x1={kp1.x * 100}
              y1={kp1.y * 100}
              x2={kp2.x * 100}
              y2={kp2.y * 100}
              stroke="#3b82f6"
              strokeWidth="0.6"
              strokeLinecap="round"
              opacity={0.8}
            />
          );
        })}

      {/* Keypoint circles */}
      {hasKeypoints &&
        keypoints.map((kp, i) => {
          if (kp.score < 0.3) return null;

          const isWrist = i === 9 || i === 10; // left_wrist or right_wrist
          const radius = isWrist ? 1.5 : 1;
          const fill = isWrist ? '#ef4444' : '#22c55e';

          return (
            <g key={`kp-${i}`}>
              {/* Pulse ring for wrists */}
              {isWrist && (
                <circle
                  cx={kp.x * 100}
                  cy={kp.y * 100}
                  r={radius}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="0.4"
                  opacity={0.6}
                >
                  <animate
                    attributeName="r"
                    values={`${radius};${radius * 2.5};${radius}`}
                    dur="1.5s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="opacity"
                    values="0.6;0;0.6"
                    dur="1.5s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
              <circle
                cx={kp.x * 100}
                cy={kp.y * 100}
                r={radius}
                fill={fill}
                opacity={0.9}
              />
            </g>
          );
        })}

      {/* Gesture label badge */}
      {detection.gesture !== 'none' && (
        <g>
          <rect
            x="72"
            y="4"
            width="24"
            height="8"
            rx="2"
            className={cn('animate-fade-in', getGestureBadgeColor(detection.gesture))}
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
            {GESTURE_LABELS[detection.gesture]}
          </text>
        </g>
      )}
    </svg>
  );
}
