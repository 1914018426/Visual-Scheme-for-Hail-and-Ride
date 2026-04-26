import { Video, Users, Radio } from 'lucide-react';
import { cn } from '@/lib/utils';
import { GestureOverlay } from './GestureOverlay';
import { GESTURE_LABELS } from '@/types';
import type { DetectionResult, Gesture } from '@/types';

interface VideoPanelProps {
  cameraId: string;
  label: string;
  frameImage: string;
  detection: DetectionResult;
  isOnline: boolean;
}

function getGestureColor(gesture: Gesture): string {
  switch (gesture) {
    case 'hand_up':
      return 'text-slate-400 bg-slate-400/15 border-slate-400/30';
    case 'waving':
      return 'text-red-400 bg-red-400/15 border-red-400/30';
    default:
      return 'text-slate-500 bg-slate-500/10 border-slate-500/20';
  }
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'bg-teal-400';
  if (confidence >= 0.5) return 'bg-amber-400';
  return 'bg-slate-500';
}

export function VideoPanel({ label, frameImage, detection, isOnline }: VideoPanelProps) {
  const getBorderColor = (gesture: Gesture) => {
    switch (gesture) {
      case 'waving':
        return 'border-red-500/60 shadow-glow-red animate-border-pulse';
      case 'hand_up':
        return 'border-slate-500/60 shadow-glow-slate animate-border-pulse';
      default:
        return 'border-slate-800/80 hover:border-slate-700/80';
    }
  };

  return (
    <div
      className={cn(
        'relative rounded-xl overflow-hidden bg-slate-850 border-2 transition-all duration-500',
        'shadow-lg',
        getBorderColor(detection.gesture)
      )}
    >
      {/* Top Info Bar */}
      <div className="absolute top-0 left-0 right-0 z-20 px-3 py-2 bg-gradient-to-b from-black/70 to-transparent">
        <div className="flex items-center gap-2">
          <Video className="w-3.5 h-3.5 text-slate-300" />
          <span className="text-xs font-medium text-slate-200">
            {label}
          </span>
        </div>
        <div className="flex items-center gap-2 mt-1">
          {/* Online Status */}
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-black/40">
            <Radio
              className={cn(
                'w-3 h-3',
                isOnline ? 'text-teal-400' : 'text-red-400'
              )}
            />
            <span
              className={cn(
                'text-[10px] font-medium',
                isOnline ? 'text-teal-400' : 'text-red-400'
              )}
            >
              {isOnline ? '在线' : '离线'}
            </span>
          </div>
          {/* Person Count */}
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-black/40">
            <Users className="w-3 h-3 text-slate-400" />
            <span className="text-[10px] font-medium text-slate-300">
              {detection.person_count}人
            </span>
          </div>
        </div>
      </div>

      {/* Video Frame */}
      <div className="relative aspect-video bg-slate-950 overflow-hidden">
        {frameImage ? (
          <img
            src={`data:image/jpeg;base64,${frameImage}`}
            alt={label}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="flex flex-col items-center justify-center w-full h-full gap-3">
            <div className="relative">
              <Video className="w-10 h-10 text-slate-700" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-2 h-2 rounded-full bg-red-400/50" />
              </div>
            </div>
            <span className="text-xs text-slate-600">等待视频流...</span>
          </div>
        )}

        {/* Gesture Overlay */}
        {frameImage && (
          <GestureOverlay detection={detection} />
        )}
      </div>

      {/* Bottom Info Bar */}
      <div className="flex items-center justify-between px-3 py-2 bg-slate-900/95 border-t border-slate-800/60">
        {/* Gesture Label */}
        <div
          className={cn(
            'flex items-center gap-2 px-2.5 py-1 rounded-full border transition-all duration-300',
            getGestureColor(detection.gesture)
          )}
        >
          <span className="text-[11px] font-medium">
            {GESTURE_LABELS[detection.gesture]}
          </span>
        </div>

        {/* Confidence Bar */}
        <div className="flex items-center gap-2 flex-1 ml-3 max-w-[120px]">
          <span className="text-[10px] text-slate-500 shrink-0">置信度</span>
          <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <div
              className={cn(
                'h-full rounded-full transition-all duration-500 ease-out',
                getConfidenceColor(detection.gesture_confidence)
              )}
              style={{
                width: `${Math.round(detection.gesture_confidence * 100)}%`,
              }}
            />
          </div>
          <span className="text-[10px] font-mono text-slate-500 w-8 text-right tabular-nums">
            {Math.round(detection.gesture_confidence * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
}
