import { useMemo } from 'react';
import {
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Pause,
  Car,
  Radio,
  Clock,
  Shield,
} from 'lucide-react';
import { cn, formatTime } from '@/lib/utils';
import { DIRECTION_LABELS, CAMERA_SHORT_LABELS } from '@/types';
import type { Direction, CameraId } from '@/types';

interface DirectionPanelProps {
  direction: Direction;
  confidence: number;
  timestamp: number;
  cameraStatuses: Record<CameraId, boolean>;
}

interface DirectionArrow {
  direction: Direction;
  icon: React.ReactNode;
  position: string;
  label: string;
}

export function DirectionPanel({
  direction,
  confidence,
  timestamp,
  cameraStatuses,
}: DirectionPanelProps) {
  const arrows: DirectionArrow[] = useMemo(
    () => [
      {
        direction: 'forward',
        icon: <ArrowUp className="w-6 h-6" />,
        position: 'top-2 left-1/2 -translate-x-1/2',
        label: '前',
      },
      {
        direction: 'backward',
        icon: <ArrowDown className="w-6 h-6" />,
        position: 'bottom-2 left-1/2 -translate-x-1/2',
        label: '后',
      },
      {
        direction: 'left',
        icon: <ArrowLeft className="w-6 h-6" />,
        position: 'left-2 top-1/2 -translate-y-1/2',
        label: '左',
      },
      {
        direction: 'right',
        icon: <ArrowRight className="w-6 h-6" />,
        position: 'right-2 top-1/2 -translate-y-1/2',
        label: '右',
      },
    ],
    []
  );

  const directionColor = useMemo(() => {
    switch (direction) {
      case 'forward':
        return 'text-teal-400 border-teal-500/50 shadow-glow';
      case 'backward':
        return 'text-amber-400 border-amber-500/50';
      case 'left':
        return 'text-blue-400 border-blue-500/50';
      case 'right':
        return 'text-purple-400 border-purple-500/50';
      default:
        return 'text-slate-500 border-slate-600/30';
    }
  }, [direction]);

  const lastDetectTime = timestamp
    ? formatTime(new Date(timestamp))
    : '--:--:--';

  const cameraOrder: CameraId[] = ['front', 'back', 'left', 'right'];

  return (
    <div className="w-full bg-slate-900/60 border-t border-slate-800/60">
      <div className="max-w-7xl mx-auto px-3 py-3 md:px-4 md:py-4">
        <div className="flex flex-col lg:flex-row items-center gap-4 lg:gap-8">
          {/* Direction Visualization */}
          <div className="flex items-center gap-4">
            {/* Circular car view */}
            <div className="relative w-28 h-28 md:w-32 md:h-32 shrink-0">
              {/* Outer ring */}
              <div className="absolute inset-0 rounded-full border-2 border-slate-700/60" />

              {/* Direction arrows */}
              {arrows.map((arrow) => {
                const isActive = direction === arrow.direction;
                return (
                  <div
                    key={arrow.direction}
                    className={cn(
                      'absolute flex items-center justify-center w-9 h-9 rounded-lg border transition-all duration-500',
                      arrow.position,
                      isActive
                        ? cn('bg-slate-800 shadow-glow', directionColor)
                        : 'bg-slate-800/40 border-slate-700/40 text-slate-600'
                    )}
                  >
                    {arrow.icon}
                    {isActive && (
                      <div className="absolute inset-0 rounded-lg animate-pulse-glow" />
                    )}
                  </div>
                );
              })}

              {/* Center car icon */}
              <div
                className={cn(
                  'absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2',
                  'flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-500',
                  direction === 'none'
                    ? 'bg-slate-800 border-slate-600 text-slate-500'
                    : 'bg-slate-800 border-slate-600 text-slate-300'
                )}
              >
                {direction === 'none' ? (
                  <Pause className="w-5 h-5" />
                ) : (
                  <Car className="w-5 h-5" />
                )}
              </div>
            </div>

            {/* Direction Info */}
            <div className="flex flex-col gap-1 min-w-[120px]">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                当前方向
              </span>
              <span
                className={cn(
                  'text-2xl md:text-3xl font-bold transition-colors duration-300',
                  direction === 'none' ? 'text-slate-500' : 'text-teal-400 text-glow'
                )}
              >
                {DIRECTION_LABELS[direction]}
              </span>

              {/* Confidence */}
              <div className="flex items-center gap-2 mt-1">
                <Shield className="w-3 h-3 text-slate-500" />
                <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden max-w-[80px]">
                  <div
                    className={cn(
                      'h-full rounded-full transition-all duration-500',
                      confidence >= 0.8
                        ? 'bg-teal-400'
                        : confidence >= 0.5
                        ? 'bg-amber-400'
                        : 'bg-slate-500'
                    )}
                    style={{ width: `${Math.round(confidence * 100)}%` }}
                  />
                </div>
                <span className="text-[10px] font-mono text-slate-500 tabular-nums">
                  {Math.round(confidence * 100)}%
                </span>
              </div>

              {/* Last detect time */}
              <div className="flex items-center gap-1.5 mt-0.5">
                <Clock className="w-3 h-3 text-slate-600" />
                <span className="text-[10px] text-slate-600">
                  最后检测: {lastDetectTime}
                </span>
              </div>
            </div>
          </div>

          {/* Divider - hidden on mobile */}
          <div className="hidden lg:block w-px h-20 bg-slate-800/60" />

          {/* Camera Status List */}
          <div className="flex items-center gap-3 flex-wrap justify-center lg:justify-start">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider mr-1">
              摄像头状态
            </span>
            {cameraOrder.map((camId) => {
              const isOnline = cameraStatuses[camId];
              return (
                <div
                  key={camId}
                  className={cn(
                    'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition-all duration-300',
                    isOnline
                      ? 'bg-slate-800/60 border-slate-700/50'
                      : 'bg-slate-850/40 border-slate-800/40'
                  )}
                >
                  <Radio
                    className={cn(
                      'w-3 h-3',
                      isOnline ? 'text-teal-400' : 'text-slate-600'
                    )}
                  />
                  <span
                    className={cn(
                      'text-[11px] font-medium',
                      isOnline ? 'text-slate-300' : 'text-slate-600'
                    )}
                  >
                    {CAMERA_SHORT_LABELS[camId]}
                  </span>
                  <span
                    className={cn(
                      'w-1.5 h-1.5 rounded-full',
                      isOnline
                        ? 'bg-teal-400 animate-pulse'
                        : 'bg-slate-600'
                    )}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
