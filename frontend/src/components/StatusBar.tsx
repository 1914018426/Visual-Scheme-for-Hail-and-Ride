import { useState, useEffect } from 'react';
import { Wifi, WifiOff, Settings, Activity, Terminal } from 'lucide-react';
import { cn, formatTime } from '@/lib/utils';

interface StatusBarProps {
  connected: boolean;
  fps: number;
  lastError?: string;
  onSettingsClick: () => void;
  onLogsClick: () => void;
}

export function StatusBar({ connected, fps, lastError, onSettingsClick, onLogsClick }: StatusBarProps) {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="relative z-50 flex items-center justify-between px-4 py-3 bg-slate-900/80 backdrop-blur-md border-b border-slate-800/60">
      {/* Left: Logo & System Name */}
      <div className="flex items-center gap-3">
        <div className="relative flex items-center justify-center w-9 h-9 rounded-lg bg-gradient-to-br from-teal-500 to-teal-700 shadow-glow">
          <Activity className="w-5 h-5 text-white" />
          <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-teal-400 animate-pulse" />
        </div>
        <div className="flex flex-col">
          <h1 className="text-sm md:text-base font-bold text-slate-100 tracking-wide">
            Hailuo Vision
          </h1>
          <span className="text-[10px] md:text-xs text-slate-400 leading-none">
            视觉招手即停系统
          </span>
        </div>
      </div>

      {/* Center: Connection Status & FPS */}
      <div className="hidden sm:flex items-center gap-4">
        {/* Connection Status */}
        <div
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all duration-500',
            connected
              ? 'bg-teal-500/10 border-teal-500/30'
              : 'bg-red-500/10 border-red-500/30'
          )}
        >
          {connected ? (
            <Wifi className="w-3.5 h-3.5 text-teal-400" />
          ) : (
            <WifiOff className="w-3.5 h-3.5 text-red-400" />
          )}
          <span
            className={cn(
              'text-xs font-medium',
              connected ? 'text-teal-400' : 'text-red-400'
            )}
            title={!connected && lastError ? lastError : undefined}
          >
            {connected ? '已连接' : '未连接'}
          </span>
          <span
            className={cn(
              'w-2 h-2 rounded-full',
              connected ? 'bg-teal-400 animate-pulse' : 'bg-red-400'
            )}
          />
        </div>

        {/* FPS */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/60 border border-slate-700/50">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">FPS</span>
          <span className="text-sm font-mono font-semibold text-teal-400 tabular-nums">
            {fps}
          </span>
        </div>
      </div>

      {/* Right: Time & Settings */}
      <div className="flex items-center gap-3">
        {!connected && lastError ? (
          <div className="hidden lg:flex max-w-[260px] px-3 py-1.5 rounded-full bg-red-500/10 border border-red-500/30">
            <span className="text-[10px] text-red-300 truncate">{lastError}</span>
          </div>
        ) : null}

        {/* Time - hidden on small screens */}
        <div className="hidden md:flex items-center px-3 py-1.5 rounded-full bg-slate-800/60 border border-slate-700/50">
          <span className="text-sm font-mono text-slate-300 tabular-nums">
            {formatTime(currentTime)}
          </span>
        </div>

        {/* Mobile connection indicator */}
        <div
          className={cn(
            'sm:hidden w-3 h-3 rounded-full',
            connected ? 'bg-teal-400 animate-pulse' : 'bg-red-400'
          )}
        />

        {/* Logs Button */}
        <button
          onClick={onLogsClick}
          className={cn(
            'flex items-center justify-center w-9 h-9 rounded-lg',
            'bg-slate-800/60 border border-slate-700/50',
            'text-slate-400 hover:text-teal-400 hover:border-teal-500/30 hover:bg-teal-500/10',
            'transition-all duration-300 ease-out',
            'focus:outline-none focus:ring-2 focus:ring-teal-500/30'
          )}
          title="查看日志"
        >
          <Terminal className="w-4.5 h-4.5" />
        </button>

        {/* Settings Button */}
        <button
          onClick={onSettingsClick}
          className={cn(
            'flex items-center justify-center w-9 h-9 rounded-lg',
            'bg-slate-800/60 border border-slate-700/50',
            'text-slate-400 hover:text-teal-400 hover:border-teal-500/30 hover:bg-teal-500/10',
            'transition-all duration-300 ease-out',
            'focus:outline-none focus:ring-2 focus:ring-teal-500/30'
          )}
          title="设置"
        >
          <Settings className="w-4.5 h-4.5" />
        </button>
      </div>
    </header>
  );
}
