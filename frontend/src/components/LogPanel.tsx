import { useRef, useEffect, useMemo } from 'react';
import { X, Trash2, Terminal } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { LogEntry } from '@/hooks/useLogWebSocket';

interface LogPanelProps {
  open: boolean;
  onClose: () => void;
  logs: LogEntry[];
  connected: boolean;
  onClear: () => void;
}

function getLevelColor(level: string): string {
  switch (level) {
    case 'ERROR':
      return 'text-red-400';
    case 'WARNING':
    case 'WARN':
      return 'text-amber-400';
    case 'INFO':
      return 'text-teal-400';
    case 'DEBUG':
      return 'text-slate-400';
    default:
      return 'text-slate-300';
  }
}

export function LogPanel({ open, onClose, logs, connected, onClear }: LogPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs.length]);

  const logLines = useMemo(() => {
    return logs.map((log, i) => {
      const date = new Date(log.timestamp * 1000);
      const timeStr = date.toLocaleTimeString('zh-CN', { hour12: false });
      return (
        <div key={i} className="font-mono text-[11px] leading-relaxed whitespace-pre-wrap break-all">
          <span className="text-slate-500">{timeStr}</span>{' '}
          <span className={cn('font-bold', getLevelColor(log.level))}>[{log.level}]</span>{' '}
          <span className="text-slate-400">{log.logger}:</span>{' '}
          <span className="text-slate-200">{log.message}</span>
        </div>
      );
    });
  }, [logs]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center sm:items-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div className="relative w-full max-w-4xl h-[70vh] sm:h-[600px] m-4 bg-slate-900 border border-slate-700/50 rounded-xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-slate-800/80 border-b border-slate-700/50">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-teal-400" />
            <span className="text-sm font-semibold text-slate-200">后端日志</span>
            <span
              className={cn(
                'w-2 h-2 rounded-full',
                connected ? 'bg-teal-400 animate-pulse' : 'bg-red-400'
              )}
            />
            <span className="text-[10px] text-slate-500">
              {logs.length} 条
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onClear}
              className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-colors"
              title="清空日志"
            >
              <Trash2 className="w-3 h-3" />
              清空
            </button>
            <button
              onClick={onClose}
              className="flex items-center justify-center w-7 h-7 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Log Content */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-3 space-y-0.5 scrollbar-thin bg-slate-950"
        >
          {logs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-600">
              <Terminal className="w-8 h-8 mb-2 opacity-50" />
              <span className="text-xs">等待日志...</span>
            </div>
          ) : (
            logLines
          )}
        </div>
      </div>
    </div>
  );
}
