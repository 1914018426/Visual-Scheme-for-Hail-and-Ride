import { TrendingUp } from 'lucide-react';

interface HighlightCardProps {
  title: string;
  value: string | number;
  unit?: string;
  description?: string;
  color?: 'green' | 'blue' | 'amber' | 'red';
}

export function HighlightCard({ title, value, unit, description, color = 'green' }: HighlightCardProps) {
  const colorMap = {
    green: 'from-emerald-500/20 to-teal-500/5 border-emerald-500/40 text-emerald-400',
    blue: 'from-sky-500/20 to-blue-500/5 border-sky-500/40 text-sky-400',
    amber: 'from-amber-500/20 to-yellow-500/5 border-amber-500/40 text-amber-400',
    red: 'from-rose-500/20 to-red-500/5 border-rose-500/40 text-rose-400',
  };

  return (
    <div className={`relative overflow-hidden rounded-xl border bg-gradient-to-br p-4 ${colorMap[color]}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium opacity-80 uppercase tracking-wider">{title}</p>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-2xl font-bold">{value}</span>
            {unit && <span className="text-sm opacity-70">{unit}</span>}
          </div>
          {description && (
            <p className="mt-1 text-xs opacity-70">{description}</p>
          )}
        </div>
        <TrendingUp className="h-5 w-5 opacity-60" />
      </div>
      <div className="absolute -bottom-2 -right-2 h-16 w-16 rounded-full bg-white/5 blur-xl" />
    </div>
  );
}
