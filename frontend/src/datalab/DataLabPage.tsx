import { useState } from 'react';
import { Beaker, Video, BarChart3 } from 'lucide-react';
import { RecordingPanel } from './RecordingPanel';
import { AblationPanel } from './AblationPanel';
import { AnalysisDashboard } from './AnalysisDashboard';

interface DataLabPageProps {
  cameraId?: string;
}

type TabKey = 'recording' | 'ablation' | 'analysis';

export function DataLabPage({ cameraId = 'front' }: DataLabPageProps) {
  const [activeTab, setActiveTab] = useState<TabKey>('recording');
  const [selectedExperimentId, setSelectedExperimentId] = useState<string | undefined>();

  const handleSelectExperiment = (expId: string) => {
    setSelectedExperimentId(expId);
    setActiveTab('analysis');
  };

  const tabs: { key: TabKey; label: string; icon: React.ReactNode }[] = [
    { key: 'recording', label: '素材录制', icon: <Video className="h-4 w-4" /> },
    { key: 'ablation', label: '消融实验', icon: <Beaker className="h-4 w-4" /> },
    { key: 'analysis', label: '结果分析', icon: <BarChart3 className="h-4 w-4" /> },
  ];

  return (
    <div className="flex h-full flex-col overflow-hidden bg-slate-950 text-slate-100">
      {/* Header */}
      <div className="flex items-center gap-4 border-b border-slate-800 bg-slate-900/50 px-6 py-4">
        <div className="flex items-center gap-2">
          <Beaker className="h-5 w-5 text-sky-400" />
          <h1 className="text-lg font-bold text-slate-100">数据实验室</h1>
        </div>
        <p className="text-xs text-slate-500">
          招手素材录制 · 消融实验 · 引擎有效性分析
        </p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800 bg-slate-900/30 px-6">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? 'border-sky-500 text-sky-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {activeTab === 'recording' && <RecordingPanel cameraId={cameraId} />}
        {activeTab === 'ablation' && (
          <AblationPanel onSelectExperiment={handleSelectExperiment} />
        )}
        {activeTab === 'analysis' && (
          <AnalysisDashboard
            experimentId={selectedExperimentId}
            onRequestNew={() => setActiveTab('ablation')}
          />
        )}
      </div>
    </div>
  );
}
