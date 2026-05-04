import React from 'react';
import { Database, Cpu, TrendingUp, Zap, Layers, ShieldCheck, BarChart3 } from 'lucide-react';

export default function SystemAuditHeader({ activeTab }) {
  const stats = {
    tft: [
      { icon: Database, label: "Input Dimensions", value: "52", sub: "Neural Features", color: "bg-blue-500/10 text-blue-400" },
      { icon: Cpu, label: "Neural Density", value: "1.82", sub: "Million Params", color: "bg-emerald-500/10 text-emerald-400" },
      { icon: TrendingUp, label: "Engine Precision", value: "94.8", sub: "% Confidence", color: "bg-purple-500/10 text-purple-400" },
      { icon: Zap, label: "Prediction Horizon", value: "14", sub: "Multi-Day", color: "bg-amber-500/10 text-amber-400" },
    ],
    lgbm: [
      { icon: Database, label: "Feature Set", value: "16", sub: "Iterative Features", color: "bg-blue-500/10 text-blue-400" },
      { icon: Cpu, label: "Architecture", value: "GBM", sub: "Gradient Trees", color: "bg-emerald-500/10 text-emerald-400" },
      { icon: TrendingUp, label: "Precision Score", value: "91.2", sub: "% Accuracy", color: "bg-purple-500/10 text-purple-400" },
      { icon: Zap, label: "Inference Latency", value: "12", sub: "ms / Point", color: "bg-amber-500/10 text-amber-400" },
    ],
    crop: [
      { icon: Database, label: "Biological Inputs", value: "07", sub: "Sensor Data", color: "bg-blue-500/10 text-blue-400" },
      { icon: Layers, label: "Class Coverage", value: "22", sub: "Crop Varieties", color: "bg-emerald-500/10 text-emerald-400" },
      { icon: TrendingUp, label: "Match Score", value: "96.4", sub: "% Confidence", color: "bg-purple-500/10 text-purple-400" },
      { icon: BarChart3, label: "Logic Type", value: "XGB", sub: "Classifier", color: "bg-amber-500/10 text-amber-400" },
    ],
    disease: [
      { icon: Database, label: "Input Matrix", value: "224x224", sub: "Image Tensor", color: "bg-blue-500/10 text-blue-400" },
      { icon: ShieldCheck, label: "Base Network", value: "MNv2", sub: "MobileNet", color: "bg-emerald-500/10 text-emerald-400" },
      { icon: TrendingUp, label: "Diagnosis Acc", value: "94.1", sub: "% Precision", color: "bg-purple-500/10 text-purple-400" },
      { icon: BarChart3, label: "Detection Classes", value: "38", sub: "Pathologies", color: "bg-amber-500/10 text-amber-400" },
    ]
  };

  const currentStats = stats[activeTab] || stats.tft;

  return (
    <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
      {currentStats.map((stat, i) => (
        <div key={i} className="p-6 rounded-[32px] bg-slate-900/40 border border-slate-800/60 backdrop-blur-xl flex items-center gap-5 transition-all hover:bg-slate-900/60 group">
          <div className={`w-12 h-12 rounded-2xl ${stat.color} flex items-center justify-center transition-transform group-hover:scale-110`}>
            <stat.icon size={22}/>
          </div>
          <div>
            <p className="text-[10px] font-black uppercase text-slate-500 tracking-widest">{stat.label}</p>
            <p className="text-xl font-black text-slate-100">{stat.value} <span className="text-[10px] font-normal text-slate-600 uppercase tracking-tighter ml-1">{stat.sub}</span></p>
          </div>
        </div>
      ))}
    </section>
  );
}
