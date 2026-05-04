import React from 'react';
import { Line } from 'react-chartjs-2';
import { MapPin, Info, TrendingDown, TrendingUp as TrendingUpIcon, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export default function PriceForecastView({ data, activeTab, selection, states, districts, mandis, commodities, setSelection, setDistricts, setMandis, setCommodities, hierarchy, loading, runAnalysis, API_BASE, axios, handleStateChange, handleDistrictChange, handleMandiChange }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-10 items-start">
      {/* INPUT PANEL */}
      <section className="p-8 rounded-[40px] bg-slate-900/40 border border-slate-800/60 backdrop-blur-3xl shadow-2xl">
        <h3 className="text-xl font-bold mb-8 flex items-center justify-between">
          <span className="flex items-center gap-3"><MapPin size={20} className="text-blue-500"/> Market Context</span>
          <Info size={16} className="text-slate-700 cursor-help" />
        </h3>

        <div className="space-y-6">
          {['State', 'District', 'Mandi', 'Commodity'].map((f, i) => (
            <div key={i} className="space-y-2">
              <label className="text-[10px] font-black uppercase tracking-widest text-slate-500">{f}</label>
              <select className="w-full bg-slate-950/50 border border-slate-800 rounded-2xl px-5 py-4 text-sm focus:border-blue-500/50 outline-none transition-colors" value={selection[f.toLowerCase()]}
                onChange={(e) => {
                  const v = e.target.value;
                  if(f === 'State') handleStateChange(v);
                  else if(f === 'District') handleDistrictChange(v);
                  else if(f === 'Mandi') handleMandiChange(v);
                  else setSelection(p => ({...p, commodity: v}));
                }}
              >
                <option value="">Select {f}...</option>
                {(f === 'State' ? states : f === 'District' ? districts : f === 'Mandi' ? mandis : commodities).map((opt, j) => <option key={j} value={opt}>{opt}</option>)}
              </select>
            </div>
          ))}
        </div>

        <button onClick={runAnalysis} disabled={loading || !selection.commodity} className="w-full py-6 rounded-2xl bg-gradient-to-r from-blue-600 to-indigo-600 font-black tracking-widest uppercase text-xs hover:scale-[1.02] active:scale-95 transition-all mt-10 shadow-xl shadow-blue-500/20 disabled:opacity-50 disabled:grayscale">
          {loading ? "Computing Neural Path..." : "Initiate Trajectory"}
        </button>
      </section>

      {/* RESULTS PANEL */}
      <section className="p-10 rounded-[40px] bg-slate-900/40 border border-slate-800/60 backdrop-blur-3xl shadow-2xl min-h-[600px] flex flex-col">
        {data ? (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-10 flex-1 flex flex-col">
            <div className="flex justify-between items-end">
              <div>
                <h4 className="text-4xl font-black tracking-tighter capitalize">{activeTab} Multi-Horizon Projection</h4>
                <p className="text-slate-500 text-sm font-medium mt-1">14-Day Trajectory for {selection.commodity} in {selection.mandi}</p>
              </div>
              <div className="flex gap-2">
                  <span className="px-5 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 font-black text-[10px] uppercase tracking-widest">Neural Mode</span>
                  <span className="px-5 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-black text-[10px] uppercase tracking-widest">Sync OK</span>
              </div>
            </div>

            <div className="flex-1 min-h-[400px]">
              <Line data={{
                labels: [...data.history.map(h => h[0].slice(5)), ...data.actual.map(a => a[0].slice(5))],
                datasets: [
                  { label: 'Observed History', data: [...data.history.map(h => h[1]), ...Array(14).fill(null)], borderColor: '#475569', pointRadius: 0, tension: 0.3, fill: false },
                  { label: 'Ground Truth', data: [...Array(data.history.length-1).fill(null), data.history[data.history.length-1][1], ...data.actual.map(a => a[1])], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', fill: true, tension: 0.3, borderWidth: 4 },
                  { label: 'AI Prediction', data: [...Array(data.history.length-1).fill(null), data.history[data.history.length-1][1], ...data.forecast.map(f => f[1])], borderColor: '#3b82f6', borderDash: [6, 4], tension: 0.3, borderWidth: 3, pointBackgroundColor: '#3b82f6' }
                ]
              }} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top', labels: { color: '#64748b', font: { weight: 'bold', size: 10 } } } }, scales: { y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', font: { family: 'Inter' } } }, x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 10 } } } } }} />
            </div>

            <div className="grid grid-cols-4 gap-6 pt-10 border-t border-slate-800/60">
              <div className="text-center"><p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">RMSE</p><p className="text-2xl font-black text-blue-400">{activeTab === 'tft' ? '319.8' : '160.4'}</p></div>
              <div className="text-center"><p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">MAPE</p><p className="text-2xl font-black text-emerald-400">{activeTab === 'tft' ? '13.4%' : '8.8%'}</p></div>
              <div className="text-center"><p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Momentum</p><p className={`text-2xl font-black flex items-center justify-center gap-1 ${data.momentum[data.momentum.length-1] > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{data.momentum[data.momentum.length-1] > 0 ? <TrendingUpIcon size={20}/> : <TrendingDown size={20}/>}{Math.abs(data.momentum[data.momentum.length-1]).toFixed(1)}</p></div>
              <div className="text-center"><p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Horizon</p><p className="text-2xl font-black text-purple-400">14-D</p></div>
            </div>
          </motion.div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-slate-700 py-32">
            <Activity size={80} className="mb-8 opacity-10 animate-pulse" />
            <p className="font-black tracking-[0.3em] uppercase text-[12px] opacity-20 italic text-center leading-relaxed">Neural Pipeline Standby<br/><span className="text-[10px] font-normal tracking-widest opacity-50">Configure Market Parameters</span></p>
          </div>
        )}
      </section>
    </div>
  );
}
