import React from 'react';
import { motion } from 'framer-motion';
import { FlaskConical, CheckCircle2, Activity, Info } from 'lucide-react';

export default function CropIntelligenceView({ cropInputs, setCropInputs, data, loading, runAnalysis }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-10 items-start">
      {/* ENV INPUTS */}
      <section className="p-8 rounded-[40px] bg-slate-900/40 border border-slate-800/60 backdrop-blur-3xl shadow-2xl">
        <h3 className="text-xl font-bold mb-8 flex items-center gap-3"><FlaskConical size={20} className="text-emerald-400"/> Soil Ecology</h3>

        <div className="space-y-6">
          <div className="grid grid-cols-3 gap-4">
            {['N', 'P', 'K'].map(nut => (
              <div key={nut}>
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">{nut}</label>
                <input type="number" value={cropInputs[nut]} onChange={e => setCropInputs(p => ({...p, [nut]: e.target.value}))} className="w-full bg-slate-950/50 border border-slate-800 rounded-xl px-3 py-3 mt-1 text-sm outline-none focus:border-emerald-500/50" />
              </div>
            ))}
          </div>
          
          {[
            {id: 'temp', label: 'Temperature', unit: '°C', max: 50}, 
            {id: 'humidity', label: 'Humidity', unit: '%', max: 100}, 
            {id: 'ph', label: 'Soil pH', unit: '', max: 14, step: 0.1}, 
            {id: 'rainfall', label: 'Rainfall', unit: 'mm', max: 300}
          ].map(env => (
            <div key={env.id} className="space-y-3">
              <div className="flex justify-between items-center">
                <label className="text-[10px] font-black uppercase tracking-widest text-slate-500">{env.label}</label>
                <span className="text-xs font-black text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-lg border border-emerald-500/20">{cropInputs[env.id]}{env.unit}</span>
              </div>
              <input type="range" min="0" max={env.max} step={env.step || 1} value={cropInputs[env.id]} onChange={e => setCropInputs(p => ({...p, [env.id]: e.target.value}))} className="w-full accent-emerald-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer" />
            </div>
          ))}
        </div>

        <button onClick={runAnalysis} disabled={loading} className="w-full py-6 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 font-black tracking-widest uppercase text-xs hover:scale-[1.02] active:scale-95 transition-all mt-10 shadow-xl shadow-emerald-500/20">
          {loading ? "Matching Bio-Assets..." : "Compute Optimization"}
        </button>
      </section>

      {/* RECOMMENDATIONS */}
      <section className="p-10 rounded-[40px] bg-slate-900/40 border border-slate-800/60 backdrop-blur-3xl shadow-2xl min-h-[600px]">
        {data ? (
          <div className="space-y-8">
            <h4 className="text-3xl font-black tracking-tighter">Optimal Biological Matches</h4>
            <div className="grid grid-cols-1 gap-4">
              {data.results.map((crop, i) => (
                <motion.div key={i} initial={{ x: 20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: i * 0.1 }} 
                  className={`p-6 rounded-[28px] border transition-all hover:scale-[1.01] ${i === 0 ? 'bg-emerald-500/10 border-emerald-500/30 ring-1 ring-emerald-500/20 shadow-lg shadow-emerald-500/5' : 'bg-slate-950/50 border-slate-800/60'}`}>
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center gap-4">
                      <div className={`w-10 h-10 rounded-2xl flex items-center justify-center font-black text-sm ${i === 0 ? 'bg-emerald-500 text-slate-950' : 'bg-slate-800 text-slate-400'}`}>0{i + 1}</div>
                      <h5 className="text-2xl font-black">{crop.name}</h5>
                    </div>
                    <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest bg-slate-950 px-3 py-1.5 rounded-full border border-slate-800">{(crop.confidence * 100).toFixed(0)}% AI Confidence</div>
                  </div>
                  <p className="text-slate-400 text-sm leading-relaxed flex gap-3"><CheckCircle2 size={18} className="shrink-0 text-emerald-500 mt-0.5" /> {crop.reason}</p>
                </motion.div>
              ))}
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-slate-700 py-32">
            <Activity size={80} className="mb-8 opacity-10 animate-pulse" />
            <p className="font-black tracking-[0.3em] uppercase text-[12px] opacity-20 italic text-center leading-relaxed">Ecological Engine Standby<br/><span className="text-[10px] font-normal tracking-widest opacity-50">Sync Environmental Parameters</span></p>
          </div>
        )}
      </section>
    </div>
  );
}
