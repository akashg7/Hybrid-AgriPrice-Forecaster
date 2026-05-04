import React from 'react';
import { motion } from 'framer-motion';
import { Microscope, Upload, ShieldCheck, CheckCircle2, AlertCircle, Activity } from 'lucide-react';

export default function DiseaseRadarView({ diseaseFile, setDiseaseFile, data, loading, runAnalysis }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-10 items-start">
      {/* SCANNER INPUT */}
      <section className="p-8 rounded-[40px] bg-slate-900/40 border border-slate-800/60 backdrop-blur-3xl shadow-2xl">
        <h3 className="text-xl font-bold mb-8 flex items-center gap-3"><Microscope size={20} className="text-purple-400"/> Field Scan</h3>
        
        <div className="space-y-8">
          <div className="border-2 border-dashed border-slate-800 rounded-[32px] p-12 text-center hover:border-purple-500/50 transition-all cursor-pointer bg-slate-950/30 group" onClick={() => document.getElementById('file-up').click()}>
            <input type="file" id="file-up" hidden onChange={e => setDiseaseFile(e.target.files[0])} />
            <div className="w-16 h-16 rounded-3xl bg-purple-500/10 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-all duration-300">
              <Upload className="text-purple-400" size={28} />
            </div>
            <p className="text-slate-300 font-bold text-sm">Upload Bio-Sample</p>
            <p className="text-[10px] text-slate-600 font-black uppercase tracking-widest mt-2">JPEG / PNG High Res</p>
          </div>
          {diseaseFile && (
            <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="p-4 rounded-2xl bg-purple-500/10 border border-purple-500/20 text-purple-400 text-[10px] font-black text-center uppercase tracking-widest italic flex items-center justify-center gap-2">
              <CheckCircle2 size={12}/> {diseaseFile.name}
            </motion.div>
          )}
        </div>

        <button onClick={runAnalysis} disabled={loading || !diseaseFile} className="w-full py-6 rounded-2xl bg-gradient-to-r from-purple-600 to-indigo-600 font-black tracking-widest uppercase text-xs hover:scale-[1.02] active:scale-95 transition-all mt-10 shadow-xl shadow-purple-500/20 disabled:opacity-50 disabled:grayscale">
          {loading ? "Sequencing DNA..." : "Initiate AI Diagnosis"}
        </button>
      </section>

      {/* DIAGNOSIS */}
      <section className="p-10 rounded-[40px] bg-slate-900/40 border border-slate-800/60 backdrop-blur-3xl shadow-2xl min-h-[600px]">
        {data ? (
          <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="space-y-10">
            <div className="flex items-center gap-8">
              <div className="w-24 h-24 bg-purple-500/10 rounded-[40px] flex items-center justify-center border border-purple-500/20 shadow-xl shadow-purple-500/5"><ShieldCheck size={48} className="text-purple-400" /></div>
              <div>
                <h4 className="text-5xl font-black text-slate-100 tracking-tighter">{data.diagnosis}</h4>
                <div className="flex items-center gap-2 mt-3">
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] bg-purple-500/20 px-4 py-1.5 rounded-full text-purple-300 border border-purple-500/30">Neural Confidence: {data.confidence * 100}%</span>
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] bg-emerald-500/10 px-4 py-1.5 rounded-full text-emerald-400 border border-emerald-500/20">Protocol Verified</span>
                </div>
              </div>
            </div>

            <div className="p-10 rounded-[48px] bg-slate-950/80 border border-slate-800/60 space-y-8 shadow-2xl relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/5 blur-[100px] rounded-full -mr-32 -mt-32 transition-all group-hover:bg-emerald-500/10"></div>
              
              <div className="space-y-4 relative z-10">
                <h5 className="text-emerald-400 font-black text-[11px] uppercase tracking-[0.3em] flex items-center gap-3 italic">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                  AI Treatment Protocol
                </h5>
                <p className="text-slate-200 leading-relaxed font-bold text-2xl tracking-tight">{data.treatment}</p>
              </div>

              <div className="h-px bg-slate-800/60 w-full" />

              <div className="space-y-5 relative z-10">
                <h5 className="text-slate-500 font-black text-[11px] uppercase tracking-[0.3em] flex items-center gap-3">
                  <AlertCircle size={14} className="text-amber-500" />
                  Differential Consideration
                </h5>
                <div className="flex flex-wrap gap-3">
                  {data.alternatives.map((alt, i) => (
                    <span key={i} className="px-5 py-3 rounded-2xl bg-slate-900/80 border border-slate-800 text-xs font-black text-slate-400 uppercase tracking-widest hover:border-slate-700 transition-colors">
                      <span className="text-slate-600 mr-2">?</span> {alt}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-slate-700 py-32">
            <Activity size={80} className="mb-8 opacity-10 animate-pulse" />
            <p className="font-black tracking-[0.3em] uppercase text-[12px] opacity-20 italic text-center leading-relaxed">Pathogen Radar Standby<br/><span className="text-[10px] font-normal tracking-widest opacity-50">Upload Biological Sample</span></p>
          </div>
        )}
      </section>
    </div>
  );
}
