import React from 'react';
import { motion } from 'framer-motion';
import { Zap, TrendingUp, Layers, ShieldCheck, X, Menu } from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, isSidebarOpen, setIsSidebarOpen, systemOnline }) {
  const NavItem = ({ id, label, icon: Icon }) => (
    <button onClick={() => { setActiveTab(id); }}
      className={`w-full flex items-center gap-4 px-6 py-4 rounded-2xl transition-all duration-300 ${activeTab === id ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shadow-[0_0_20px_rgba(16,185,129,0.15)]' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-900'}`}
    >
      <Icon size={20} /> <span className="font-bold text-sm tracking-wide">{label}</span>
    </button>
  );

  return (
    <>
      {!isSidebarOpen && (
        <button onClick={() => setIsSidebarOpen(true)} className="fixed top-6 left-6 z-50 p-4 bg-slate-900/80 backdrop-blur-xl rounded-2xl border border-slate-800 shadow-2xl transition-all hover:scale-105 active:scale-95">
          <Menu size={24} className="text-emerald-400" />
        </button>
      )}
      
      <motion.aside initial={{ x: -320 }} animate={{ x: isSidebarOpen ? 0 : -320 }} className={`fixed lg:relative w-80 border-r border-slate-800/60 p-8 flex flex-col gap-10 bg-[#0d121f] z-40 h-screen shrink-0 transition-transform duration-300`}>
        <button onClick={() => setIsSidebarOpen(false)} className="absolute top-6 right-6 p-2 text-slate-600 hover:text-slate-400 transition-colors"><X size={20}/></button>
        
        <div className="space-y-1">
          <h1 className="text-3xl font-black tracking-tighter bg-gradient-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent uppercase">AgriSense</h1>
          <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-700">Intelligence Hub</p>
        </div>

        <nav className="flex-1 space-y-2">
          <NavItem id="tft" label="Price Forecast (TFT)" icon={Zap} />
          <NavItem id="lgbm" label="Price Forecast (LGBM)" icon={TrendingUp} />
          <NavItem id="crop" label="Crop Intelligence" icon={Layers} />
          <NavItem id="disease" label="Disease Radar" icon={ShieldCheck} />
        </nav>

        <div className="space-y-4 pt-6 border-t border-slate-800/60">
            <div className={`p-4 rounded-2xl border text-[10px] font-black tracking-[0.1em] ${systemOnline ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-amber-500/10 border-amber-500/30 text-amber-400 animate-pulse'}`}>
            {systemOnline ? '● BRAIN CLUSTER ONLINE' : '○ ENGINE SYNCING...'}
            </div>
            <p className="text-[10px] text-center text-slate-600 font-bold uppercase tracking-widest">v2.4.0 Production</p>
        </div>
      </motion.aside>
    </>
  );
}
