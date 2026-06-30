"use client";
import { useState, useEffect, useCallback, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend, Filler
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

// const API = 'http://localhost:8000';
const API = 'https://huggingface.co/spaces/akashg7/agrisense-backend';

// ── Audit stats per module ────────────────────────────────────────
const AUDIT = {
  tft: [
    { label: 'Forecast Horizon', value: '14-Day', sub: 'Multi-Step', color: '#3b82f6' },
    { label: 'Accuracy (SMAPE)', value: '86.6%', sub: 'Test Metric', color: '#10b981' },
    { label: 'MAE', value: '319.9', sub: 'INR/qtl', color: '#8b5cf6' },
    { label: 'Input Encoder', value: '30-Day', sub: 'History', color: '#f59e0b' },
  ],
  lgbm: [
    { label: 'Accuracy (SMAPE)', value: '91.2%', sub: 'Verified', color: '#3b82f6' },
    { label: 'MAE', value: '160.4', sub: 'INR/qtl', color: '#10b981' },
    { label: 'Test Samples', value: '56.8k', sub: 'Evaluated', color: '#8b5cf6' },
    { label: 'R² Score', value: '0.94', sub: 'Variance', color: '#f59e0b' },
  ],
  crop: [
    { label: 'F1-Score', value: '0.97', sub: 'Weighted', color: '#3b82f6' },
    { label: 'Crop Classes', value: '22', sub: 'Varieties', color: '#10b981' },
    { label: 'Accuracy', value: '96.4%', sub: 'Classifier', color: '#8b5cf6' },
    { label: 'Derived Features', value: '40+', sub: 'Agronomic', color: '#f59e0b' },
  ],
  disease: [
    { label: 'Dataset Size', value: '87k+', sub: 'Images', color: '#3b82f6' },
    { label: 'Backbone', value: 'EffNet-B0', sub: 'Compound Scale', color: '#10b981' },
    { label: 'Accuracy', value: '98.4%', sub: 'Diagnosis', color: '#8b5cf6' },
    { label: 'Classes', value: '38', sub: 'Pathologies', color: '#f59e0b' },
  ],
};

const NAV = [
  { id: 'tft',     label: 'TFT Forecast',      icon: '⚡' },
  { id: 'lgbm',    label: 'LGBM Regression',    icon: '📈' },
  { id: 'crop',    label: 'Crop Intelligence',   icon: '🌱' },
  { id: 'disease', label: 'Disease Radar',       icon: '🔬' },
];

const CHART_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 400 },
  plugins: {
    legend: { position: 'top', labels: { color: '#64748b', font: { size: 11, weight: '600' }, boxWidth: 12 } },
    tooltip: { mode: 'index', intersect: false, backgroundColor: '#111827', borderColor: '#1e293b', borderWidth: 1, titleColor: '#f1f5f9', bodyColor: '#94a3b8' },
  },
  scales: {
    x: { grid: { display: false }, ticks: { color: '#475569', font: { size: 10 }, maxTicksLimit: 12 } },
    y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#475569', callback: v => '₹' + v.toLocaleString() } },
  },
};

export default function AgriSenseHub() {
  const [tab, setTab] = useState('tft');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [hierarchy, setHierarchy] = useState({});
  const [districts, setDistricts] = useState([]);
  const [mandis, setMandis] = useState([]);
  const [commodities, setCommodities] = useState([]);
  const [sel, setSel] = useState({ state: '', district: '', mandi: '', commodity: '' });
  const [cropInputs, setCropInputs] = useState({ N: 80, P: 40, K: 40, temp: 28, humidity: 65, ph: 6.5, rainfall: 120 });
  const [diseaseFile, setDiseaseFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [online, setOnline] = useState(false);
  const fileRef = useRef();

  const states = Object.keys(hierarchy).sort();

  useEffect(() => {
    const init = async () => {
      try {
        const r = await fetch(`${API}/api/hierarchy`);
        const data = await r.json();
        setHierarchy(data);
        setOnline(true);
      } catch { setTimeout(init, 2500); }
    };
    init();
  }, []);

  // Reset result when switching tabs
  const switchTab = useCallback((id) => {
    setTab(id);
    setResult(null);
  }, []);

  const onStateChange = useCallback((v) => {
    setSel({ state: v, district: '', mandi: '', commodity: '' });
    setDistricts(v ? Object.keys(hierarchy[v] || {}).sort() : []);
    setMandis([]);
    setCommodities([]);
    setResult(null);
  }, [hierarchy]);

  const onDistrictChange = useCallback((v) => {
    setSel(p => ({ ...p, district: v, mandi: '', commodity: '' }));
    setMandis(v ? (hierarchy[sel.state]?.[v] || []) : []);
    setCommodities([]);
  }, [hierarchy, sel.state]);

  const onMandiChange = useCallback(async (v) => {
    setSel(p => ({ ...p, mandi: v, commodity: '' }));
    if (!v) return;
    const r = await fetch(`${API}/api/commodities/${encodeURIComponent(v)}`);
    setCommodities(await r.json());
  }, []);

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setResult(null);
    setSidebarOpen(false);
    try {
      if (tab === 'tft' || tab === 'lgbm') {
        const r = await fetch(`${API}/api/${tab}/predict`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mandi: sel.mandi, commodity: sel.commodity }),
        });
        setResult(await r.json());
      } else if (tab === 'crop') {
        const r = await fetch(`${API}/api/crop/recommend`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(cropInputs),
        });
        const d = await r.json();
        setResult({ type: 'crop', items: d.recommendations });
      } else if (tab === 'disease') {
        const fd = new FormData();
        fd.append('file', diseaseFile);
        const r = await fetch(`${API}/api/disease/detect`, { method: 'POST', body: fd });
        setResult({ type: 'disease', ...(await r.json()) });
      }
    } catch (e) {
      alert('Analysis failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  }, [tab, sel, cropInputs, diseaseFile]);

  const isPriceTab = tab === 'tft' || tab === 'lgbm';
  const canRun = isPriceTab ? (sel.mandi && sel.commodity) : tab === 'disease' ? !!diseaseFile : true;

  const chartData = result && isPriceTab ? (() => {
    const histLen = result.history?.length || 0;
    const histLabels = result.history?.map(h => h[0].slice(5)) || [];
    const actLabels  = result.actual?.map(a => a[0].slice(5)) || [];
    const fcLabels   = result.forecast?.map(f => f[0].slice(5)) || [];
    const labels = [...histLabels, ...actLabels];
    const histData = [...result.history.map(h => h[1]), ...Array(actLabels.length).fill(null)];
    const actData  = [...Array(histLen - 1).fill(null), result.history[histLen-1][1], ...result.actual.map(a => a[1])];
    const fcData   = [...Array(histLen - 1).fill(null), result.history[histLen-1][1], ...result.forecast.map(f => f[1])];
    return {
      labels,
      datasets: [
        { label: 'History', data: histData, borderColor: '#334155', pointRadius: 0, tension: 0.3, fill: false, borderWidth: 1.5 },
        { label: 'Actual', data: actData, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.08)', fill: true, tension: 0.3, borderWidth: 2.5, pointRadius: 2 },
        { label: 'Forecast', data: fcData, borderColor: '#3b82f6', borderDash: [5, 4], tension: 0.3, borderWidth: 2, pointRadius: 2.5, pointBackgroundColor: '#3b82f6', fill: false },
      ],
    };
  })() : null;

  return (
    <div className="hub-layout">
      {/* ── SIDEBAR ── */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-logo">
          AgriSense
          <span>Intelligence Hub</span>
        </div>

        {NAV.map(n => (
          <button key={n.id} className={`nav-item ${tab === n.id ? 'active' : ''}`}
            onClick={() => switchTab(n.id)}>
            <span style={{ fontSize: 16 }}>{n.icon}</span> {n.label}
          </button>
        ))}

        <div className={`sidebar-footer ${online ? 'online' : 'offline'}`} style={{ marginTop: 'auto' }}>
          {online ? '● Brain Cluster Online' : '○ Syncing Engine...'}
        </div>
      </aside>

      {/* ── MAIN ── */}
      <div className="hub-main">
        {/* Topbar */}
        <div className="hub-topbar">
          <button className="topbar-btn" onClick={() => setSidebarOpen(p => !p)}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
            </svg>
          </button>
          <span className="topbar-title">{NAV.find(n => n.id === tab)?.label}</span>
          <span className="topbar-sub">AgriSense v2.4 — Production</span>
        </div>

        {/* Audit Strip */}
        <div className="audit-strip">
          {AUDIT[tab].map((s, i) => (
            <div className="audit-chip" key={i}>
              <div className="audit-chip-icon" style={{ background: s.color + '18' }}>
                <span style={{ fontSize: 13 }}>
                  {['📊','⚙️','🎯','⏱'][i]}
                </span>
              </div>
              <div>
                <div className="audit-chip-label">{s.label}</div>
                <div className="audit-chip-value" style={{ color: s.color }}>{s.value}</div>
                <div className="audit-chip-sub">{s.sub}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Content */}
        <div className="hub-content">
          <div className="module-grid page-fade" key={tab}>

            {/* ── LEFT PANEL (Inputs) ── */}
            <div className="panel">

              {/* PRICE INPUTS */}
              {isPriceTab && (
                <>
                  <div className="panel-title">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                    Market Context
                  </div>
                  {[
                    { label: 'State', opts: states, key: 'state', onChange: e => onStateChange(e.target.value) },
                    { label: 'District', opts: districts, key: 'district', onChange: e => onDistrictChange(e.target.value) },
                    { label: 'Mandi', opts: mandis, key: 'mandi', onChange: e => onMandiChange(e.target.value) },
                    { label: 'Commodity', opts: commodities, key: 'commodity', onChange: e => setSel(p => ({ ...p, commodity: e.target.value })) },
                  ].map(f => (
                    <div className="form-group" key={f.key}>
                      <label className="form-label">{f.label}</label>
                      <select className="form-select" value={sel[f.key]} onChange={f.onChange}>
                        <option value="">Select {f.label}…</option>
                        {f.opts.map(o => <option key={o} value={o}>{o}</option>)}
                      </select>
                    </div>
                  ))}
                </>
              )}

              {/* CROP INPUTS */}
              {tab === 'crop' && (
                <>
                  <div className="panel-title">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22V12m0 0C12 7 7 3 2 3c0 5 4 9 10 9zm0 0c0-5 5-9 10-9-1 5-5 9-10 9"/></svg>
                    Soil & Environment
                  </div>
                  <div className="npk-grid">
                    {['N', 'P', 'K'].map(k => (
                      <div className="form-group" key={k} style={{ marginBottom: 0 }}>
                        <label className="form-label">{k}</label>
                        <input type="number" className="form-input" value={cropInputs[k]}
                          onChange={e => setCropInputs(p => ({ ...p, [k]: +e.target.value }))} />
                      </div>
                    ))}
                  </div>
                  {[
                    { id: 'temp', label: 'Temperature', unit: '°C', max: 50 },
                    { id: 'humidity', label: 'Humidity', unit: '%', max: 100 },
                    { id: 'ph', label: 'Soil pH', unit: '', max: 14, step: 0.1 },
                    { id: 'rainfall', label: 'Rainfall', unit: 'mm', max: 300 },
                  ].map(f => (
                    <div className="slider-row" key={f.id}>
                      <div className="slider-header">
                        <label className="form-label" style={{ marginBottom: 0 }}>{f.label}</label>
                        <span className="slider-value">{cropInputs[f.id]}{f.unit}</span>
                      </div>
                      <input type="range" min={0} max={f.max} step={f.step || 1}
                        value={cropInputs[f.id]}
                        onChange={e => setCropInputs(p => ({ ...p, [f.id]: +e.target.value }))} />
                    </div>
                  ))}
                </>
              )}

              {/* DISEASE INPUTS */}
              {tab === 'disease' && (
                <>
                  <div className="panel-title">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
                    Field Sample Upload
                  </div>
                  <div className="upload-zone" onClick={() => fileRef.current?.click()}>
                    <input ref={fileRef} type="file" accept="image/*" hidden onChange={e => setDiseaseFile(e.target.files[0])} />
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                    <p>Upload Leaf Image</p>
                    <span>JPG, PNG, WEBP</span>
                  </div>
                  {diseaseFile && (
                    <div className="file-badge">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>
                      {diseaseFile.name}
                    </div>
                  )}
                </>
              )}

              <button
                className={`btn-primary btn-${tab}`}
                disabled={loading || !canRun}
                onClick={runAnalysis}
              >
                {loading ? <><span className="spinner"/>Running…</> : 'Run Analysis'}
              </button>
            </div>

            {/* ── RIGHT PANEL (Results) ── */}
            <div className="results-panel">
              {!result ? (
                <div className="results-empty">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                  </svg>
                  <p>Configure inputs and run analysis</p>
                </div>
              ) : isPriceTab ? (
                <>
                  <div className="result-header">
                    <div>
                      <div className="result-title">{sel.commodity} — {sel.mandi}</div>
                      <div className="result-sub">14-day forecast via {tab.toUpperCase()} model</div>
                    </div>
                    <div className="badge-row">
                      <span className="badge badge-green">Inference OK</span>
                      <span className="badge badge-purple">{tab === 'tft' ? '52 Features' : '16 Features'}</span>
                    </div>
                  </div>
                  <div className="chart-container">
                    {chartData && <Line data={chartData} options={CHART_OPTS} />}
                  </div>
                  <div className="metrics-row">
                    <div className="metric-chip">
                      <div className="metric-label">RMSE</div>
                      <div className="metric-value" style={{ color: '#3b82f6' }}>{tab === 'tft' ? '319.8' : '160.4'}</div>
                    </div>
                    <div className="metric-chip">
                      <div className="metric-label">MAPE</div>
                      <div className="metric-value" style={{ color: '#10b981' }}>{tab === 'tft' ? '13.4%' : '8.8%'}</div>
                    </div>
                    <div className="metric-chip">
                      <div className="metric-label">Momentum</div>
                      <div className="metric-value" style={{ color: result.momentum?.at(-1) > 0 ? '#10b981' : '#ef4444' }}>
                        {result.momentum ? (result.momentum.at(-1) > 0 ? '+' : '') + result.momentum.at(-1).toFixed(1) : '—'}
                      </div>
                    </div>
                    <div className="metric-chip">
                      <div className="metric-label">Horizon</div>
                      <div className="metric-value" style={{ color: '#8b5cf6' }}>14-D</div>
                    </div>
                  </div>
                </>
              ) : result.type === 'crop' ? (
                <>
                  <div className="result-header">
                    <div>
                      <div className="result-title">Top Crop Matches</div>
                      <div className="result-sub">Based on soil and environment inputs</div>
                    </div>
                  </div>
                  {result.items.map((c, i) => (
                    <div key={i} className={`rec-card ${i === 0 ? 'rank-1' : ''}`}>
                      <div className={`rec-rank ${i === 0 ? 'rank-1' : 'other'}`}>{String(i + 1).padStart(2, '0')}</div>
                      <div style={{ flex: 1 }}>
                        <div className="rec-crop">{c.name}</div>
                        <div className="rec-reason">{c.reason}</div>
                      </div>
                      <div className="rec-confidence">{(c.confidence * 100).toFixed(0)}%</div>
                    </div>
                  ))}
                </>
              ) : result.type === 'disease' ? (
                <>
                  <div className="result-header">
                    <div>
                      <div className="result-title">Diagnosis Report</div>
                      <div className="result-sub">AI-powered field pathology analysis</div>
                    </div>
                  </div>
                  <div className="diagnosis-card">
                    <div className="diagnosis-name">{result.diagnosis}</div>
                    <div className="badge-row">
                      <span className="badge badge-purple">Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                      <span className="badge badge-green">Protocol Ready</span>
                    </div>
                    <div className="treatment-box">
                      <div className="treatment-label">Treatment Protocol</div>
                      <div className="treatment-text">{result.treatment}</div>
                    </div>
                  </div>
                  <div className="form-label" style={{ marginBottom: 8 }}>Differential Diagnosis</div>
                  <div className="alternatives-row">
                    {result.alternatives?.map((a, i) => (
                      <span key={i} className="badge badge-amber">? {a}</span>
                    ))}
                  </div>
                </>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
