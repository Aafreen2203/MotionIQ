import { lazy, Suspense, useState } from 'react';

const Plot = lazy(() => import('react-plotly.js'));

type Result = {
  video_id: string; video_url: string; video: { width: number; height: number; fps: number; duration_seconds: number };
  pose: { status: string; frames: Array<{ frame: number; status: string; landmarks?: Array<{x:number;y:number;visibility:number}> }> };
  segmentation: { status: string; mask_count?: number };
  overlay: { status: string; url: string | null };
  movement_measurements: { status: string; reason?: string };
  prediction: { status: string; reason?: string };
  evaluation: { status: string; reason?: string };
};

const api = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export default function App() {
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function upload(file?: File) {
    if (!file) return;
    setBusy(true); setError(''); setResult(null);
    try {
      const form = new FormData(); form.append('file', file);
      const response = await fetch(`${api}/api/videos`, { method: 'POST', body: form });
      if (!response.ok) throw new Error((await response.json()).detail ?? 'Analysis failed');
      setResult(await response.json());
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to reach the analysis API'); }
    finally { setBusy(false); }
  }
  const sample = result?.pose.frames.find(frame => frame.landmarks?.length)?.landmarks ?? [];
  return <main>
    <header><div className="brand">M<span>·</span>Q</div><div><b>MotionIQ</b><small> MOVEMENT ANALYSIS</small></div><div className="status">RESEARCH PROTOTYPE</div></header>
    <section className="intro"><p className="eyebrow">COMPUTER VISION · BIOMECHANICS</p><h1>Understand movement<br/><em>with evidence.</em></h1><p>Upload a movement video to inspect available pose and segmentation outputs.</p>
      <label className="upload">{busy ? 'Analyzing video…' : '↑  Choose a video'}<input type="file" accept="video/*" disabled={busy} onChange={e => upload(e.target.files?.[0])}/></label>
      <small>MP4, MOV, AVI or MKV · Analysis runs locally on the configured API</small>
      {error && <p className="error">{error}</p>}
    </section>
    {result && <section className="workspace">
      <div className="video-panel"><div className="panel-title">SOURCE VIDEO <span>{result.video.width} × {result.video.height} · {result.video.fps.toFixed(1)} FPS</span></div><video controls src={`${api}${result.video_url}`}/>{result.overlay.status === 'available' && result.overlay.url && <><div className="panel-title overlay-title">POSE / SEGMENTATION OVERLAY <span>sampled frame · unvalidated</span></div><img className="overlay-image" src={`${api}${result.overlay.url}`} alt="Sampled frame with pose or segmentation overlay"/></>}</div>
      <div className="panel metrics"><div className="panel-title">ANALYSIS STATUS</div><Status label="Pose estimation" value={result.pose.status}/><Status label="Human segmentation" value={result.segmentation.status}/><Status label="Movement measurements" value={result.movement_measurements.status}/><Status label="Model prediction" value={result.prediction.status}/><Status label="Evaluation" value={result.evaluation.status}/><a className="download" href={`${api}/api/videos/${result.video_id}/report.csv`}>Download landmark CSV ↓</a></div>
      <div className="panel chart"><div className="panel-title">POSE LANDMARKS <span>sampled frame</span></div>{sample.length ? <Suspense fallback={<div className="unavailable">Loading chart…</div>}><Plot data={[{x:sample.map(p=>p.x),y:sample.map(p=>p.y),mode:'markers',type:'scatter',marker:{color:'#20c997',size:8},text:sample.map((_,i)=>`Landmark ${i}`),hoverinfo:'text'}]} layout={{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#b9c5c8'},xaxis:{title:{text:'Normalized x'},range:[0,1]},yaxis:{title:{text:'Normalized y'},autorange:'reversed',range:[1,0]}}} style={{width:'100%',height:'360px'}} useResizeHandler/></Suspense> : <div className="unavailable">{result.pose.status === 'unavailable' ? 'Install MediaPipe to enable pose landmarks.' : 'No pose found in sampled frames.'}</div>}</div>
      <div className="panel note"><div className="panel-title">INTERPRETATION</div><p>Measurements, predictions, overlays and evaluation are labeled unavailable until the required models, calibration, and ground-truth data are configured. This prototype is not clinically validated.</p></div>
    </section>}
    {!result && <section className="cards"><article><strong>01</strong><h2>Pose & tracking</h2><p>Frame-wise landmarks with confidence values when MediaPipe is installed.</p></article><article><strong>02</strong><h2>Movement features</h2><p>Geometric and temporal analysis functions are available to the research pipeline.</p></article><article><strong>03</strong><h2>Reproducible evaluation</h2><p>Pose, segmentation and classification metric implementations require ground truth.</p></article></section>}
    <footer>MotionIQ · Research software · No clinical claims</footer>
  </main>;
}

function Status({label,value}:{label:string;value:string}) { return <div className="status-row"><span>{label}</span><b className={value === 'ok' ? 'available' : ''}>{value.replace(/_/g,' ')}</b></div>; }
