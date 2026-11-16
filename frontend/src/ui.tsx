import { lazy, Suspense, useState } from 'react';

const Plot = lazy(() => import('react-plotly.js'));

type Result = {
  video_id: string; video_url: string; video: { width: number; height: number; fps: number; duration_seconds: number };
  pose: { status: string; reason?: string; model?: string; detected_frames?: number; frames: Array<{ frame: number; timestamp_seconds: number; status: string; detection?: string; landmarks?: Array<{x:number;y:number;z:number;pixel_x:number;pixel_y:number;visibility:number}> }> };
  segmentation: { status: string; reason?: string; model?: string; overlay_url?: string; detected_frames?: number };
  sampling: { status: string; sample_fps: number; effective_sample_fps: number; sampled_frames: number; truncated: boolean };
  overlay: { status: string; url: string | null };
  processed_video: { status: string; url: string | null; reason?: string };
  segmentation_video: { status: string; url: string | null; reason?: string };
  movement_measurements: { status: string; reason?: string; coordinate_space?: string; physical_calibration?: string; frames?: Array<Record<string, number | null>>; valid_measurement_counts?: Record<string, number>; symmetry_valid_frame_count?: number; symmetry_valid_interval_count?: number; symmetry_definition?: string; symmetry_reason?: string; left_right_symmetry?: number; movement_duration_seconds?: number; repetition_count?: {status: string; reason: string} };
  prediction: { status: string; reason?: string };
  evaluation: { status: string; reason?: string; tasks?: Record<string, Record<string, number | string | boolean>> };
};

// The API origin is configurable for deployment; development defaults to FastAPI.
const api = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export default function App() {
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [evaluationBusy, setEvaluationBusy] = useState('');
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
  async function evaluate(task: 'pose' | 'segmentation', file?: File) {
    if (!file || !result) return;
    setEvaluationBusy(task); setError('');
    try {
      const annotation = JSON.parse(await file.text());
      const response = await fetch(`${api}/api/videos/${result.video_id}/evaluation/${task}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(annotation),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? 'Evaluation failed');
      setResult(current => current ? {
        ...current,
        evaluation: {
          status: 'completed',
          tasks: { ...(current.evaluation.tasks ?? {}), [task]: body },
        },
      } : current);
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not evaluate annotations'); }
    finally { setEvaluationBusy(''); }
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
      <div className="video-panel"><div className="panel-title">SOURCE VIDEO <span>{result.video.width} × {result.video.height} · {result.video.fps.toFixed(1)} FPS · {result.video.duration_seconds.toFixed(1)} s</span></div><VideoPlayer src={`${api}${result.video_url}`} title="Source video"/><div className="panel-title overlay-title">SAMPLED VIDEO WITH POSE OVERLAY <span>{result.pose.model ?? result.processed_video.status}</span></div>{result.processed_video.url ? <VideoPlayer src={`${api}${result.processed_video.url}`} title="Pose overlay video"/> : <p className="unavailable">Pose video {result.processed_video.status.replaceAll('_',' ')}{result.processed_video.reason ? `: ${result.processed_video.reason}` : ''}</p>}<div className="panel-title overlay-title">HUMAN SEGMENTATION <span>{result.segmentation.model ?? result.segmentation_video.status}</span></div>{result.segmentation_video.url ? <VideoPlayer src={`${api}${result.segmentation_video.url}`} title="Segmentation video"/> : <p className="unavailable">Segmentation video {result.segmentation_video.status.replaceAll('_',' ')}{result.segmentation_video.reason ? `: ${result.segmentation_video.reason}` : ''}</p>}</div>
      <div className="panel metrics"><div className="panel-title">ANALYSIS STATUS</div><p>{result.sampling.sampled_frames} frames sampled at {result.sampling.effective_sample_fps.toFixed(1)} FPS{result.sampling.truncated ? ' · analysis frame limit reached' : ''}</p><Status label="Pose estimation" value={result.pose.status}/>{result.pose.reason && <p className="unavailable">{result.pose.reason}</p>}<small>{result.pose.detected_frames ?? 0} / {result.pose.frames.length} sampled frames contain a person</small><Status label="Human segmentation" value={result.segmentation.status}/><small>{result.segmentation.detected_frames ?? 0} / {result.sampling.sampled_frames} sampled frames contain a person mask</small>{result.segmentation.reason && <p className="unavailable">{result.segmentation.reason}</p>}<Status label="Movement measurements" value={result.movement_measurements.status}/>{result.movement_measurements.reason && <p className="unavailable">{result.movement_measurements.reason}</p>}<Status label="Model prediction" value={result.prediction.status}/><p className="unavailable">{result.prediction.reason}</p><Status label="Evaluation" value={result.evaluation.status}/>{result.evaluation.reason && <p className="unavailable">{result.evaluation.reason}</p>}{(['pose','segmentation'] as const).map(task => { const scored = result.evaluation.tasks?.[task]; return scored && <p key={task} className="evaluation-result">{task === 'pose' ? `Pose · MPJPE ${Number(scored.mpjpe).toFixed(4)} · PCK@0.05 ${Number(scored.pck).toFixed(3)}` : `Segmentation · mIoU ${Number(scored.mean_iou).toFixed(3)} · Dice ${Number(scored.mean_dice).toFixed(3)}`}</p>; })}<label className="annotation-upload">{evaluationBusy === 'pose' ? 'Evaluating pose…' : 'Evaluate pose ground truth'}<input type="file" accept="application/json,.json" disabled={!!evaluationBusy} onChange={e => evaluate('pose', e.target.files?.[0])}/></label><label className="annotation-upload">{evaluationBusy === 'segmentation' ? 'Evaluating masks…' : 'Evaluate segmentation masks'}<input type="file" accept="application/json,.json" disabled={!!evaluationBusy} onChange={e => evaluate('segmentation', e.target.files?.[0])}/></label>{error && <p className="error">{error}</p>}<a className="download" href={`${api}/api/videos/${result.video_id}/report.csv`}>Download landmark CSV ↓</a></div>
      <div className="panel chart"><div className="panel-title">POSE LANDMARKS <span>normalized (hover for pixel coordinates)</span></div>{sample.length ? <Suspense fallback={<div className="unavailable">Loading chart…</div>}><Plot data={[{x:sample.map(p=>p.x),y:sample.map(p=>p.y),mode:'markers',type:'scatter',marker:{color:'#20c997',size:8},text:sample.map((p,i)=>`Landmark ${i}<br>Pixel (${p.pixel_x.toFixed(0)}, ${p.pixel_y.toFixed(0)})<br>Visibility ${p.visibility.toFixed(2)}`),hoverinfo:'text'}]} layout={{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#b9c5c8'},xaxis:{title:{text:'Normalized x'},range:[0,1]},yaxis:{title:{text:'Normalized y'},autorange:'reversed',range:[1,0]}}} style={{width:'100%',height:'360px'}} useResizeHandler/></Suspense> : <div className="unavailable">{result.pose.reason ?? 'No person was detected in sampled frames.'}</div>}{result.movement_measurements.status === 'completed' && result.movement_measurements.frames?.length ? <>
        <div className="panel-title">KNEE ANGLES OVER TIME · DEGREES</div>
        <p>Internal angle is the geometric joint angle. Flexion = 180° − internal angle. Counts in the legend are valid samples; gaps remain missing.</p>
        <Suspense fallback={<div className="unavailable">Loading chart…</div>}><Plot data={[
          ['left_knee_internal_angle_deg','left_knee_internal_angle','Left internal angle'],
          ['right_knee_internal_angle_deg','right_knee_internal_angle','Right internal angle'],
          ['left_knee_flexion_deg','left_knee_flexion','Left flexion'],
          ['right_knee_flexion_deg','right_knee_flexion','Right flexion'],
        ].map(([key,countKey,label]) => {
          const values = result.movement_measurements.frames!.map(row => row[key]);
          const count = result.movement_measurements.valid_measurement_counts?.[countKey] ?? values.filter(v => v !== null).length;
          return {x:result.movement_measurements.frames!.map(row=>row.timestamp_seconds),y:values,mode:'lines+markers',type:'scatter' as const,name:`${label} (n=${count})`,connectgaps:false};
        })} layout={{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#b9c5c8'},xaxis:{title:{text:'Time (seconds)'}},yaxis:{title:{text:'Angle (degrees)'}}}} style={{width:'100%',height:'340px'}} useResizeHandler/></Suspense>
        <div className="panel-title">TRUNK INCLINATION</div>
        <p>Positive means toward image-right; negative means toward image-left. This sign does not identify anatomical forward/backward lean without a known camera view.</p>
        <Suspense fallback={<div className="unavailable">Loading chart…</div>}><Plot data={[{x:result.movement_measurements.frames.map(row=>row.timestamp_seconds),y:result.movement_measurements.frames.map(row=>row.trunk_inclination_deg),mode:'lines+markers',type:'scatter',name:`Trunk inclination (n=${result.movement_measurements.valid_measurement_counts?.trunk_inclination ?? 0})`,connectgaps:false}]} layout={{autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#b9c5c8'},xaxis:{title:{text:'Time (seconds)'}},yaxis:{title:{text:'Angle from image vertical (degrees)'}}}} style={{width:'100%',height:'280px'}} useResizeHandler/></Suspense>
        <p>Symmetry score: {result.movement_measurements.left_right_symmetry?.toFixed(3) ?? 'Not available'} (0–1 coefficient; not a percentage) · {result.movement_measurements.symmetry_valid_interval_count ?? 0} valid paired intervals across {result.movement_measurements.symmetry_valid_frame_count ?? 0} valid frames.{result.movement_measurements.symmetry_reason ? ` ${result.movement_measurements.symmetry_reason}` : ''}</p><small>{result.movement_measurements.symmetry_definition}</small>
      </> : <p className="unavailable">{result.movement_measurements.reason ?? 'Valid measurements are not available.'}</p>}</div>
      <div className="panel note"><div className="panel-title">INTERPRETATION</div><p>{result.movement_measurements.coordinate_space ?? 'Measurements unavailable.'} Measurements are image-space estimates, not physically calibrated or clinically validated. Evaluation requires ground-truth annotations. Analysis uses sampled frames at the configured rate.</p></div>
    </section>}
    {!result && <section className="cards"><article><strong>01</strong><h2>Pose & tracking</h2><p>Sampled COCO-17 keypoints from the pretrained YOLO11n pose model.</p></article><article><strong>02</strong><h2>Movement features</h2><p>2D joint angles, trunk inclination and normalized joint displacement from detected landmarks.</p></article><article><strong>03</strong><h2>Evaluation</h2><p>Pose MPJPE and PCK are reported only after ground-truth landmarks are submitted.</p></article></section>}
    <footer>MotionIQ · Research software · No clinical claims</footer>
  </main>;
}

function Status({label,value}:{label:string;value:string}) { return <div className="status-row"><span>{label}</span><b className={value === 'completed' ? 'available' : ''}>{value.replaceAll('_',' ')}</b></div>; }

function VideoPlayer({src,title}:{src:string;title:string}) {
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  return <div className="video-player"><video key={src} controls preload="metadata" src={src} onLoadedMetadata={() => setError('')} onCanPlay={() => setReady(true)} onError={() => { setReady(false); setError('Video could not be loaded. Check that the backend artifact exists and uses a browser-supported codec.'); }} aria-label={title}/>{!ready && !error && <small>Loading video…</small>}{error && <p className="error" role="alert">{error}</p>}</div>;
}
