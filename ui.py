"""
Premium web UI for the script-to-video tool.

    python ui.py            # then open http://localhost:8000

Workflow mirrors the CLI but visual:
  1. Paste script + upload voiceover
  2. Analyze -> voice-aligned scene plan appears; edit any scene's
     search phrases inline
  3. Render -> live progress bar (reads the pipeline's checkpoint)
  4. Download finished videos from the Outputs panel
"""

import json
import subprocess
import sys
from pathlib import Path

import uvicorn
from clip_finder import ClipSearchError, find_candidates_for_scene
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from scene_parser import Scene

ROOT = Path(__file__).parent
UPLOADS = ROOT / "my_scripts"
OUTPUT = ROOT / "output"
TEMP = ROOT / "temp"
PLAN = UPLOADS / "ui_plan.json"
SCRIPT = UPLOADS / "ui_script.txt"
VOICE = UPLOADS / "ui_voiceover.mp3"
DEFAULT_BEAT_SECONDS = 3.0

app = FastAPI(title="Script to Video Studio")
_render_proc: subprocess.Popen | None = None
_expected_scenes: int = 0


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AI Editing Tool</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg:#090b10;--bg2:#0f131b;--panel:#121823;--panel2:#171e2b;--line:#273042;
  --text:#f3f0e8;--muted:#97a4bb;--accent:#f4ba41;--accent2:#c47b22;--ok:#45d48f;--warn:#ff8a5b;
  --shadow:0 22px 70px rgba(0,0,0,.45)
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  min-height:100vh;color:var(--text);
  font:14px/1.5 "Segoe UI",system-ui,sans-serif;
  background:
    radial-gradient(circle at top left, rgba(244,186,65,.16), transparent 28%),
    radial-gradient(circle at top right, rgba(110,95,255,.12), transparent 24%),
    linear-gradient(180deg, #0a0d12 0%, #090b10 100%)
}
.wrap{max-width:1320px;margin:0 auto;padding:28px 20px 64px}
header{
  display:flex;align-items:flex-start;justify-content:space-between;gap:20px;
  margin-bottom:24px;padding:18px 22px;border:1px solid rgba(255,255,255,.08);
  background:rgba(18,24,35,.78);backdrop-filter:blur(14px);border-radius:24px;box-shadow:var(--shadow)
}
.brand{display:flex;gap:16px;align-items:flex-start}
.logo{
  width:52px;height:52px;border-radius:16px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#121212;display:flex;align-items:center;justify-content:center;
  font-size:24px;font-weight:800;letter-spacing:.04em
}
h1{
  font:700 24px/1.1 Georgia,"Times New Roman",serif;
  letter-spacing:.02em;margin-bottom:6px
}
.sub{color:var(--muted);max-width:760px}
.hero-stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
.pill,.badge{
  display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);
  border-radius:999px;padding:6px 12px;background:rgba(255,255,255,.02);font-size:12px;color:var(--muted)
}
.badge.live{color:var(--ok);border-color:rgba(69,212,143,.45);background:rgba(69,212,143,.08)}
.shell{display:grid;grid-template-columns:380px minmax(0,1fr);gap:20px}
@media(max-width:1080px){.shell{grid-template-columns:1fr}}
.stack{display:grid;gap:20px}
.card{
  background:linear-gradient(180deg, rgba(18,24,35,.96), rgba(14,19,28,.96));
  border:1px solid rgba(255,255,255,.07);border-radius:24px;padding:20px;box-shadow:var(--shadow)
}
.card h2{
  font:700 12px/1 "Segoe UI",system-ui,sans-serif;
  letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:16px
}
.lead{color:var(--muted);margin-bottom:14px}
textarea{
  width:100%;min-height:320px;resize:vertical;border-radius:18px;border:1px solid var(--line);
  background:var(--panel2);color:var(--text);padding:16px;
  font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace
}
input[type=file]{width:100%;color:var(--muted);font-size:13px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.controls{display:grid;gap:12px}
.field{
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:12px 14px;border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.02)
}
.field span{color:var(--muted);font-size:13px}
.field input[type=number]{
  width:84px;border-radius:10px;border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:8px 10px
}
.checks{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.check{
  display:flex;gap:8px;align-items:center;padding:10px 12px;border:1px solid var(--line);
  border-radius:14px;background:rgba(255,255,255,.02);color:var(--muted);font-size:13px
}
button{
  border:0;border-radius:14px;padding:12px 16px;font-size:14px;font-weight:700;cursor:pointer;
  transition:transform .15s ease,opacity .15s ease,background .15s ease
}
button:hover{transform:translateY(-1px)}
button:disabled{opacity:.55;cursor:not-allowed;transform:none}
.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#17130c}
.ghost{background:var(--panel2);color:var(--text);border:1px solid var(--line)}
.progress{height:11px;background:var(--panel2);border-radius:999px;overflow:hidden;margin:14px 0 8px}
.progress i{display:block;height:100%;width:0;border-radius:999px;background:linear-gradient(90deg,var(--accent),#ffd78a);transition:width .6s ease}
.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:18px}
@media(max-width:760px){.summary{grid-template-columns:1fr}}
.metric{
  border:1px solid var(--line);border-radius:18px;padding:14px;background:rgba(255,255,255,.02)
}
.metric .label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.metric .value{font-size:22px;font-weight:700}
#warnings{display:grid;gap:8px;margin-bottom:14px}
.warning{
  border:1px solid rgba(255,138,91,.25);background:rgba(255,138,91,.08);color:#ffd7c8;
  border-radius:14px;padding:10px 12px;font-size:12px
}
#scenes{display:grid;gap:14px;max-height:calc(100vh - 250px);overflow:auto;padding-right:4px}
.scene{
  border:1px solid var(--line);border-radius:22px;padding:16px;background:linear-gradient(180deg, rgba(255,255,255,.025), rgba(255,255,255,.01))
}
.scene-top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:10px}
.scene-id{display:flex;gap:10px;align-items:center}
.scene-no{
  width:34px;height:34px;border-radius:12px;background:rgba(244,186,65,.12);border:1px solid rgba(244,186,65,.24);
  display:flex;align-items:center;justify-content:center;color:var(--accent);font-weight:800
}
.scene-meta{display:flex;gap:8px;flex-wrap:wrap}
.chip{
  border:1px solid var(--line);border-radius:999px;padding:5px 10px;font-size:11px;color:var(--muted);background:rgba(255,255,255,.02)
}
.scene-text{color:#d7ddeb;margin-bottom:12px}
.query-label{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.query-input{
  width:100%;border-radius:14px;border:1px solid var(--line);background:var(--panel2);
  color:var(--text);padding:12px 14px;font-size:13px;margin-bottom:12px
}
.cand-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.cand{
  border:1px solid var(--line);border-radius:18px;overflow:hidden;background:rgba(255,255,255,.025)
}
.thumb{
  aspect-ratio:16/9;background:#0c1018 center/cover no-repeat;border-bottom:1px solid var(--line)
}
.thumb.empty{
  display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;
  background:linear-gradient(135deg,#111826,#1a2231)
}
.cand-body{padding:10px}
.cand-title{font-size:12px;color:var(--text);margin-bottom:6px;min-height:36px}
.cand-meta{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:11px;margin-bottom:8px}
.cand button{width:100%;padding:9px 10px;font-size:12px}
.empty-plan{
  border:1px dashed var(--line);border-radius:24px;padding:32px;color:var(--muted);text-align:center
}
.out{
  display:flex;justify-content:space-between;align-items:center;gap:12px;
  padding:12px 0;border-bottom:1px solid var(--line)
}
.out:last-child{border-bottom:0}
.out a{color:var(--accent);text-decoration:none;font-weight:700}
.footer{margin-top:18px;color:var(--muted);font-size:12px;text-align:center}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">AI</div>
      <div>
        <h1>AI Editing Tool</h1>
        <div class="sub">Paste a script, break it into 3-second beats, preview matching visuals for the whole script, then render a local first cut.</div>
        <div class="hero-stats">
          <div class="pill">Local pipeline</div>
          <div class="pill">Voice-aligned timing</div>
          <div class="pill">Ranked stock matches</div>
        </div>
      </div>
    </div>
    <div id="state" class="badge">idle</div>
  </header>

  <div class="shell">
    <div class="stack">
      <div class="card">
        <h2>Script Input</h2>
        <div class="lead">Drop in your narration, optionally attach voiceover, and generate a visual plan for the full script.</div>
        <textarea id="script" placeholder="Paste your script here..."></textarea>
        <div class="row" style="margin-top:14px">
          <input type="file" id="voice" accept="audio/*">
        </div>
      </div>

      <div class="card">
        <h2>Beat Settings</h2>
        <div class="controls">
          <label class="field">
            <span>Target beat seconds</span>
            <input type="number" id="sdur" value="3" min="2" max="12" step="0.5">
          </label>
          <div class="checks">
            <label class="check"><input type="checkbox" id="cards" checked> statement cards</label>
            <label class="check"><input type="checkbox" id="sfx" checked> sound design</label>
            <label class="check"><input type="checkbox" id="vision" checked> vision ranking</label>
            <label class="check"><input type="checkbox" id="motion" checked> camera motion</label>
          </div>
          <div class="row">
            <button class="primary" onclick="analyze()" id="btnA">Generate matches</button>
            <button class="ghost" onclick="render()" id="btnR">Render video</button>
            <button class="ghost" onclick="stopRender()">Stop</button>
          </div>
          <div class="progress"><i id="bar"></i></div>
          <div class="sub" id="astat">Ready for a script.</div>
          <div class="sub" id="rstat">No render running.</div>
        </div>
      </div>

      <div class="card">
        <h2>Outputs</h2>
        <div id="outs" class="sub">No exports yet.</div>
      </div>
    </div>

    <div class="stack">
      <div class="card">
        <h2>Beat Plan</h2>
        <div class="summary">
          <div class="metric"><div class="label">Beats</div><div class="value" id="beatCount">0</div></div>
          <div class="metric"><div class="label">Timeline</div><div class="value" id="beatDuration">0.0s</div></div>
          <div class="metric"><div class="label">Confident Matches</div><div class="value" id="beatConf">0</div></div>
        </div>
        <div id="warnings"></div>
        <div id="scenes" class="empty-plan">Run Generate matches to preview the whole script beat-by-beat.</div>
      </div>
    </div>
  </div>

  <div class="footer">Everything runs locally. Stock search uses Pexels and Pixabay. Alignment uses Whisper when voiceover is provided.</div>
</div>

<script>
function formatTime(value){
  return (Number(value)||0).toFixed(1)+'s';
}

function escapeHtml(value){
  return String(value)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

function useCandidate(sceneNo, query){
  const input=document.querySelector(`input[data-scene="${sceneNo}"]`);
  if(input){input.value=query;input.focus();}
}

function renderWarnings(items){
  const wrap=document.getElementById('warnings');
  wrap.innerHTML='';
  if(!items || !items.length){return;}
  items.forEach(item=>{
    const div=document.createElement('div');
    div.className='warning';
    div.textContent=item;
    wrap.appendChild(div);
  });
}

function renderPlan(scenes){
  const el=document.getElementById('scenes');
  const confident=scenes.filter(s=>typeof s.confidence==='number' && s.confidence >= 0.24).length;
  document.getElementById('beatCount').textContent=String(scenes.length);
  document.getElementById('beatDuration').textContent=scenes.length?formatTime(scenes[scenes.length-1].end):'0.0s';
  document.getElementById('beatConf').textContent=String(confident);

  if(!scenes.length){
    el.className='empty-plan';
    el.textContent='No beats generated.';
    return;
  }

  el.className='';
  el.innerHTML='';

  scenes.forEach(scene=>{
    const card=document.createElement('div');
    card.className='scene';
    const confidence=(typeof scene.confidence==='number') ? 'match '+scene.confidence.toFixed(2) : 'no score';
    const query=(scene.keywords && scene.keywords[0]) ? scene.keywords[0] : '';
    const candidates=(scene.candidates||[]).map(candidate=>{
      const thumb=candidate.thumb
        ? `<div class="thumb" style="background-image:url('${escapeHtml(candidate.thumb)}')"></div>`
        : `<div class="thumb empty">No preview</div>`;
      const score=(typeof candidate.sim==='number') ? candidate.sim.toFixed(2) : 'text';
      const desc=escapeHtml(candidate.desc || candidate.query || 'Candidate visual');
      const queryText=JSON.stringify(candidate.query || query);
      return `
        <div class="cand">
          ${thumb}
          <div class="cand-body">
            <div class="cand-title">${desc.slice(0,82)}</div>
            <div class="cand-meta">
              <span>${escapeHtml(candidate.source || 'stock')}</span>
              <span>${score}</span>
            </div>
            <button class="ghost" type="button" onclick='useCandidate(${scene.scene}, ${queryText})'>Use query</button>
          </div>
        </div>
      `;
    }).join('');

    card.innerHTML = `
      <div class="scene-top">
        <div class="scene-id">
          <div class="scene-no">${scene.scene}</div>
          <div>
            <div style="font-weight:700">Beat ${scene.scene}</div>
            <div class="scene-meta">
              <span class="chip">${formatTime(scene.start)} - ${formatTime(scene.end)}</span>
              <span class="chip">${formatTime(scene.end - scene.start)}</span>
              <span class="chip">${confidence}</span>
            </div>
          </div>
        </div>
      </div>
      <div class="scene-text">${escapeHtml(scene.text || '')}</div>
      <div class="query-label">Visual search phrase</div>
      <input class="query-input" value="${escapeHtml(query)}" data-scene="${scene.scene}">
      <div class="cand-grid">${candidates || '<div class="chip">No preview matches found for this beat yet.</div>'}</div>
    `;
    el.appendChild(card);
  });
}

async function analyze(){
  const script=document.getElementById('script').value.trim();
  const voice=document.getElementById('voice').files[0];
  const duration=document.getElementById('sdur').value;
  if(!script){alert('Paste your script first.');return;}
  const fd=new FormData();
  fd.append('script', script);
  fd.append('scene_duration', duration);
  if(voice){fd.append('voice', voice);}
  document.getElementById('btnA').disabled=true;
  document.getElementById('astat').textContent=voice
    ? 'Analyzing script, aligning voiceover, and finding preview visuals...'
    : 'Analyzing script and finding preview visuals...';
  renderWarnings([]);
  const response=await fetch('/api/analyze', {method:'POST', body:fd});
  const data=await response.json();
  document.getElementById('btnA').disabled=false;
  if(data.error){
    document.getElementById('astat').textContent='Error: '+data.error;
    return;
  }
  document.getElementById('astat').textContent =
    data.scenes.length+' beats across '+data.duration.toFixed(1)+'s.';
  renderWarnings(data.warnings || []);
  renderPlan(data.scenes || []);
}

async function savePlan(){
  const inputs=[...document.querySelectorAll('.query-input')];
  if(!inputs.length){return;}
  const edits={};
  inputs.forEach(input=>{edits[input.dataset.scene]=input.value;});
  await fetch('/api/plan', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(edits)
  });
}

async function render(){
  await savePlan();
  const opts={
    scene_duration:+document.getElementById('sdur').value,
    cards:document.getElementById('cards').checked,
    sfx:document.getElementById('sfx').checked,
    vision:document.getElementById('vision').checked,
    motion:document.getElementById('motion').checked
  };
  const response=await fetch('/api/render', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(opts)
  });
  const data=await response.json();
  if(data.error){alert(data.error);}
}

async function stopRender(){
  await fetch('/api/stop', {method:'POST'});
}

async function poll(){
  const progressResp=await fetch('/api/progress');
  const progress=await progressResp.json();
  const badge=document.getElementById('state');
  badge.textContent=progress.state;
  badge.className='badge'+(progress.state==='rendering' ? ' live' : '');

  if(progress.total){
    document.getElementById('bar').style.width=(100*progress.done/progress.total)+'%';
    document.getElementById('rstat').textContent='Rendered '+progress.done+' of '+progress.total+' beats.';
  }else{
    document.getElementById('bar').style.width='0%';
    document.getElementById('rstat').textContent='No render running.';
  }
  if(progress.state==='done'){
    document.getElementById('rstat').textContent='Render finished.';
  }else if(progress.state==='error'){
    document.getElementById('rstat').textContent='Render failed. Check output/render.log for details.';
  }

  const outputsResp=await fetch('/api/outputs');
  const outputs=await outputsResp.json();
  const outWrap=document.getElementById('outs');
  outWrap.innerHTML=outputs.length ? '' : 'No exports yet.';
  outputs.forEach(file=>{
    const div=document.createElement('div');
    div.className='out';
    div.innerHTML=`<span>${escapeHtml(file.name)} - ${file.mb} MB</span><a href="/outputs/${encodeURIComponent(file.name)}" download>Download</a>`;
    outWrap.appendChild(div);
  });
}

setInterval(poll, 2500);
poll();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/api/analyze")
async def analyze(script: str = Form(...),
                  scene_duration: float = Form(DEFAULT_BEAT_SECONDS),
                  voice: UploadFile = File(None)):
    UPLOADS.mkdir(exist_ok=True)
    SCRIPT.write_text(script, encoding="utf-8")
    voice_arg = []
    if voice is not None:
        VOICE.write_bytes(await voice.read())
        VOICE.with_suffix(".words.json").unlink(missing_ok=True)
        voice_arg = ["--voiceover", str(VOICE)]
    else:
        VOICE.unlink(missing_ok=True)
        VOICE.with_suffix(".words.json").unlink(missing_ok=True)
    PLAN.unlink(missing_ok=True)
    cmd = [sys.executable, "main.py", str(SCRIPT), *voice_arg,
           "--scene-duration", str(scene_duration), "--export-plan", str(PLAN)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if not PLAN.exists():
        return JSONResponse({"error": proc.stdout[-400:] + proc.stderr[-400:]})
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    warnings: list[str] = []
    preview_used_ids: set[str] = set()
    enriched = []
    for index, entry in enumerate(plan):
        scene = Scene(entry["text"], entry["start"], entry["end"], list(entry["keywords"]))
        try:
            candidates = find_candidates_for_scene(scene, index, preview_used_ids, limit=3)
        except ClipSearchError as exc:
            candidates = []
            warnings.append(str(exc).splitlines()[0])
        if candidates:
            preview_used_ids.add(candidates[0].key)
        enriched.append({
            **entry,
            "confidence": round(candidates[0].sim, 3) if candidates and candidates[0].sim >= 0 else None,
            "candidates": [
                {
                    "source": c.source,
                    "video_id": c.video_id,
                    "query": c.query,
                    "desc": c.desc,
                    "sim": round(c.sim, 3) if c.sim >= 0 else None,
                    "thumb": c.thumb,
                }
                for c in candidates
            ],
        })

    return {
        "scenes": enriched,
        "duration": enriched[-1]["end"] if enriched else 0.0,
        "warnings": warnings[:6],
    }


@app.post("/api/plan")
async def save_plan(edits: dict):
    if PLAN.exists():
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        for s in plan:
            key = str(s["scene"])
            if key in edits and edits[key].strip():
                s["keywords"] = [edits[key].strip()] + list(s["keywords"][1:])
        PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return {"ok": True}


@app.post("/api/render")
async def render(opts: dict):
    global _render_proc, _expected_scenes
    if _render_proc is not None and _render_proc.poll() is None:
        return JSONResponse({"error": "A render is already running"})
    if not SCRIPT.exists():
        return JSONResponse({"error": "Run Analyze first"})
    _expected_scenes = len(json.loads(PLAN.read_text(encoding="utf-8"))) if PLAN.exists() else 0
    cmd = [sys.executable, "main.py", str(SCRIPT),
           "--scene-duration", str(opts.get("scene_duration", 6))]
    if VOICE.exists():
        cmd += ["--voiceover", str(VOICE)]
    if PLAN.exists():
        cmd += ["--use-plan", str(PLAN)]
    if not opts.get("cards", True):
        cmd.append("--no-cards")
    if not opts.get("sfx", True):
        cmd.append("--no-sfx")
    if not opts.get("vision", True):
        cmd.append("--no-vision")
    if not opts.get("motion", True):
        cmd.append("--no-motion")
    log = (ROOT / "output"); log.mkdir(exist_ok=True)
    _render_proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=open(log / "render.log", "w"),
        stderr=subprocess.STDOUT, text=True)
    return {"ok": True}


@app.post("/api/stop")
async def stop():
    global _render_proc
    if _render_proc is not None and _render_proc.poll() is None:
        _render_proc.terminate()
    return {"ok": True}


@app.get("/api/progress")
def progress():
    done = 0
    prog = TEMP / "progress.json"
    if prog.exists():
        try:
            done = len(json.loads(prog.read_text()).get("completed", {}))
        except (json.JSONDecodeError, OSError):
            pass
    if _render_proc is None:
        state = "idle"
    elif _render_proc.poll() is None:
        state = "rendering"
    else:
        state = "done" if _render_proc.returncode == 0 else "error"
    return {"state": state, "done": done, "total": _expected_scenes}


@app.get("/api/outputs")
def outputs():
    if not OUTPUT.exists():
        return []
    files = sorted(OUTPUT.glob("*.mp4"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return [{"name": f.name, "mb": f.stat().st_size // (1024 * 1024)}
            for f in files[:10]]


@app.get("/outputs/{name}")
def download(name: str):
    f = (OUTPUT / name).resolve()
    if f.parent != OUTPUT.resolve() or not f.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(f, media_type="video/mp4", filename=name)


if __name__ == "__main__":
    print("Script → Video Studio: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
