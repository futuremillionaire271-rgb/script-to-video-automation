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
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT = Path(__file__).parent
UPLOADS = ROOT / "my_scripts"
OUTPUT = ROOT / "output"
TEMP = ROOT / "temp"
PLAN = UPLOADS / "ui_plan.json"
SCRIPT = UPLOADS / "ui_script.txt"
VOICE = UPLOADS / "ui_voiceover.mp3"

app = FastAPI(title="Script to Video Studio")
_render_proc: subprocess.Popen | None = None
_expected_scenes: int = 0


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Script → Video Studio</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0b0e14;--panel:#12161f;--panel2:#171c27;--line:#232a38;
--text:#e8ecf3;--dim:#8b94a7;--gold:#ffd147;--gold2:#b8912b;--ok:#3ecf8e;--err:#ff6b6b}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:15px/1.6 -apple-system,'Segoe UI',Roboto,sans-serif;min-height:100vh}
.wrap{max-width:1100px;margin:0 auto;padding:32px 20px 80px}
header{display:flex;align-items:center;gap:14px;margin-bottom:28px}
.logo{width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,var(--gold),var(--gold2));
display:flex;align-items:center;justify-content:center;font-size:22px}
h1{font-size:22px;font-weight:700}h1 span{color:var(--gold)}
.sub{color:var(--dim);font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px}
.card h2{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin-bottom:14px}
textarea{width:100%;min-height:180px;background:var(--panel2);border:1px solid var(--line);border-radius:10px;
color:var(--text);padding:12px;font:13px/1.5 ui-monospace,monospace;resize:vertical}
input[type=file]{color:var(--dim);font-size:13px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:12px}
label.opt{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--dim);cursor:pointer}
input[type=number]{width:70px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;color:var(--text);padding:6px 8px}
button{background:linear-gradient(135deg,var(--gold),var(--gold2));color:#141414;font-weight:700;border:0;
border-radius:10px;padding:11px 22px;font-size:14px;cursor:pointer;transition:transform .1s}
button:hover{transform:translateY(-1px)}
button.ghost{background:var(--panel2);color:var(--text);border:1px solid var(--line)}
button:disabled{opacity:.45;cursor:not-allowed;transform:none}
.progress{height:10px;background:var(--panel2);border-radius:99px;overflow:hidden;margin:14px 0 6px}
.progress i{display:block;height:100%;width:0%;border-radius:99px;background:linear-gradient(90deg,var(--gold),var(--gold2));transition:width .8s}
.stat{font-size:13px;color:var(--dim)}
#scenes{max-height:420px;overflow:auto;margin-top:8px}
.scene{display:grid;grid-template-columns:44px 1fr 1fr;gap:10px;padding:10px;border-bottom:1px solid var(--line);font-size:13px}
.scene .no{color:var(--gold);font-weight:700}
.scene .txt{color:var(--dim)}
.scene input{width:100%;background:var(--panel2);border:1px solid var(--line);border-radius:8px;color:var(--text);padding:6px 9px;font-size:12px}
.out{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--line);font-size:13px}
.out a{color:var(--gold);text-decoration:none;font-weight:600}
.badge{font-size:11px;padding:3px 10px;border-radius:99px;background:var(--panel2);border:1px solid var(--line);color:var(--dim)}
.badge.live{color:var(--ok);border-color:var(--ok)}
.footer{margin-top:26px;color:var(--dim);font-size:12px;text-align:center}
</style></head><body><div class="wrap">
<header><div class="logo">🎬</div><div>
<h1>Script → Video <span>Studio</span></h1>
<div class="sub">voice-locked captions · vision-verified footage · retention sound design</div>
</div><div style="margin-left:auto" id="state" class="badge">idle</div></header>

<div class="grid">
<div class="card"><h2>1 · Script & Voiceover</h2>
<textarea id="script" placeholder="Paste your narration script here..."></textarea>
<div class="row"><input type="file" id="voice" accept="audio/*">
<button class="ghost" onclick="analyze()" id="btnA">Analyze scenes</button></div>
<div class="stat" id="astat"></div></div>

<div class="card"><h2>2 · Render settings</h2>
<div class="row">
<label class="opt">Scene sec <input type="number" id="sdur" value="6" min="3" max="12"></label>
<label class="opt"><input type="checkbox" id="cards" checked> statement cards</label>
<label class="opt"><input type="checkbox" id="sfx" checked> sound design</label>
<label class="opt"><input type="checkbox" id="vision" checked> vision check</label>
<label class="opt"><input type="checkbox" id="motion" checked> motion</label>
</div>
<div class="row"><button onclick="render()" id="btnR">Render video</button>
<button class="ghost" onclick="stopRender()">Stop</button></div>
<div class="progress"><i id="bar"></i></div>
<div class="stat" id="rstat">no render running</div></div>
</div>

<div class="card" style="margin-top:20px"><h2>3 · Scene plan (edit search phrases, then re-render)</h2>
<div id="scenes" class="stat">Run Analyze to see the voice-aligned scene plan.</div></div>

<div class="card" style="margin-top:20px"><h2>4 · Outputs</h2><div id="outs" class="stat">none yet</div></div>
<div class="footer">Everything runs locally · Pexels + Pixabay stock · Whisper alignment · CLIP verification</div>
</div>
<script>
async function analyze(){
  const s=document.getElementById('script').value.trim();
  const f=document.getElementById('voice').files[0];
  if(!s){alert('Paste your script first');return}
  const fd=new FormData();fd.append('script',s);if(f)fd.append('voice',f);
  document.getElementById('btnA').disabled=true;
  document.getElementById('astat').textContent=f?'Analyzing + aligning to voiceover (first time ~4 min for Whisper)...':'Analyzing...';
  const r=await fetch('/api/analyze',{method:'POST',body:fd});const d=await r.json();
  document.getElementById('btnA').disabled=false;
  if(d.error){document.getElementById('astat').textContent='Error: '+d.error;return}
  document.getElementById('astat').textContent=d.scenes.length+' scenes · '+d.duration.toFixed(1)+'s';
  renderPlan(d.scenes);
}
function renderPlan(sc){
  const el=document.getElementById('scenes');el.innerHTML='';
  sc.forEach(s=>{const div=document.createElement('div');div.className='scene';
    div.innerHTML=`<div class="no">${s.scene}</div><div class="txt">${s.text}</div>
    <div><input value="${(s.keywords[0]||'').replace(/"/g,'&quot;')}" data-scene="${s.scene}"></div>`;
    el.appendChild(div);});
}
async function savePlan(){
  const inputs=[...document.querySelectorAll('#scenes input')];
  if(!inputs.length)return;
  const edits={};inputs.forEach(i=>edits[i.dataset.scene]=i.value);
  await fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(edits)});
}
async function render(){
  await savePlan();
  const opts={scene_duration:+document.getElementById('sdur').value,
    cards:document.getElementById('cards').checked,sfx:document.getElementById('sfx').checked,
    vision:document.getElementById('vision').checked,motion:document.getElementById('motion').checked};
  const r=await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(opts)});
  const d=await r.json();if(d.error)alert(d.error);
}
async function stopRender(){await fetch('/api/stop',{method:'POST'})}
async function poll(){
  const r=await fetch('/api/progress');const d=await r.json();
  document.getElementById('state').textContent=d.state;
  document.getElementById('state').className='badge'+(d.state==='rendering'?' live':'');
  if(d.total){document.getElementById('bar').style.width=(100*d.done/d.total)+'%';
    document.getElementById('rstat').textContent=`scene ${d.done}/${d.total}`+(d.state==='done'?' · finished!':'');}
  const o=await fetch('/api/outputs');const outs=await o.json();
  const el=document.getElementById('outs');
  el.innerHTML=outs.length?'':'none yet';
  outs.forEach(f=>{const div=document.createElement('div');div.className='out';
    div.innerHTML=`<span>${f.name} · ${f.mb} MB</span><a href="/outputs/${f.name}" download>download</a>`;
    el.appendChild(div);});
}
setInterval(poll,2500);poll();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/api/analyze")
async def analyze(script: str = Form(...), voice: UploadFile = File(None)):
    UPLOADS.mkdir(exist_ok=True)
    SCRIPT.write_text(script)
    voice_arg = []
    if voice is not None:
        VOICE.write_bytes(await voice.read())
        VOICE.with_suffix(".words.json").unlink(missing_ok=True)
        voice_arg = ["--voiceover", str(VOICE)]
    cmd = [sys.executable, "main.py", str(SCRIPT), *voice_arg,
           "--scene-duration", "6", "--export-plan", str(PLAN)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if not PLAN.exists():
        return JSONResponse({"error": proc.stdout[-400:] + proc.stderr[-400:]})
    plan = json.loads(PLAN.read_text())
    return {"scenes": plan, "duration": plan[-1]["end"] if plan else 0.0}


@app.post("/api/plan")
async def save_plan(edits: dict):
    if PLAN.exists():
        plan = json.loads(PLAN.read_text())
        for s in plan:
            key = str(s["scene"])
            if key in edits and edits[key].strip():
                s["keywords"] = [edits[key].strip()] + list(s["keywords"][1:])
        PLAN.write_text(json.dumps(plan, indent=2))
    return {"ok": True}


@app.post("/api/render")
async def render(opts: dict):
    global _render_proc, _expected_scenes
    if _render_proc is not None and _render_proc.poll() is None:
        return JSONResponse({"error": "A render is already running"})
    if not SCRIPT.exists():
        return JSONResponse({"error": "Run Analyze first"})
    _expected_scenes = len(json.loads(PLAN.read_text())) if PLAN.exists() else 0
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
