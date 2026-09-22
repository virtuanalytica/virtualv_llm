#!/usr/bin/env python3
"""Disk-bounded Qwen3.8 Flash-Next vLLM benchmark cascade.

One quant is downloaded, smoke-tested and fully evaluated at a time.  It is
never deleted before its source/revision, probe and completed benchmark rows
are durable.  The all-four profile is intentionally separate from V100-only.
"""
from __future__ import annotations

import argparse, csv, io, json, os, shutil, signal, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path('/media/knight2/EDS2/models/llm')
REPORTS = ROOT / 'reports'
REPORT = REPORTS / 'well_known_suite_20260917.json'
STATE = REPORTS / 'qwen38_flash_next_vllm_cascade_20260922.json'
WKS = ROOT / 'scripts/benchmarks/well_known_suite.py'
RENDER = ROOT / 'scripts/benchmarks/build_dual_v100_html.py'
SERVE = ROOT / 'infra/model_serve_configs/serve_qwen38_flash_next_w4a16.sh'
SERVICES = ('local-chat-qwen38.service', 'llama-qwen.service')
RESERVE = 24_000_000_000
CANDIDATES = [
    {'key': 'merlin-w4a16', 'repo': 'halt95/Qwen3.8-Flash-Next-W4A16-Merlin',
     'revision': '483ee0015419568c7014b011db0c08b6ddcb2ceb',
     'published_at': '2026-09-19T09:43:31Z', 'bytes': 123_683_455_834},
    {'key': 'awq-w4a16', 'repo': 'wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16',
     'revision': '0939125b929543a783ce700c90e36dd1a575c00c',
     'published_at': '2026-08-27T15:59:21Z', 'bytes': 180_725_866_024},
]
PROFILES = (('v100', '1,2', '2x Tesla V100-SXM2-32GB + NVLink', 112),
            ('allfour', '0,1,2,3', 'RTX A4000 + 2x V100 NVLink + RTX 4000 Ada', 72))

def now(): return datetime.now(timezone.utc).isoformat()
def load(p, default): return json.loads(p.read_text()) if p.exists() else default
def atomic(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n'); t.replace(p)
def record(state, candidate, event, **data):
    state.setdefault('events', []).append({'at': now(), 'quant': candidate['key'], 'event': event, **data}); atomic(STATE, state)
def model_dir(c): return MODELS / ('qwen38-flash-next-' + c['key'])
def profile_model(c, suffix): return f"qwen38-flash-next-{c['key']}-{suffix}"

def download(c, state):
    target = model_dir(c)
    if (target / 'config.json').exists() and sum(x.stat().st_size for x in target.rglob('*') if x.is_file()) >= c['bytes'] * .98:
        return target
    free = shutil.disk_usage(MODELS).free
    if free < c['bytes'] + RESERVE: raise RuntimeError(f'insufficient disk: {free} < {c["bytes"] + RESERVE}')
    record(state, c, 'download_started', repo=c['repo'], revision=c['revision'], free_before=free)
    subprocess.run(['hf', 'download', c['repo'], '--revision', c['revision'], '--local-dir', str(target)], check=True)
    got = sum(x.stat().st_size for x in target.rglob('*') if x.is_file())
    if got < c['bytes'] * .98: raise RuntimeError(f'incomplete download: {got} bytes')
    record(state, c, 'download_verified', bytes=got)
    return target

def services_stop():
    active=[]
    for service in SERVICES:
        if subprocess.run(['systemctl','--user','is-active','--quiet',service]).returncode == 0:
            subprocess.run(['systemctl','--user','stop',service],check=True); active.append(service)
    return active
def services_restore(active):
    for service in active: subprocess.run(['systemctl','--user','start',service],check=False)
def assert_exclusive_gpus(devices):
    """Refuse an all-GPU measurement while another compute context remains.

    `nvidia-smi` can retain a small display/driver allocation.  That is not a
    competing inference workload; a listed compute application is.  Capture
    both facts so the dashboard evidence distinguishes them.
    """
    target = {int(i) for i in devices.split(',')}
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    gpu_map = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    inventory = {}
    for row in csv.reader(io.StringIO(gpu_map.stdout)):
        if len(row) >= 4: inventory[row[1].strip()] = {
            'index': int(row[0]), 'used_mib': int(row[2]), 'free_mib': int(row[3])}
    occupants = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) >= 4 and row[0].strip() in inventory:
            info = inventory[row[0].strip()]
            if info['index'] in target:
                occupants.append({'gpu': info['index'], 'pid': int(row[1]),
                                  'process': row[2].strip(), 'used_mib': int(row[3])})
    if occupants:
        raise RuntimeError(f'external GPU compute contexts remain: {occupants}')
    return {str(v['index']): v for v in inventory.values() if v['index'] in target}
def wait_ready(port, proc, log):
    deadline=time.time()+900
    while time.time()<deadline:
        if proc.poll() is not None: raise RuntimeError(f'vLLM exited {proc.returncode}; see {log}')
        try:
            with urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=3) as r:
                if r.status == 200: return
        except Exception: pass
        time.sleep(3)
    raise RuntimeError(f'vLLM readiness timeout; see {log}')
def probe(port):
    body=json.dumps({'model':'qwen38-flash-next','messages':[{'role':'user','content':'Answer only: what is 2+2?'}], 'max_tokens':256, 'temperature':0}).encode()
    t=time.monotonic()
    with urlopen(Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=body,headers={'Content-Type':'application/json'}),timeout=900) as r: data=json.load(r)
    n=data.get('usage',{}).get('completion_tokens',0); seconds=time.monotonic()-t
    if not n: raise RuntimeError('smoke response contained no completion tokens')
    return {'completion_tokens':n,'seconds':round(seconds,3),'tokens_per_second':round(n/seconds,3)}
def inject_error(model, c, suffix, topology, error):
    payload=load(REPORT, {'results':[]}); rows=payload.setdefault('results',[])
    row={'model':model,'error':error,'engine':'1Cat-vLLM 1.5.0','topology':topology,
         'hardware_profile':suffix,'source_repo':c['repo'],'model_source':f'https://huggingface.co/{c["repo"]}',
         'source_revision':c['revision'],'source_published_at':c['published_at'],'quantization':c['key']}
    payload['results']=[r for r in rows if r.get('model') != model]+[row]; atomic(REPORT,payload)
def annotate(model,c,suffix,topology,smoke):
    payload=load(REPORT, {'results':[]})
    for r in payload.get('results',[]):
        if r.get('model')==model: r.update({'engine':'1Cat-vLLM 1.5.0','topology':topology,'hardware_profile':suffix,'source_repo':c['repo'],'model_source':f'https://huggingface.co/{c["repo"]}','source_revision':c['revision'],'source_published_at':c['published_at'],'quantization':c['key'],'weight_bytes':c['bytes'],'smoke_probe':smoke})
    atomic(REPORT,payload)
def run_profile(c,state,suffix,devices,topology,offload):
    model=profile_model(c,suffix); port=18021 if suffix=='v100' else 18022; active=[]; proc=None
    log=REPORTS/'vllm_logs'/f'{model}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    try:
        if suffix=='allfour':
            active=services_stop()
            # llama.cpp's idle/sleep teardown is asynchronous; do not race it.
            deadline=time.time()+90
            while True:
                try:
                    idle_snapshot=assert_exclusive_gpus(devices)
                    break
                except RuntimeError:
                    if time.time() >= deadline: raise
                    time.sleep(2)
        else:
            idle_snapshot=assert_exclusive_gpus(devices)
        record(state,c,'profile_started',model=model,devices=devices,
               preflight_gpu_memory=idle_snapshot)
        with log.open('w') as out:
            proc=subprocess.Popen([str(SERVE),devices,str(len(devices.split(','))),str(model_dir(c)),str(port),str(offload)],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        wait_ready(port,proc,log); smoke=probe(port); record(state,c,'smoke_complete',model=model,**smoke)
        if smoke['tokens_per_second'] < 5: raise RuntimeError(f'smoke below 5 t/s: {smoke}')
        subprocess.run(['python3',str(WKS),model,'--external-url',f'http://127.0.0.1:{port}','--external-model','qwen38-flash-next','--physical-gpus',devices,'--topology',topology,'--engine','1Cat-vLLM 1.5.0','--out',str(REPORT)],cwd=ROOT,check=True)
        annotate(model,c,suffix,topology,smoke); record(state,c,'profile_complete',model=model); subprocess.run(['python3',str(RENDER)],cwd=ROOT,check=True)
    except Exception as exc:
        inject_error(model,c,suffix,topology,f'{type(exc).__name__}: {exc}'); record(state,c,'profile_failed',model=model,error=str(exc)); subprocess.run(['python3',str(RENDER)],cwd=ROOT,check=False)
    finally:
        if proc and proc.poll() is None: os.killpg(proc.pid,signal.SIGTERM); proc.wait(timeout=45)
        services_restore(active)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--only',choices=[x['key'] for x in CANDIDATES]); p.add_argument('--keep-weights',action='store_true'); a=p.parse_args()
    state=load(STATE,{'candidates':CANDIDATES,'events':[]})
    for c in CANDIDATES:
        if a.only and c['key']!=a.only: continue
        download(c,state)
        for profile in PROFILES: run_profile(c,state,*profile)
        if not a.keep_weights:
            d=model_dir(c); size=sum(x.stat().st_size for x in d.rglob('*') if x.is_file()); shutil.rmtree(d); record(state,c,'pruned',bytes_freed=size)
    subprocess.run(['python3',str(RENDER)],cwd=ROOT,check=False)
if __name__=='__main__': main()
