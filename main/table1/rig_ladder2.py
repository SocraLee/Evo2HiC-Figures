#!/usr/bin/env python
"""Evo2HiC-cg training protocol (fixed before training): every decision uses BN-RECALIBRATED chr8 validation.
For each checkpoint the trainer writes (every 1000 steps): copy it, BN-recalibrate it (50 training batches, batch 3,
seed 0), run chr8 validation on the recalibrated weights (--valid-only), and use that MSE for 'global best' and for
the plateau counter.  Phases: A 5e-4 (>= MIN_STEP, then patience) -> B 1e-4 from the global best (patience).
The reported model is the global best by recalibrated chr8 MSE, scored once on the 417 test windows.
Usage:
  rig_ladder2.py <cell> [--attach A <pgid>]            # attach to a running phase-A job (log logs/rig_<cell>_A.log)
  rig_ladder2.py <cell> --resume-A <ckpt> [--init-best <ckpt> <recal_ckpt> <mse> <where>]   # continue phase A (5e-4)
Env: EVAL_GPU (default 1) - GPU shared with training for recalibration / validation."""
import glob, json, os, re, shutil, signal, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths  # noqa: E402

W = str(paths.RA_WORK('orca'))       # outputs: checkpoints/, logs/, results/, tensors/
HERE = os.path.dirname(os.path.abspath(__file__))  # scripts
for _d in ('logs', 'results'): os.makedirs(f'{W}/{_d}', exist_ok=True)
CK = f'{W}/checkpoints'                          # rig_keep2/<cell>: copied, recalibrated and best checkpoints
RUNS = os.path.join(paths.REPO, 'checkpoints')   # run folders written by the trainer (config.save_dir)
ACC = {'H1ESC': '4DNFI2TK7L2F', 'HFF': '4DNFIPC7P27B'}   # cg / aux-peak targets are stored under the .hic accession
P = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS='8')
PATIENCE, MIN_STEP = 5, {'A': 10000}
EVAL_GPU = os.environ.get('EVAL_GPU', '1')
COMMON = ['--max-step', '300000', '--accumulate-step', '1', '--batch-size', '10', '--no-encoder-dropout',
          '--aux-weight', '0.05', '--mixed-precision', 'bf16', '--warmup-step', '1000']
EVAL_ARGS = ['--no-encoder-dropout', '--aux-weight', '0.05', '--batch-size', '3', '--seed', '0']
VAL = re.compile(r'Step: (\d+), \s*valid loss \(mse\) is ([\d.eE+-]+)')


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M")}] {msg}'
    print(line, flush=True)
    with open(f'{W}/results/rig_ladder_status.log', 'a') as f:
        f.write(line + '\n')


def recal_valid(cell, ck, tag, port):
    """BN-recalibrate `ck` and return (recal_path, chr8 MSE, chr8 PCC) on the recalibrated weights."""
    d = os.path.dirname(ck); rec = os.path.join(d, 'recal_' + os.path.basename(ck))
    if not os.path.exists(rec):
        subprocess.run(['bash', f'{HERE}/run_train_cg.sh', cell, EVAL_GPU, str(port)] + EVAL_ARGS +
                       ['--aux-peaks', ACC[cell], '--recalibrate-bn', ck, '--bn-recal-batches', '50', '--method-name', f'recal2-{tag}'],
                       stdout=open(f'{W}/logs/recal2_{cell}_{tag}.log', 'w'), stderr=subprocess.STDOUT)
    lg = f'{W}/logs/valid2_{cell}_{tag}.log'
    subprocess.run(['bash', f'{HERE}/run_train_cg.sh', cell, EVAL_GPU, str(port + 1)] + EVAL_ARGS +
                   ['--aux-peaks', ACC[cell], '--valid-only', rec, '--method-name', f'valid2-{tag}'],
                   stdout=open(lg, 'w'), stderr=subprocess.STDOUT)
    m = re.search(r'VALID-ONLY\t\S+\tstep=(-?\d+)\tmse=([\d.]+)\tpcc=([\d.]+)', open(lg).read())
    return (rec, float(m.group(2)), float(m.group(3))) if m else (rec, float('nan'), float('nan'))


def run_phase(cell, phase, lr, keep, state, attach_pgid=None, init_ckpt=None, attach_tag='rig'):
    tag = 'rig2' if attach_pgid is None else attach_tag
    name = f'CDNA2d-Seq2HiC-cg-{cell}-{tag}{phase}'
    logf = f'{W}/logs/{tag}_{cell}_{phase}.log'
    t_start = time.time() - 120 if attach_pgid is None else 0   # launch mode: only accept a run folder created after this
    steps_dir = f'{keep}/steps'; os.makedirs(steps_dir, exist_ok=True)
    if attach_pgid is None:
        args = ['bash', f'{HERE}/run_train_cg.sh', cell, '0,1,2,3', '29671'] + COMMON + ['--aux-peaks', ACC[cell], '--method-name', name]
        start = init_ckpt or (f'{keep}/best.pt' if phase != 'A' else None)
        if start is None:
            args += ['--base-learning-rate', str(lr)]
        else:
            rd = f'{keep}/resume_{phase}'; os.makedirs(rd, exist_ok=True)
            shutil.copy(start, f'{rd}/start.pt')
            a = os.path.join(os.path.dirname(start), 'args.json')
            shutil.copy(a if os.path.exists(a) else f'{keep}/args.json', f'{rd}/args.json')
            json.dump({}, open(f'{rd}/best.json', 'w'))         # trainer reads best.json next to the checkpoint
            args += ['--load-checkpoint', f'{rd}/start.pt', '--reset-lr', str(lr)]
        log(f'{cell} phase {phase} lr={lr} start from {start or "pretrained init"} (global best so far: {state["best"]})')
        proc = subprocess.Popen(args, stdout=open(logf, 'w'), stderr=subprocess.STDOUT, start_new_session=True)
        pgid = proc.pid
    else:
        pgid = attach_pgid; log(f'{cell} phase {phase}: attached to running job (pgid {pgid}); decisions on recalibrated chr8')

    def alive():
        try: os.killpg(pgid, 0); return True
        except ProcessLookupError: return False

    seen, bad, run_dir, port = set(), 0, None, 29800
    while True:
        if run_dir is None:
            ds = [d for d in sorted(glob.glob(f'{RUNS}/*_{name}_4000')) if os.stat(d).st_ctime >= t_start]
            run_dir = ds[-1] if ds else None
            if run_dir is None:
                time.sleep(30); continue
            log(f'{cell} phase {phase}: run folder {os.path.basename(run_dir)}')
        new = [(int(s), float(v)) for s, v in VAL.findall(open(logf, errors='ignore').read()) if int(s) not in seen]
        for step, raw in new:
            seen.add(step)
            src = f'{run_dir}/{step}.pt'
            if not os.path.exists(src):
                log(f'{cell} {phase} step {step}: checkpoint no longer on disk, skipped'); continue
            ck = f'{steps_dir}/{phase}_{step}.pt'; shutil.copy(src, ck)
            if not os.path.exists(f'{steps_dir}/args.json'): shutil.copy(f'{run_dir}/args.json', f'{steps_dir}/args.json')
            port += 2
            rec, mse, pcc = recal_valid(cell, ck, f'{phase}_{step}', port)
            state['curve'].append((cell, phase, lr, step, raw, mse, pcc))
            if mse < state['best']:
                shutil.copy(ck, f'{keep}/best.pt.tmp'); os.replace(f'{keep}/best.pt.tmp', f'{keep}/best.pt')
                shutil.copy(f'{run_dir}/args.json', f'{keep}/args.json')
                state.update(best=mse, best_where=f'{phase}:{step}', best_recal=rec); bad = 0
                log(f'{cell} {phase} step {step} chr8 MSE recal {mse:.6f} (raw {raw:.6f}), PCC {pcc:.4f}  NEW GLOBAL BEST')
            else:
                bad += 1
                log(f'{cell} {phase} step {step} chr8 MSE recal {mse:.6f} (raw {raw:.6f}), PCC {pcc:.4f}  no improvement {bad}/{PATIENCE} (best {state["best"]:.6f} @ {state["best_where"]})')
            state['bad'] = bad
            json.dump(state, open(f'{keep}/ladder2_state.json', 'w'), indent=1)
            if bad >= PATIENCE and step >= MIN_STEP.get(phase, 0):
                os.killpg(pgid, signal.SIGTERM); time.sleep(30)
                try: os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError: pass
                log(f'{cell} phase {phase} plateau (recalibrated) -> stopped'); return True
        if not new and not alive():
            log(f'{cell} phase {phase} EXITED unexpectedly; see {logf}'); return False
        time.sleep(30)


def final_eval(cell, state):
    tag = f'rig2_{cell}'; rec = state['best_recal']
    subprocess.run([P, f'{HERE}/build_tensors.py', cell, '0', rec, tag], env=ENV, stdout=open(f'{W}/logs/build_{tag}.log', 'w'), stderr=subprocess.STDOUT)
    out = subprocess.run([P, f'{HERE}/pcc_only.py', cell, tag], env=ENV, capture_output=True, text=True).stdout
    open(f'{W}/results/rig_test.tsv', 'a').write(out)
    subprocess.run([P, f'{HERE}/eval_cg.py', cell, tag], env=ENV, stdout=open(f'{W}/logs/eval_{tag}.log', 'w'), stderr=subprocess.STDOUT)
    log(f'{cell} FINAL (recalibrated chr8 best {state["best_where"]}, MSE {state["best"]:.6f}): {out.strip()}')


if __name__ == '__main__':
    argv = sys.argv[1:]; cell = argv[0]
    keep = f'{CK}/rig_keep2/{cell}'; os.makedirs(keep, exist_ok=True)
    state = {'best': float('inf'), 'best_where': None, 'best_recal': None, 'curve': []}
    attach = int(argv[argv.index('--attach') + 2]) if '--attach' in argv else None
    attach_tag = argv[argv.index('--attach-tag') + 1] if '--attach-tag' in argv else 'rig'
    init = argv[argv.index('--resume-A') + 1] if '--resume-A' in argv else None
    if '--init-best' in argv:                              # seed the global best with an evaluated checkpoint
        i = argv.index('--init-best'); ck, rec, mse, where = argv[i + 1:i + 5]
        shutil.copy(ck, f'{keep}/best.pt'); a = os.path.join(os.path.dirname(ck), 'args.json')
        if os.path.exists(a): shutil.copy(a, f'{keep}/args.json')
        state.update(best=float(mse), best_where=where, best_recal=rec)
        log(f'{cell} ladder2: global best seeded with {where} (recalibrated chr8 MSE {float(mse):.6f})')
    if attach and os.path.exists(f'{keep}/ladder2_state.json'):
        state = json.load(open(f'{keep}/ladder2_state.json'))
    ok = run_phase(cell, 'A', 5e-4, keep, state, attach_pgid=attach, init_ckpt=init, attach_tag=attach_tag)
    if ok:
        ok = run_phase(cell, 'B', 1e-4, keep, state)
    if ok:
        final_eval(cell, state)
    log(f'{cell} ladder2 done')
