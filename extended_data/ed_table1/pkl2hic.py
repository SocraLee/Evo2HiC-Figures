#!/usr/bin/env python
"""Convert a HiCFoundation task-3 output (HiCFoundation_enhanced.pkl) to .hic.

HiCFoundation's own converter breaks on DNA Zoo scaffold names: main_worker
splits keys on '_' (``HiC_scaffold_1`` -> ``HiC``) and array2hic prepends
``chr``.  We recover the real chromosome names from the ``input.pkl`` written
next to the output (same key order / matrix shapes) and call juicer ``pre``
with the chrom.size file directly.

Usage: pkl2hic.py <out_dir> <resolution> <chrom.size or genome_id> <output.hic>
"""
import os
import pickle
import subprocess
import sys

JAR = os.environ.get('JUICER_JAR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'juicer_tools.jar'))


def main(out_dir, resolution, genome, output_hic):
    resolution = int(resolution)
    pred = pickle.load(open(os.path.join(out_dir, 'HiCFoundation_enhanced.pkl'), 'rb'))
    inp = pickle.load(open(os.path.join(out_dir, 'input.pkl'), 'rb'))
    in_keys = list(inp.keys())
    # map mangled keys back to real names by shape (unique per chromosome in practice)
    shape2name = {}
    for k in in_keys:
        shape2name.setdefault(inp[k].shape, []).append(k)
    raw = output_hic + '.raw'
    n = 0
    with open(raw, 'w') as w:
        for k, m in pred.items():
            if k in inp:
                name = k
            else:
                cands = shape2name.get(m.shape, [])
                if len(cands) != 1:
                    raise RuntimeError(f'cannot map key {k!r} (shape {m.shape}) to input chrom: {cands}')
                name = cands[0]
            m = m.tocoo()
            m.eliminate_zeros()
            m.sum_duplicates()
            for r, c, v in zip(m.row, m.col, m.data):
                if c < r:
                    continue  # upper triangle only; juicer symmetrises intra-chrom
                w.write(f'0 {name} {int(r) * resolution + 1} 0 0 {name} {int(c) * resolution + 1} 1 {v:.3f}\n')
                n += 1
    print(f'[pkl2hic] wrote {n} records for {len(pred)} chromosome(s) -> {raw}', flush=True)
    cmd = ['java', '-Xmx32g', '-jar', JAR, 'pre', '-j', '8', '-d', '-r', str(resolution), raw, output_hic, genome]
    subprocess.run(cmd, check=True)
    os.remove(raw)
    print(f'[pkl2hic] done -> {output_hic}', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
