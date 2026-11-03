"""Recreate the per-run spread of material error s1 for every supervised and physics-only run recorded for the main arms.
Reads each value through Research OS (`research --json show RUN-...`); stdlib only; writes a deterministic SVG."""
import json, math, subprocess, sys

root, out = sys.argv[1], sys.argv[2]


def s1(run):
    d = json.loads(subprocess.run([sys.executable, "-m", "research_cli", "--root", root, "--json", "show", run],
                                  capture_output=True, text=True, check=True).stdout)
    return d["metrics"]["s1"]["value"]


GROUPS = [  # (label lines, runs, colour)
    (("Supervised,", "reference", "recipe"), [41, 42, 43], "#2e6da4"),
    (("Supervised,", "higher", "learning rate"), [38, 39, 40, 44, 45, 46], "#2e6da4"),
    (("Label-free,", "baseline", "output"), [47, 48, 49], "#c0392b"),
    (("Label-free,", "unbounded", "linear output"), [70, 71, 72], "#c0392b"),
    (("Label-free,", "occupancy", "softmax"), [64, 65, 66], "#c0392b"),
    (("Label-free,", "3-way", "softmax"), [67, 68, 69], "#c0392b"),
    (("Label-free,", "sum penalty", "w=1.6e-5"), [74, 75, 76], "#c0392b"),
    (("Label-free,", "sum penalty", "w=1.6e-4"), [77, 78, 79], "#c0392b"),
    (("Label-free,", "softmax, true", "occupancy given"), [83, 84, 85], "#c0392b"),
]
zero = json.loads(subprocess.run([sys.executable, "-m", "research_cli", "--root", root, "--json", "show", "RUN-0037"],
                                 capture_output=True, text=True, check=True).stdout)["metrics"]["s1_zero"]["value"]

W, H, L, R, T, B = 860, 470, 70, 20, 44, 96
ylo, yhi = math.log10(0.01), math.log10(1.0)
Y = lambda v: H - B - (math.log10(v) - ylo) / (yhi - ylo) * (H - B - T)
cw = (W - L - R) / len(GROUPS)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
     f'<text x="{L}" y="22" font-size="14" font-weight="bold">Material error s1 of every run in the main arms (100 validation cases; one point per run)</text>']
for v in (0.01, 0.03, 0.1, 0.3, 1.0):
    s.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" stroke="#e3e3e3"/>'
             f'<text x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end">{v:g}</text>')
s.append(f'<line x1="{L}" y1="{Y(zero):.1f}" x2="{W-R}" y2="{Y(zero):.1f}" stroke="#777" stroke-dasharray="5 4"/>'
         f'<text x="{W-R-4}" y="{Y(zero)-5:.1f}" text-anchor="end" fill="#555">predicting zero ({zero:.3f})</text>')
s.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/><line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/>')
for i, (lab, runs, col) in enumerate(GROUPS):
    cx = L + cw * (i + 0.5)
    vals = [s1(f"RUN-{r:04d}") for r in runs]
    n = len(vals)
    for j, v in enumerate(sorted(vals)):
        dx = (j - (n - 1) / 2) * (14 if n <= 3 else 11)
        s.append(f'<circle cx="{cx+dx:.1f}" cy="{Y(v):.1f}" r="5" fill="{col}" fill-opacity="0.85"/>')
    for k, t in enumerate(lab):
        s.append(f'<text x="{cx:.1f}" y="{H-B+18+14*k}" text-anchor="middle">{t}</text>')
    s.append(f'<text x="{cx:.1f}" y="{H-B+18+14*3+4}" text-anchor="middle" fill="#555" font-size="11">{n} runs</text>')
s.append(f'<text transform="translate(16 {(T+H-B)/2:.0f}) rotate(-90)" text-anchor="middle">Material error s1 (lower = better; log scale)</text>')
s.append('<text x="76" y="56" fill="#2e6da4">blue: trained with labels</text><text x="260" y="56" fill="#c0392b">red: no labels</text></svg>')
open(out, "w").write("\n".join(s) + "\n")
