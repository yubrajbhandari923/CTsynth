"""Recreate the 'data fit vs material accuracy' scatter from recorded run metrics (no simulation, stdlib only).
Reads each value through Research OS (`research --json show RUN-…`); writes a deterministic SVG."""
import json, math, os, subprocess, sys

root, out = sys.argv[1], sys.argv[2]


def metrics(run):
    d = json.loads(subprocess.run([sys.executable, "-m", "research_cli", "--root", root, "--json", "show", run],
                                  capture_output=True, text=True, check=True).stdout)
    return {k: v["value"] for k, v in d["metrics"].items()}


pts = []  # (label, projection RMSE, s1, group)
m = metrics("RUN-0041"); pts.append(("Supervised network", m["proj_rmse_T"], m["s1"], "sup"))
m = metrics("RUN-0053"); pts.append(("Supervised + 200 physics steps", m["proj_rmse_K200"], m["s1_K200"], "sup"))
m = metrics("RUN-0047"); pts.append(("Physics-only network", m["proj_rmse_T"], m["s1"], "phys"))
m = metrics("RUN-0056"); pts.append(("Physics-only + entropy prior", m["proj_rmse_T"], m["s1"], "phys"))
m = metrics("RUN-0050"); pts.append(("Per-case iterative fit", m["proj_rmse_T_mean"], m["s1_mean"], "phys"))
m = metrics("RUN-0066"); pts.append(("Physics-only, one structured run", m["proj_rmse_T"], m["s1"], "phys"))
zero = metrics("RUN-0037")["s1_zero"]

W, H, L, R, T, B = 700, 430, 70, 20, 40, 60
xlo, xhi, ylo, yhi = math.log10(1e-4), math.log10(5e-3), math.log10(0.01), math.log10(1.0)
X = lambda v: L + (math.log10(v) - xlo) / (xhi - xlo) * (W - L - R)
Y = lambda v: H - B - (math.log10(v) - ylo) / (yhi - ylo) * (H - B - T)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
     f'<text x="{L}" y="22" font-size="14" font-weight="bold">Data fit versus material accuracy (100 validation cases)</text>']
for v in (1e-4, 3e-4, 1e-3, 3e-3):
    s.append(f'<line x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{H-B}" stroke="#e3e3e3"/>'
             f'<text x="{X(v):.1f}" y="{H-B+16}" text-anchor="middle">{v:g}</text>')
for v in (0.01, 0.03, 0.1, 0.3, 1.0):
    s.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" stroke="#e3e3e3"/>'
             f'<text x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end">{v:g}</text>')
s.append(f'<line x1="{L}" y1="{Y(zero):.1f}" x2="{W-R}" y2="{Y(zero):.1f}" stroke="#777" stroke-dasharray="5 4"/>'
         f'<text x="{W-R-4}" y="{Y(zero)-5:.1f}" text-anchor="end" fill="#555">predicting zero ({zero:.3f})</text>')
s.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/><line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/>')
off = {"Supervised network": (8, 16), "Supervised + 200 physics steps": (10, -10), "Physics-only network": (-8, 18),
       "Physics-only + entropy prior": (-8, -10), "Per-case iterative fit": (10, -10), "Physics-only, one structured run": (8, 4)}
for lab, x, y, g in pts:
    c = "#2e6da4" if g == "sup" else "#c0392b"
    dx, dy = off[lab]
    s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="6" fill="{c}"/>'
             f'<text x="{X(x)+dx:.1f}" y="{Y(y)+dy:.1f}" text-anchor="{"start" if dx > 0 else "end"}">{lab}</text>')
s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-14}" text-anchor="middle">Projection error: RMS error of predicted transmission (lower = better fit to the measurements)</text>')
s.append(f'<text transform="translate(16 {(T+H-B)/2:.0f}) rotate(-90)" text-anchor="middle">Material error s1 (lower = better)</text>')
s.append('<text x="76" y="56" fill="#2e6da4">blue: trained with labels</text><text x="76" y="72" fill="#c0392b">red: no labels</text></svg>')
open(out, "w").write("\n".join(s) + "\n")
