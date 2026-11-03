"""Recreate the estimator comparison on the real kit from the recorded RUN-0009 metrics (no simulation, stdlib only).
Reads values through Research OS (`research --json show RUN-0009`); writes figures/real-kit-rmse.svg deterministically."""
import json, subprocess, sys, os

root = sys.argv[1]
out = sys.argv[2]
d = json.loads(subprocess.run([sys.executable, "-m", "research_cli", "--root", root, "--json", "show", "RUN-0009"],
                              capture_output=True, text=True, check=True).stdout)
m = d["metrics"]
rows = [("Unconstrained", "real_rmse_unconstrained"), ("Simplex-constrained", "real_rmse_simplex"),
        ("Sum-to-one, clipped", "real_rmse_sum_to_one_clipped"), ("Predict zero", "real_rmse_zero"),
        ("Sum-to-one (unclipped)", "real_rmse_sum_to_one")]
vals = [(lab, m[k]["value"]) for lab, k in rows]
W, H, L, T, B = 640, 330, 190, 40, 50
vmax = 1.8
sc = (W - L - 30) / vmax
bar = 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="13">',
       f'<text x="{L}" y="22" font-size="14" font-weight="bold">Volume-fraction RMSE on official benchmark images (3 cases)</text>']
for i, (lab, v) in enumerate(vals):
    y = T + i * (bar + 12)
    color = "#c0392b" if v > 1 else "#2e6da4"
    svg.append(f'<text x="{L-8}" y="{y+22}" text-anchor="end">{lab}</text>')
    svg.append(f'<rect x="{L}" y="{y}" width="{v*sc:.1f}" height="{bar}" fill="{color}"/>')
    svg.append(f'<text x="{L+v*sc+6:.1f}" y="{y+22}">{v:.3f}</text>')
ax = T + len(vals) * (bar + 12)
svg.append(f'<line x1="{L}" y1="{T-6}" x2="{L}" y2="{ax}" stroke="#333"/>')
svg.append(f'<line x1="{L}" y1="{ax}" x2="{W-20}" y2="{ax}" stroke="#333"/>')
for t in (0, 0.5, 1.0, 1.5):
    x = L + t * sc
    svg.append(f'<line x1="{x:.1f}" y1="{ax}" x2="{x:.1f}" y2="{ax+5}" stroke="#333"/><text x="{x:.1f}" y="{ax+20}" text-anchor="middle">{t}</text>')
svg.append(f'<text x="{(L+W)/2:.0f}" y="{H-8}" text-anchor="middle">RMSE of estimated volume fraction (dimensionless; lower is better)</text></svg>')
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").write("\n".join(svg) + "\n")
