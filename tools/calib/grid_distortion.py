"""Skreslenie objektívu z fotky mriežkovaného papiera (metóda rovných čiar, "plumb-line").

Čiary mriežky sú v skutočnosti rovné. Hľadá sa korekcia (stred skreslenia cx, cy a koeficienty
a1, a2), po ktorej sú rovné aj v obraze. Nezáleží na tom, či papier stojí presne kolmo, ani na tom,
že ide o dva hárky vedľa seba (každý úsek čiary sa narovnáva samostatne – zadaj --seam).

Model (rovnaký ako v aplikácii, len v pixeloch):  p_u = c + (p_d - c) * (1 + a1 ρ² + a2 ρ⁴),  ρ = |p_d - c| / 1000
Prepočet na model kamery s ohniskom f:  k1 = a1 (f/1000)²,  k2 = a2 (f/1000)⁴  (robí calibrate.py --grid).

Postup: papier (čo najväčší, cez celý záber, aj rohy) pred kameru, kamera ostane na mieste,
nasnímaj ~10 snímok (capture.sh) a spusti:
  python3 tools/calib/grid_distortion.py calib/grid/*.png --out calib/grid
Výstup: grid_result.json (parametre) a grid_curves.jpg (nájdené čiary – skontroluj, že sedia na mriežke).
"""
import argparse, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.signal import find_peaks

ap = argparse.ArgumentParser(description="skreslenie objektívu z fotky mriežky")
ap.add_argument("images", nargs="+", help="snímky mriežky (spriemerujú sa)")
ap.add_argument("--out", required=True, help="priečinok na výsledok")
ap.add_argument("--seam", help="spoj dvoch hárkov: x_hore,x_dole v px (čiary sa cezeň nespájajú)")
ap.add_argument("--mask", action="append", default=[],
                help="vynechaná oblasť x,y,w,h v zlomkoch záberu (predvolene časová pečiatka a logo Tapo)")
args = ap.parse_args()

imgs = [np.asarray(Image.open(p).convert("L"), float) for p in args.images]
img = np.mean(imgs, axis=0)
H, W = img.shape

# --- tmavé čiary -> svetlé hrebene ---
dark = ndimage.gaussian_filter(img, 40) - img
dark = ndimage.gaussian_filter(dark, 3)

# maska: časová pečiatka, logo, okraj, spoj dvoch hárkov
yy, xx = np.mgrid[0:H, 0:W]
mask = np.ones((H, W), bool)
for m in (args.mask or ["0,0,0.38,0.05", "0,0.92,0.12,0.08"]):  # Tapo: pečiatka vľavo hore, logo vľavo dole
    x, y, w, h = (float(v) for v in m.split(","))
    mask[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)] = False
mask[:15] = mask[-15:] = False; mask[:, :15] = mask[:, -15:] = False
if args.seam:
    top, bot = (float(v) for v in args.seam.split(","))
    seam_x = top + (bot - top) * yy / H
    mask &= np.abs(xx - seam_x) > 30
    side = np.where(xx < seam_x, 0, 1)
else:
    side = np.zeros((H, W), int)
safe = ndimage.binary_erosion(mask, iterations=12)  # vrcholy tesne pri maske sú falošné

def trace(horizontal):
    """Sleduje čiary: pre vodorovné čiary hľadá maximá v stĺpcoch a spája ich do kriviek."""
    D = dark if horizontal else dark.T
    M = mask if horizontal else mask.T
    S = side if horizontal else side.T
    SAFE = safe if horizontal else safe.T
    n_along = D.shape[1]
    step = 12
    cols = list(range(20, n_along - 20, step))
    peaks = {}
    for x in cols:
        prof = D[:, x].copy(); prof[~M[:, x]] = 0.0
        pk, pr = find_peaks(prof, distance=35, prominence=0.5)
        peaks[x] = pk[SAFE[pk, x]]
    curves = []
    used = set()
    for start in cols[::20]:
        for p in peaks[start]:
            if (start, p) in used: continue
            s0 = S[p, start]
            curve = [(start, p)]
            for direction in (1, -1):
                pts = [(start, float(p))]
                miss = 0
                i = cols.index(start)
                while True:
                    i += direction
                    if i < 0 or i >= len(cols): break
                    x = cols[i]
                    # predikcia z posledných bodov
                    if len(pts) >= 3:
                        (x1, y1), (x2, y2) = pts[-3], pts[-1]
                        pred = y2 + (y2 - y1) / (x2 - x1) * (x - x2)
                    else:
                        pred = pts[-1][1]
                    cand = peaks[x]
                    if len(cand) == 0: miss += 1
                    else:
                        j = np.argmin(np.abs(cand - pred))
                        if abs(cand[j] - pred) < 8 and S[cand[j], x] == s0:
                            pts.append((x, float(cand[j]))); used.add((x, cand[j])); miss = 0
                        else: miss += 1
                    if miss > 2: break
                if direction == 1: curve = pts
                else: curve = pts[:0:-1][::-1] if False else list(reversed(pts[1:])) + curve
            if len(curve) >= 25:
                c = np.array(curve, float)
                # čistenie: hladká krivka 3. stupňa, body ďalej než 2,5 px sa vyhodia
                ok = np.ones(len(c), bool)
                for _ in range(3):
                    co = np.polyfit(c[ok, 0], c[ok, 1], 4)
                    ok = np.abs(np.polyval(co, c[:, 0]) - c[:, 1]) < 3.0
                if ok.mean() < 0.7 or ok.sum() < 25: continue
                c = c[ok]
                # subpixelová poloha hrebeňa (parabola cez 3 body)
                ys = []
                for x, y in c:
                    y, x = int(y), int(x)
                    a, b, cc = D[y - 1, x], D[y, x], D[y + 1, x]
                    den = a - 2 * b + cc
                    ys.append(y + (0.5 * (a - cc) / den if den != 0 else 0))
                c[:, 1] = ys
                curves.append(c if horizontal else c[:, ::-1])  # vždy (x, y)
    return curves

curves = trace(True) + trace(False)
print(f"nájdených čiar: {len(curves)}, bodov: {sum(len(c) for c in curves)}")

def undistort(q, P):
    cx, cy, a1, a2 = q
    d = P - [cx, cy]; rho2 = (d ** 2).sum(1) / 1e6
    return [cx, cy] + d * (1 + a1 * rho2 + a2 * rho2 ** 2)[:, None]

def line_res(q):
    out = []
    for c in curves:
        u = undistort(q, c)
        m = u.mean(0); _, _, vt = np.linalg.svd(u - m, full_matrices=False)
        d = c - [q[0], q[1]]; rho2 = (d ** 2).sum(1) / 1e6
        out.append(((u - m) @ vt[1]) / (1 + q[2] * rho2 + q[3] * rho2 ** 2))
    return np.concatenate(out)

def rms(q): r = line_res(q); return float(np.sqrt(np.mean(r ** 2))), float(np.percentile(np.abs(r), 95))

fits = {}
r0 = least_squares(line_res, [W / 2, H / 2, 0.1, 0.0], loss="soft_l1", f_scale=1.0, x_scale=[100, 100, 0.01, 0.001])
fits["mriežka: a1, a2, stred"] = r0.x
r1 = least_squares(lambda q: line_res([W / 2, H / 2, q[0], q[1]]), [0.1, 0.0], loss="soft_l1", f_scale=1.0)
fits["mriežka: a1, a2, stred v strede obrazu"] = np.array([W / 2, H / 2, *r1.x])
r2 = least_squares(lambda q: line_res([q[0], q[1], q[2], 0.0]), [W / 2, H / 2, 0.1], loss="soft_l1", f_scale=1.0, x_scale=[100, 100, 0.01])
fits["mriežka: len a1, stred"] = np.array([*r2.x[:3], 0.0])

fits["žiadna korekcia"] = np.array([W / 2, H / 2, 0.0, 0.0])

res = {}
print(f"\n{'model':42s} {'stred':>13s} {'a1':>7s} {'a2':>8s} | odchýlka čiar od priamky RMS / 95 %")
for name, q in fits.items():
    r, p95 = rms(q)
    res[name] = {"cx": float(q[0]), "cy": float(q[1]), "a1": float(q[2]), "a2": float(q[3]), "rms_px": r, "p95_px": p95}
    print(f"{name:42s} {q[0]:6.0f},{q[1]:6.0f} {q[2]:7.4f} {q[3]:8.5f} | {r:6.2f} px / {p95:6.2f} px")

json.dump({"curves": len(curves), "models": res}, open(f"{args.out}/grid_result.json", "w"), indent=1, ensure_ascii=False)
prev = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGB"); d = ImageDraw.Draw(prev)
rng = np.random.default_rng(1)
for c in curves:
    d.line([tuple(p) for p in c], fill=tuple(int(v) for v in rng.integers(60, 255, 3)), width=5)
prev.resize((W // 2, H // 2)).save(f"{args.out}/grid_curves.jpg", quality=85)
print(f"uložené: {args.out}/grid_result.json, {args.out}/grid_curves.jpg")
