"""Automatický sklon a náklon kamery zo zvislých hrán a vyrovnanie obrazu.

1. rovný obraz (bez fisheye) podľa modelu objektívu (skreslenie z mriežky)
2. zvislé hrany v rovnom obraze -> spoločný úbežník (RANSAC)
3. z úbežníka: sklon (pitch) a náklon (roll) kamery, horizont
4. vyrovnaný obraz (virtuálna kamera vodorovne, bez náklonu) a pohľad zhora na cestu

Použitie: python3 tools/calib/rectify.py calib/camera_model.json calib/snapshot_4k.jpg <výstupný priečinok>
"""
import json, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

MODEL, IMG, OUT = sys.argv[1:4]
m = json.load(open(MODEL)); p = m["params"]
F, K1, K2, CX, CY = p["f"], p["k1"], p["k2"], m["cx"], m["cy"]
W, H = 3840, 2160
FO = 2000.0  # ohnisko rovného obrazu

def distort(x, y):
    """Normalizované súradnice bez skreslenia -> pixel v pôvodnom (fisheye) obraze."""
    ru = np.hypot(x, y); rd = ru.copy()
    for _ in range(40):  # rd (1 + k1 rd² + k2 rd⁴) = ru
        g = rd * (1 + K1 * rd**2 + K2 * rd**4) - ru
        rd -= g / (1 + 3 * K1 * rd**2 + 5 * K2 * rd**4)
    s = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
    return CX + F * x * s, CY + F * y * s

def remap(src, dst, su, sv, w, h):
    bad = ~np.isfinite(su) | ~np.isfinite(sv) | (su < 0) | (su > W - 1) | (sv < 0) | (sv > H - 1)
    su = np.where(bad, 65535, su); sv = np.where(bad, 65535, sv)
    for name, a in (("xmap", su), ("ymap", sv)):
        with open(f"{OUT}/{name}.pgm", "wb") as fh:
            fh.write(f"P5 {w} {h} 65535\n".encode()); fh.write(np.round(a).astype(">u2").tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", f"{OUT}/xmap.pgm", "-i", f"{OUT}/ymap.pgm",
                    "-lavfi", "[0]format=rgb24[s];[s][1][2]remap=fill=black", "-q:v", "2", dst], check=True)

# ---------- 1. rovný obraz ----------
u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
remap(IMG, f"{OUT}/1_rovny.jpg", *distort((u - W / 2) / FO, (v - H / 2) / FO), W, H)

# ---------- 2. zvislé úsečky -> úbežník ----------
def vertical_segments(path, min_len=80):
    """Zvislé hrany: silný vodorovný gradient, ztenčenie na 1 px, súvislé kúsky -> priamka."""
    g = np.asarray(Image.open(path).convert("L"), float)
    g = ndimage.gaussian_filter(g, 1.5)
    gx, gy = ndimage.sobel(g, 1), ndimage.sobel(g, 0)
    mag = np.hypot(gx, gy)
    valid = ndimage.binary_erosion(g > 3, iterations=8)        # nie čierny okraj po remap
    m = (np.abs(gx) > 3 * np.abs(gy)) & (mag > 60) & valid
    ax = np.abs(gx)
    m &= (ax >= np.roll(ax, 1, 1)) & (ax >= np.roll(ax, -1, 1))  # tenká hrana
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    segs = []
    for sl, i in zip(ndimage.find_objects(lab), range(1, n + 1)):
        if sl[0].stop - sl[0].start < min_len: continue
        ys, xs = np.nonzero(lab[sl] == i)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        a, b = np.polyfit(ys, xs, 1)
        if np.sqrt(np.mean((xs - (a * ys + b)) ** 2)) > 1.0: continue
        y0, y1 = ys.min(), ys.max()
        segs.append([a * y0 + b, y0, a * y1 + b, y1])
    return np.array(segs).reshape(-1, 4)
L = vertical_segments(f"{OUT}/1_rovny.jpg")
np.savetxt(f"{OUT}/lines.csv", L, delimiter=",", header="x1,y1,x2,y2", comments="")
dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
ang = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))  # 0 = zvislá
ln = np.hypot(dx, dy)
# len vo vnútri platného obrazu (nie hrana čierneho okraja) a nie text pečiatky
cand = (ang < 20) & (np.minimum(L[:, 1], L[:, 3]) > 120)
V = L[cand]; Vlen = ln[cand]
# homogénne priamky v normalizovaných súradniciach
P1 = np.c_[(V[:, 0] - W / 2) / FO, (V[:, 1] - H / 2) / FO, np.ones(len(V))]
P2 = np.c_[(V[:, 2] - W / 2) / FO, (V[:, 3] - H / 2) / FO, np.ones(len(V))]
Lh = np.cross(P1, P2); Lh /= np.linalg.norm(Lh[:, :2], axis=1)[:, None]
rng = np.random.default_rng(0)
def vp_err(vp, lines):
    """Uhlová chyba (°) medzi úsečkou a smerom k úbežníku."""
    mid = (P1 + P2) / 2
    d_seg = (P2 - P1)[:, :2]; d_vp = vp[:2] - mid[:, :2] * vp[2]
    cosang = np.abs((d_seg * d_vp).sum(1)) / (np.linalg.norm(d_seg, axis=1) * np.linalg.norm(d_vp, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, 0, 1)))
best = None
for _ in range(3000):
    i, j = rng.choice(len(V), 2, replace=False)
    vp = np.cross(Lh[i], Lh[j])
    if np.linalg.norm(vp) < 1e-12: continue
    inl = vp_err(vp / np.linalg.norm(vp), Lh) < 0.5
    score = Vlen[inl].sum()
    if best is None or score > best[0]: best = (score, inl)
inl = best[1]
# spresnenie: najmenšie štvorce cez inliery (váhované dĺžkou)
A = Lh[inl] * np.sqrt(Vlen[inl])[:, None]
vp = np.linalg.svd(A)[2][-1]
if vp[1] * np.sign(vp[2] if vp[2] != 0 else 1) < 0: vp = -vp
err = vp_err(vp, Lh[inl])
xv, yv = vp[0] / vp[2], vp[1] / vp[2]
roll = np.arctan2(xv, yv)
pitch = np.arctan(1 / np.hypot(xv, yv))
print(f"zvislých úsečiek {len(V)}, zhodných s úbežníkom {inl.sum()} (odchýlka {np.median(err):.2f}°, 90 % do {np.percentile(err, 90):.2f}°)")
print(f"OBRAZ:  sklon {np.degrees(pitch):5.2f}°, náklon {np.degrees(roll):+5.2f}°")
print(f"MODEL:  sklon {np.degrees(p['pitch']):5.2f}°, náklon {np.degrees(p['roll']):+5.2f}°  (z merania laserom)")

# obrázok: zvislé úsečky + horizont v rovnom obraze
im = Image.open(f"{OUT}/1_rovny.jpg"); d = ImageDraw.Draw(im)
for (x1, y1, x2, y2), ok in zip(V, inl):
    d.line([(x1, y1), (x2, y2)], fill=(0, 255, 0) if ok else (255, 0, 0), width=6)
for pp, rr, col in [(pitch, roll, (255, 220, 0)), (p["pitch"], p["roll"], (0, 200, 255))]:
    # horizont: sin(r) x + cos(r) y = -tan(p)  (normalizované súradnice)
    xs = np.array([-W / 2, W / 2]) / FO
    ys = (-np.tan(pp) - np.sin(rr) * xs) / np.cos(rr)
    d.line([(W / 2 + FO * xs[0], H / 2 + FO * ys[0]), (W / 2 + FO * xs[1], H / 2 + FO * ys[1])], fill=col, width=8)
im.resize((1920, 1080)).save(f"{OUT}/2_zvisle_a_horizont.jpg", quality=90)

# ---------- 3. vyrovnaný obraz: virtuálna kamera vodorovne, bez náklonu ----------
def real_pixel(D, pitch, roll):
    """Smer vo svete (X doprava, Y dopredu, Z hore) -> pixel pôvodného obrazu."""
    sp, cp = np.sin(pitch), np.cos(pitch)
    fwd = D[1] * cp - D[2] * sp            # D · (0, cos p, -sin p)
    down = -D[1] * sp - D[2] * cp          # D · (0, -sin p, -cos p)
    xr, yr = D[0] / fwd, down / fwd
    c, s = np.cos(roll), np.sin(roll)
    x, y = c * xr + s * yr, -s * xr + c * yr
    su, sv = distort(x, y)
    return np.where(fwd > 0, su, np.nan), np.where(fwd > 0, sv, np.nan)
HOR = 0.3 * H  # horizont v 30 % výšky
a, b = (u - W / 2) / FO, (v - HOR) / FO
remap(IMG, f"{OUT}/3_vyrovnany.jpg", *real_pixel((a, np.ones_like(a), -b), pitch, roll), W, H)

# ---------- 4. pohľad zhora (1 px = 2 cm) ----------
S = 0.02; X0, X1, Y0, Y1 = -14.0, 16.0, 10.5, 24.0
bw, bh = int((X1 - X0) / S), int((Y1 - Y0) / S)
gx, gy = np.meshgrid(X0 + (np.arange(bw) + 0.5) * S, Y1 - (np.arange(bh) + 0.5) * S)
hcam = p["h"]
remap(IMG, f"{OUT}/4_zhora.jpg", *real_pixel((gx, gy, np.full_like(gx, -hcam)), p["pitch"], p["roll"]), bw, bh)
json.dump({"pitch_img_deg": float(np.degrees(pitch)), "roll_img_deg": float(np.degrees(roll)),
           "pitch_model_deg": float(np.degrees(p["pitch"])), "roll_model_deg": float(np.degrees(p["roll"])),
           "vertical_lines": int(len(V)), "inliers": int(inl.sum()), "median_err_deg": float(np.median(err))},
          open(f"{OUT}/rectify_result.json", "w"), indent=1)
