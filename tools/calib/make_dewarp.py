"""Nastavenie rovného (dewarp) obrazu pre živý server.

Z modelu kamery (skreslenie z mriežky + sklon/náklon z merania) pripraví do priečinka <out>:
  dewarp.json        – parametre prevodu fisheye -> rovný, vyrovnaný obraz (číta src/dewarp.rs)
  camera_model.json  – model rýchlosti pre rovný obraz (bez skreslenia, sklon 0, náklon 0)
  snapshot_4k.jpg    – snímka z merania prevedená rovnako (na zarovnanie pri štarte)
  ref_points.csv     – body z merania v rovnom obraze

Rovný obraz: virtuálna kamera na tom istom mieste, ale vodorovne a bez náklonu (zvislice sú
zvislé, horizont vodorovný). Ohnisko a stred sa zvolia tak, aby v obraze nebol čierny okraj.

Použitie: python3 tools/calib/make_dewarp.py calib/camera_model.json calib/dewarp [pitch]   (pitch = vyrovnať aj sklon dopredu)
"""
import json, os, sys
import numpy as np
from PIL import Image

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DIR = os.path.dirname(os.path.abspath(SRC))
m = json.load(open(SRC)); p = m["params"]
F, K1, K2, CX, CY = p["f"], p["k1"], p["k2"], m["cx"], m["cy"]
PITCH, ROLL, HCAM = p["pitch"], p["roll"], p["h"]
# vyrovnať aj sklon dopredu? (pri širokom zábere to stiahne dolné rohy a treba viac orezať)
LEVEL_PITCH = len(sys.argv) > 3 and sys.argv[3] == "pitch"
VPITCH = 0.0 if LEVEL_PITCH else PITCH  # sklon virtuálnej (rovnej) kamery
W, H = 3840, 2160

def distort(x, y):
    ru = np.hypot(x, y); rd = ru.copy()
    for _ in range(40):
        g = rd * (1 + K1 * rd**2 + K2 * rd**4) - ru
        rd -= g / (1 + 3 * K1 * rd**2 + 5 * K2 * rd**4)
    s = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
    return CX + F * x * s, CY + F * y * s

def src_pixel(u, v, fo, pu, pv):
    """Pixel rovného obrazu -> pixel pôvodného (rovnaké vzorce ako src/dewarp.rs)."""
    a, b = (u - pu) / fo, (v - pv) / fo
    # smer vo svete z virtuálnej kamery so sklonom VPITCH
    svp, cvp = np.sin(VPITCH), np.cos(VPITCH)
    DX, DY, DZ = a, -b * svp + cvp, -b * cvp - svp
    sp, cp = np.sin(PITCH), np.cos(PITCH)
    fwd = DY * cp - DZ * sp
    down = -DY * sp - DZ * cp
    xr, yr = a / fwd, down / fwd
    c, s = np.cos(ROLL), np.sin(ROLL)
    x, y = c * xr + s * yr, -s * xr + c * yr
    su, sv = distort(x, y)
    return np.where(fwd > 0, su, np.nan), np.where(fwd > 0, sv, np.nan)

def to_virtual(u, v, fo, pu, pv):
    """Pixel pôvodného obrazu -> pixel rovného obrazu."""
    x, y = (u - CX) / F, (v - CY) / F
    r2 = x * x + y * y; s = 1 + K1 * r2 + K2 * r2 * r2
    x, y = x * s, y * s
    c, sn = np.cos(ROLL), np.sin(ROLL)
    xr, yr = c * x - sn * y, sn * x + c * y
    sp, cp = np.sin(PITCH), np.cos(PITCH)
    dX, dY, dZ = xr, -yr * sp + cp, -yr * cp - sp  # svet: X doprava, Y dopredu, Z hore
    svp, cvp = np.sin(VPITCH), np.cos(VPITCH)
    f_ = dY * cvp - dZ * svp; d_ = -dY * svp - dZ * cvp  # do virtuálnej kamery
    return pu + fo * dX / f_, pv + fo * d_ / f_

# --- najmenšie ohnisko (najširší záber) bez čierneho okraja ---
border = np.concatenate([np.c_[np.linspace(0, W - 1, 400), np.zeros(400)], np.c_[np.linspace(0, W - 1, 400), np.full(400, H - 1)],
                         np.c_[np.zeros(300), np.linspace(0, H - 1, 300)], np.c_[np.full(300, W - 1), np.linspace(0, H - 1, 300)]])
def ok(fo, pu, pv, margin=4):
    su, sv = src_pixel(border[:, 0], border[:, 1], fo, pu, pv)
    return np.all(np.isfinite(su)) and su.min() >= margin and su.max() <= W - 1 - margin and sv.min() >= margin and sv.max() <= H - 1 - margin
best = None
pu0, pv0 = to_virtual(np.array([CX]), np.array([CY]), 1000.0, 0.0, 0.0)  # kam padne stred objektívu (pri fo=1000)
for du in np.linspace(-0.25, 0.25, 21):
    for dv in np.linspace(-0.5, 0.5, 41):
        lo, hi = 800.0, 6000.0
        def center(fo): return W / 2 - fo * (pu0[0] / 1000 + du), H / 2 - fo * (pv0[0] / 1000 + dv)
        if not ok(hi, *center(hi)): continue
        for _ in range(30):
            mid = (lo + hi) / 2
            if ok(mid, *center(mid)): hi = mid
            else: lo = mid
        if best is None or hi < best[0]: best = (hi, *center(hi))
FO, PU, PV = best
print(f"rovný obraz: ohnisko {FO:.0f} px (pôvodné {F:.0f}), stred ({PU:.0f}, {PV:.0f}), sklon {np.degrees(VPITCH):.1f}°")

json.dump({"popis": "fisheye -> rovný vyrovnaný obraz (tools/calib/make_dewarp.py)", "width": W, "height": H,
           "lens_f": F, "lens_k1": K1, "lens_k2": K2, "lens_cx": CX, "lens_cy": CY,
           "level_pitch": PITCH, "level_roll": ROLL, "virtual_pitch": VPITCH, "out_f": FO, "out_cx": PU, "out_cy": PV},
          open(f"{OUT}/dewarp.json", "w"), indent=2)

# --- model rýchlosti pre rovný obraz ---
json.dump({"popis": "model kamery pre rovný obraz (dewarp): bez skreslenia, vodorovne", "width": W, "height": H,
           "params": {"f": FO, "k1": 0.0, "k2": 0.0, "pitch": VPITCH, "roll": 0.0, "h": HCAM}, "cx": PU, "cy": PV},
          open(f"{OUT}/camera_model.json", "w"), indent=2)

# --- body z merania ---
rows = [l.strip().split(",", 3) for l in open(f"{DIR}/ref_points.csv").read().strip().splitlines()[1:]]
lines = ["id,x,y,popis"]
for i, x, y, desc in rows:
    u, v = to_virtual(np.array([float(x)]), np.array([float(y)]), FO, PU, PV)
    lines.append(f"{i},{u[0]:.0f},{v[0]:.0f},{desc}")
open(f"{OUT}/ref_points.csv", "w").write("\n".join(lines) + "\n")

# kontrola: poloha bodov na zemi musí byť rovnaká v oboch modeloch
def ground(f, k1, k2, pitch, roll, h, cx, cy, u, v):
    x, y = (u - cx) / f, (v - cy) / f
    r2 = x * x + y * y; s = 1 + k1 * r2 + k2 * r2 * r2; x, y = x * s, y * s
    c, sn = np.cos(roll), np.sin(roll); x, y = c * x - sn * y, sn * x + c * y
    dy = -y * np.sin(pitch) + np.cos(pitch); dz = -y * np.cos(pitch) - np.sin(pitch)
    return h / -dz * x, h / -dz * dy
for (i, x, y, _), l in zip(rows, lines[1:]):
    _, u, v, _ = l.split(",", 3)
    a = ground(F, K1, K2, PITCH, ROLL, HCAM, CX, CY, float(x), float(y))
    b = ground(FO, 0, 0, VPITCH, 0, HCAM, PU, PV, float(u), float(v))
    assert np.hypot(a[0] - b[0], a[1] - b[1]) < 0.1, (i, a, b)
print("body z merania sedia v oboch modeloch (do 10 cm)")

# --- snímka z merania v rovnom obraze (bilineárne, ako cv::remap v src/dewarp.rs) ---
from scipy import ndimage
uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
su, sv = src_pixel(uu, vv, FO, PU, PV)
img = np.asarray(Image.open(f"{DIR}/snapshot_4k.jpg").convert("RGB"), float)
out = np.stack([ndimage.map_coordinates(img[:, :, c], [sv, su], order=1, mode="constant") for c in range(3)], axis=2)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(f"{OUT}/snapshot_4k.jpg", quality=95)
print(f"hotovo: {OUT}")
