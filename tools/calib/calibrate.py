"""Model kamery z meraní na mieste: pixel -> metre na zemi (pre meranie rýchlosti).

Model: kamera vo výške h, sklon (pitch) a náklon (roll), ohnisko f, radiálne skreslenie k1, k2
so stredom (cx, cy). Lúč z pixelu sa pretne so zemou; porovnáva sa s nameranými vzdialenosťami.

Vstup: measurements.json (vzor: measurements.example.json) – pixely referenčných bodov na snímke
z merania a vzdialenosti v metroch:
  "ground": vzdialenosť po zemi od päty kamery (bod pod kamerou) k bodu,
  "slant":  vzdialenosť od kamery (laser priamo z kamery) k bodu,
  "pairs":  vzdialenosť medzi dvoma bodmi,
  "height": výška kamery nad zemou.

Odporúčanie (z overovania): 6–8 meraní po zemi od päty kamery, blízko aj na druhej strane ulice,
vľavo aj vpravo, + výška kamery. Pri 1–3 meraniach model náhodou sedí na jednom mieste, inde nie.

Skreslenie objektívu:
  --grid grid_result.json  skreslenie z fotky mriežky (grid_distortion.py) – najpresnejšie;
                           z meraní sa potom počíta len ohnisko, sklon, náklon a výška
  bez --grid               skreslenie (len k1) sa odhadne z meraní – na okrajoch záberu menej presné

Použitie:
  python3 tools/calib/calibrate.py calib/measurements.json --grid calib/grid/grid_result.json --out calib/camera_model.json
Vedľa výstupu sa zapíše aj ref_points.csv (body na kreslenie). Snímka z merania musí byť v tom istom
priečinku ako calib/snapshot_4k.jpg (podľa nej sa pri štarte zarovná pootočená kamera).
"""
import argparse, json, os
import numpy as np
from scipy.optimize import least_squares

ap = argparse.ArgumentParser(description="model kamery z meraní")
ap.add_argument("measurements")
ap.add_argument("--grid", help="grid_result.json z grid_distortion.py")
ap.add_argument("--out", default="calib/camera_model.json")
args = ap.parse_args()

M = json.load(open(args.measurements))
W, H = M["width"], M["height"]
PX = {k: tuple(v[:2]) for k, v in M["points"].items()}
DESC = {k: (v[2] if len(v) > 2 else "") for k, v in M["points"].items()}
GROUND = {str(k): v for k, v in M.get("ground", {}).items()}
SLANT = {str(k): v for k, v in M.get("slant", {}).items()}
PAIRS = [(str(a), str(b), d) for a, b, d in M.get("pairs", [])]
HEIGHT = M.get("height")

def ground_xy(q, u, v):
    f, k1, k2, pitch, roll, h, cx, cy = q
    x, y = (u - cx) / f, (v - cy) / f
    r2 = x * x + y * y; s = 1 + k1 * r2 + k2 * r2 * r2
    x, y = x * s, y * s
    c, sn = np.cos(roll), np.sin(roll); x, y = c * x - sn * y, sn * x + c * y
    dy = -y * np.sin(pitch) + np.cos(pitch); dz = -y * np.cos(pitch) - np.sin(pitch)
    if dz >= -1e-6: return np.array([np.nan, 1e3])
    return h / -dz * np.array([x, dy])

def measurements(q):
    """(názov, nameraná, model) pre každé meranie."""
    pts = {i: ground_xy(q, *PX[i]) for i in PX}
    out = [(f"päta–{i}", m, np.linalg.norm(pts[i])) for i, m in GROUND.items()]
    out += [(f"kamera–{i}", m, np.hypot(np.linalg.norm(pts[i]), q[5])) for i, m in SLANT.items()]
    out += [(f"{a}–{b}", d, np.linalg.norm(pts[a] - pts[b])) for a, b, d in PAIRS]
    return out

if args.grid:
    G = json.load(open(args.grid))["models"]["mriežka: a1, a2, stred"]
    def full(x):
        f, pitch, roll, h = x
        return np.array([f, G["a1"] * (f / 1000) ** 2, G["a2"] * (f / 1000) ** 4, pitch, roll, h, G["cx"], G["cy"]])
    x0s = [[f0, p0, 0.0, HEIGHT or 4.0] for f0 in (0.5 * W, 0.7 * W, 0.9 * W) for p0 in (0.1, 0.3)]
    lo, hi = [0.2 * W, -0.3, -0.3, 1.0], [3 * W, 1.4, 0.3, 30]
else:
    def full(x):
        f, k1, pitch, roll, h = x
        return np.array([f, k1, 0.0, pitch, roll, h, W / 2, H / 2])
    x0s = [[f0, 0.0, p0, 0.0, HEIGHT or 4.0] for f0 in (0.3 * W, 0.5 * W, 0.7 * W) for p0 in (0.1, 0.3)]
    lo, hi = [0.1 * W, -3, -0.3, -0.3, 1.0], [3 * W, 3, 1.4, 0.3, 30]

def res(x):
    q = full(x)
    r = [model - meas for _, meas, model in measurements(q)]
    if HEIGHT: r.append((q[5] - HEIGHT) * 3)  # výška z laseru je spoľahlivá
    return np.nan_to_num(np.array(r), nan=100.0)

n = len(GROUND) + len(SLANT) + len(PAIRS)
if n < 3: raise SystemExit(f"málo meraní ({n}); treba aspoň 4, odporúčané 6–8")
best = None
for x0 in x0s:
    r = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.3)
    if best is None or r.cost < best.cost: best = r
q = full(best.x)
f, k1, k2, pitch, roll, h, cx, cy = q
print(f"f={f:.0f} px  k1={k1:.3f} k2={k2:.3f}  stred ({cx:.0f}, {cy:.0f})  sklon {np.degrees(pitch):.2f}°  "
      f"náklon {np.degrees(roll):+.2f}°  výška {h:.2f} m")
print(f"{'meranie':14s} {'namerané':>9s} {'model':>8s}  rozdiel")
errs = []
for name, meas, model in measurements(q):
    e = (model - meas) / meas * 100; errs.append(abs(e))
    print(f"{name:14s} {meas:9.2f} {model:8.2f}  {e:+6.1f} %{'   <- nesedí, skontroluj meranie' if abs(e) > 15 else ''}")
print(f"medián odchýlky {np.median(errs):.1f} %")

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
pts = {i: ground_xy(q, *PX[i]) for i in PX}
json.dump({"popis": "model kamery (tools/calib/calibrate.py" + (", skreslenie z mriežky)" if args.grid else ")"),
           "width": W, "height": H,
           "params": {"f": f, "k1": k1, "k2": k2, "pitch": pitch, "roll": roll, "h": h}, "cx": cx, "cy": cy,
           "points": {i: [round(float(p[0]), 3), round(float(p[1]), 3)] for i, p in pts.items()}},
          open(args.out, "w"), indent=2, ensure_ascii=False)
ref = os.path.join(os.path.dirname(os.path.abspath(args.out)), "ref_points.csv")
open(ref, "w").write("id,x,y,popis\n" + "".join(f"{i},{PX[i][0]:.0f},{PX[i][1]:.0f},{DESC[i]}\n" for i in PX))
print(f"uložené: {args.out}, {ref}")
