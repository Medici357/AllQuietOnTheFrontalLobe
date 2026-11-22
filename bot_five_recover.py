"""
bot_five_recover.py — cache + LOS + scan-before-recover + optional hunt mode.
Derived from the user's uploaded bot_v5.py. See README_bot_five_recover.md.

Survivor bot for mechfight/1.

Ideas (placement counts, not kills, so staying alive comes first):
  1. Never stand outside the zone; move to the next circle BEFORE it shrinks.
  2. Build a coarse "can I walk here?" grid from the map files and path-find around buildings.
  3. Aim with lead (predict where the enemy will be), fire only when the torso really points there.
  4. Keep moving sideways (strafe), dodge bullets, shells and charged-laser lines.
  5. Face enemies (hits on the back do x1.5 damage) while the legs run somewhere else.
  6. Grab a second weapon, cubes, supply drops; choose defensive upgrades.
"""
import heapq
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from array import array

import mechfight as mf

TAU = 2 * math.pi
NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
      (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
RANK = {"laser": 3.0, "lightning": 2.0, "mortar": 1.0, "gun": 9.0}
UPGRADE_PRIORITY = ["armor", "plating", "shield", "hp", "health", "regen", "repair",
                    "gun_rate", "rate", "damage", "speed", "boost", "energy", "sensor"]


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def unit(vx, vy):
    n = math.hypot(vx, vy)
    return (vx / n, vy / n) if n > 1e-9 else (0.0, 0.0)


def hdist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def inside_poly(x, y, xs, ys):
    res = False
    j = len(xs) - 1
    for i in range(len(xs)):
        if (ys[i] > y) != (ys[j] > y) and x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i]) + xs[i]:
            res = not res
        j = i
    return res


# --------------------------------------------------------------------------------------
# Walkability grid (4 m cells) built once from the map files
# --------------------------------------------------------------------------------------
class Grid:
    CELL = 4.0

    def __init__(self, map_dir, budget=8.0):
        t0 = time.perf_counter()
        # The simulator may provide a host-side path that is not mounted at
        # that same path inside Docker. Prefer the actual /arena mount there.
        world_path = os.path.join(map_dir, "v2", "world.json")
        if not os.path.isfile(world_path) and os.path.isfile("/arena/v2/world.json"):
            map_dir = "/arena"
            world_path = "/arena/v2/world.json"
        self.map_dir = map_dir

        # Cache is optional. Validate both source files and all grid parameters;
        # on any mismatch/corruption, rebuild from the source map.
        if FEATURES.get("map_cache", True) and self._load_cache(map_dir):
            return

        with open(world_path) as f:
            world = json.load(f)
        tm = world["terrain"]
        self.nx, self.ny = int(tm["nx"]), int(tm["ny"])
        self.minx, self.miny, self.step = float(tm["minX"]), float(tm["minY"]), float(tm["step"])
        ter = array("h")
        with open(os.path.join(map_dir, "v2", "terrain.i16"), "rb") as f:
            ter.frombytes(f.read())
        if sys.byteorder == "big":
            ter.byteswap()
        if len(ter) != self.nx * self.ny:
            raise ValueError("terrain size mismatch")
        self.ter = ter
        C = self.CELL
        W = int((self.nx - 1) * self.step / C) + 1
        H = int((self.ny - 1) * self.step / C) + 1
        self.W, self.H = W, H
        blk = bytearray(W * H)
        minx, miny = self.minx, self.miny

        # 1) buildings / stadium / low bridges
        for p in world.get("prisms", []):
            if time.perf_counter() - t0 > budget:
                break
            ring = p.get("ring")
            if not ring or len(ring) < 6:
                continue
            xs, ys = ring[0::2], ring[1::2]
            if p.get("base", 0) - self.h(xs[0], ys[0]) >= 9.5:
                continue  # a mech (9 m tall) walks under it
            holes = [(hh[0::2], hh[1::2]) for hh in (p.get("holes") or []) if len(hh) >= 6]
            gx0 = max(0, int((min(xs) - minx) / C))
            gx1 = min(W - 1, int((max(xs) - minx) / C))
            gy0 = max(0, int((min(ys) - miny) / C))
            gy1 = min(H - 1, int((max(ys) - miny) / C))
            for gy in range(gy0, gy1 + 1):
                y = miny + (gy + 0.5) * C
                for gx in range(gx0, gx1 + 1):
                    k = gy * W + gx
                    if blk[k]:
                        continue
                    x = minx + (gx + 0.5) * C
                    if inside_poly(x, y, xs, ys) and not any(inside_poly(x, y, hx, hy) for hx, hy in holes):
                        blk[k] = 1
            n = len(xs)
            for i in range(n):  # thin walls: also mark the outline
                ax, ay, bx, by = xs[i], ys[i], xs[(i + 1) % n], ys[(i + 1) % n]
                m = max(1, int(math.hypot(bx - ax, by - ay) / 2))
                for s in range(m + 1):
                    gx = int((ax + (bx - ax) * s / m - minx) / C)
                    gy = int((ay + (by - ay) * s / m - miny) / C)
                    if 0 <= gx < W and 0 <= gy < H:
                        blk[gy * W + gx] = 1

        # 2) slopes too steep to walk
        for gy in range(H):
            y = miny + (gy + 0.5) * C
            for gx in range(W):
                k = gy * W + gx
                if blk[k]:
                    continue
                x = minx + (gx + 0.5) * C
                sx = (self.h(x + 2, y) - self.h(x - 2, y)) / 4.0
                sy = (self.h(x, y + 2) - self.h(x, y - 2)) / 4.0
                if sx * sx + sy * sy > 1.3:
                    blk[k] = 1

        # 3) grow every obstacle by one cell (the mech is 2.5 m wide on each side)
        inf = bytearray(blk)
        for gy in range(H):
            for gx in range(W):
                if blk[gy * W + gx]:
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            a, b = gx + dx, gy + dy
                            if 0 <= a < W and 0 <= b < H:
                                inf[b * W + a] = 1
        self.inf = inf
        if sum(inf) > 0.5 * W * H:
            raise ValueError("grid looks wrong (over half blocked): navigating without it")
        self.map_dir = map_dir
        if FEATURES.get("map_cache", True):
            try:
                self._save_cache(map_dir)
            except Exception as exc:
                mf.log("grid cache write skipped: %s" % exc)

    @staticmethod
    def _source_hashes(map_dir):
        out = {}
        for name in ("world.json", "terrain.i16"):
            h = hashlib.sha256()
            with open(os.path.join(map_dir, "v2", name), "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            out[name] = h.hexdigest()
        return out

    def _cache_paths(self, map_dir):
        d = os.path.join(map_dir, "v2")
        return os.path.join(d, CACHE_META), os.path.join(d, CACHE_DATA)

    def _load_cache(self, map_dir):
        try:
            meta_path, data_path = self._cache_paths(map_dir)
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("version") != CACHE_VERSION or meta.get("cell") != self.CELL:
                return False
            if meta.get("source_hashes") != self._source_hashes(map_dir):
                return False
            self.nx, self.ny = int(meta["nx"]), int(meta["ny"])
            self.minx, self.miny, self.step = float(meta["minx"]), float(meta["miny"]), float(meta["step"])
            self.W, self.H = int(meta["W"]), int(meta["H"])
            with open(data_path, "rb") as f:
                inf = bytearray(f.read())
            if len(inf) != self.W * self.H:
                return False
            if sum(inf) > 0.5 * self.W * self.H:
                return False
            self.inf = inf
            self.map_dir = map_dir
            mf.log("grid cache loaded %dx%d" % (self.W, self.H))
            return True
        except Exception:
            return False

    def _save_cache(self, map_dir):
        meta_path, data_path = self._cache_paths(map_dir)
        meta = {
            "version": CACHE_VERSION, "cell": self.CELL,
            "nx": self.nx, "ny": self.ny, "minx": self.minx,
            "miny": self.miny, "step": self.step, "W": self.W, "H": self.H,
            "source_hashes": self._source_hashes(map_dir),
        }
        # Atomic replace prevents a partial cache being mistaken for a valid one.
        tmp_data, tmp_meta = data_path + ".tmp", meta_path + ".tmp"
        with open(tmp_data, "wb") as f:
            f.write(self.inf)
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, sort_keys=True)
        os.replace(tmp_data, data_path)
        os.replace(tmp_meta, meta_path)

    def h(self, x, y):
        i = int(round((x - self.minx) / self.step))
        j = int(round((y - self.miny) / self.step))
        i = 0 if i < 0 else self.nx - 1 if i >= self.nx else i
        j = 0 if j < 0 else self.ny - 1 if j >= self.ny else j
        return self.ter[j * self.nx + i] / 100.0

    def idx(self, x, y):
        gx = int((x - self.minx) / self.CELL)
        gy = int((y - self.miny) / self.CELL)
        gx = 0 if gx < 0 else self.W - 1 if gx >= self.W else gx
        gy = 0 if gy < 0 else self.H - 1 if gy >= self.H else gy
        return gy * self.W + gx

    def center(self, i):
        return (self.minx + (i % self.W + 0.5) * self.CELL, self.miny + (i // self.W + 0.5) * self.CELL)

    def blocked(self, x, y):
        gx = int((x - self.minx) / self.CELL)
        gy = int((y - self.miny) / self.CELL)
        if gx < 0 or gy < 0 or gx >= self.W or gy >= self.H:
            return True
        return bool(self.inf[gy * self.W + gx])

    def free_dist(self, x, y, ang, maxd, skip=4.0, step=2.0):
        c, s = math.cos(ang), math.sin(ang)
        d = skip
        while d <= maxd:
            if self.blocked(x + c * d, y + s * d):
                return d
            d += step
        return maxd

    def ray_free(self, ax, ay, bx, by, skip=4.0):
        L = math.hypot(bx - ax, by - ay)
        if L <= skip:
            return True
        return self.free_dist(ax, ay, math.atan2(by - ay, bx - ax), L, skip, 2.0) >= L

    def nearest_free(self, x, y):
        i = self.idx(x, y)
        if not self.inf[i]:
            return i
        cx, cy = i % self.W, i // self.W
        best, bd = None, 1e9
        for r in range(1, 8):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    a, b = cx + dx, cy + dy
                    if 0 <= a < self.W and 0 <= b < self.H and not self.inf[b * self.W + a]:
                        d = dx * dx + dy * dy
                        if d < bd:
                            best, bd = b * self.W + a, d
            if best is not None:
                return best
        return None

    def find_path(self, sx, sy, tx, ty, max_exp=1500, tlimit=0.014):
        W, H, inf = self.W, self.H, self.inf
        s = self.idx(sx, sy)
        g = self.nearest_free(tx, ty)
        if g is None:
            return None
        gx_, gy_ = g % W, g // W

        def hf(i):
            dx, dy = abs(i % W - gx_), abs(i // W - gy_)
            return dx + dy - 0.586 * min(dx, dy)

        t0 = time.perf_counter()
        openh = [(hf(s), 0.0, s)]
        cost = {s: 0.0}
        came = {s: -1}
        best, besth, exp = s, hf(s), 0
        while openh:
            f, c, i = heapq.heappop(openh)
            if c > cost.get(i, 1e18):
                continue
            if i == g:
                best = i
                break
            hi = f - c
            if hi < besth:
                best, besth = i, hi
            exp += 1
            if exp > max_exp or (exp & 31 == 0 and time.perf_counter() - t0 > tlimit):
                break
            x0, y0 = i % W, i // W
            for dx, dy, w in NB:
                a, b = x0 + dx, y0 + dy
                if a < 0 or b < 0 or a >= W or b >= H:
                    continue
                j = b * W + a
                if inf[j]:
                    continue
                if dx and dy and (inf[y0 * W + a] or inf[b * W + x0]):
                    continue  # a diagonal step would clip a building's corner
                nc = c + w
                if nc < cost.get(j, 1e18):
                    cost[j] = nc
                    came[j] = i
                    heapq.heappush(openh, (nc + hf(j), nc, j))
        path = []
        i = best
        while i != -1:
            path.append(self.center(i))
            i = came[i]
        path.reverse()
        return path


# --------------------------------------------------------------------------------------
# Aiming helpers
# --------------------------------------------------------------------------------------
def angles(o, p):
    dx, dy, dz = p[0] - o[0], p[1] - o[1], p[2] - o[2]
    return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))


def lead(o, p, v, speed, extra=0.1):
    """Where will the target be when a shot of this speed arrives? (a few refinement passes)"""
    pred = p
    t = math.hypot(p[0] - o[0], p[1] - o[1]) / speed
    for _ in range(3):
        pred = (p[0] + v[0] * (t + extra), p[1] + v[1] * (t + extra), p[2])
        t = math.hypot(pred[0] - o[0], pred[1] - o[1]) / speed
    return pred


def mortar_pitch(hd, dz, v, g):
    A = v * v
    disc = A * A - g * (g * hd * hd + 2 * dz * A)
    if disc < 0 or hd < 1:
        return None
    return math.atan2(A - math.sqrt(disc), g * hd)


STRATEGY = "agg"  # keep the existing movement/combat policy as the default

# Each switch is one change from bot_agg. Flip a switch to False to test without it.
FEATURES = {
    "pref_fix": True,         # lightning only works inside ~42 m; bot_agg hovered at 48 m where it cannot fire
    "drop_fix": True,         # actually stand still in a supply-drop ring (bot_agg walked away and never captured)
    "energy_reserve": True,   # boost only with spare energy when holding lightning (lightning burns energy to fire)
    "smart_search": True,     # with no enemy in view: walk toward the last sound/hit, else toward the zone centre
    "lightning_first": True,  # prefer lightning over laser when picking up weapons (we fight close up)
    "flee_hp": 0.13,          # run away below this HP fraction (bot_agg used 0.13)
    "back_circle": True,      # when close, circle toward the enemy's back (hits from behind do x1.5 damage)
    "loot_dash": True,        # with an empty weapon slot, grab a weapon within 30 m even if an enemy is far away
    "zone_eta": True,         # when buildings block the way to the safe circle, assume a longer walk and leave earlier
    "stagger_hit": True,      # after being hit, pick a side and hold it 0.4-0.8 s instead of re-rolling every hit
    "regen_retreat": False,   # (experimental) back off at 30% HP from a healthier enemy; HP regrows 45/s after 5 s unhurt
    "record": True,           # one short status line in the debug drawing (and stderr if MECH_LOG is set)
    "map_cache": False,        # load validated precomputed walkability grid when available
    "los_gate": False,         # block direct-fire weapons when the coarse map ray is obstructed
    "recover_mode": True,     # disengage at low HP, scan 360 degrees, then recover if safe
    "hunt_mode": False,       # override recovery and pursue enemies aggressively
}

# Runtime overrides allow clean ablations without editing the bot source.
FEATURES["map_cache"] = os.environ.get("MECH_MAP_CACHE", "1") != "0"
FEATURES["los_gate"] = os.environ.get("MECH_LOS", "1") != "0"
FEATURES["recover_mode"] = os.environ.get("MECH_RECOVER", "1") != "0"
FEATURES["hunt_mode"] = os.environ.get("MECH_HUNT", "0") == "1"

# Recovery policy. These are starting hypotheses, not tuned values.
RECOVER_HP_ONE_THREAT = 0.40
RECOVER_HP_TWO_THREATS = 0.60
RECOVER_HP_THREE_THREATS = 0.85
RECOVER_TO_HP = 0.78
RECOVER_SCAN_OBSERVATIONS = 20  # ~2 seconds at 10 decisions/s; torso makes a full turn
RECOVER_THREAT_RANGE = 145.0
RECOVER_RETRY_TICKS = 30
CACHE_VERSION = 1
CACHE_META = "grid_cache.json"
CACHE_DATA = "grid_cache.bin"

if FEATURES["lightning_first"]:
    RANK["lightning"] = 3.5

class Bot:
    name = "v5-recover-py"

    def __init__(self, hello):
        cfg = hello["config"]
        self.reach = cfg["loot"]["pickup_radius"]
        self.rng = random.Random(hello.get("seed", 0))
        self.grid = None
        try:
            self.grid = Grid(hello["map"]["dir"])
        except Exception:
            mf.log("grid build failed, steering without map:\n" + traceback.format_exc())
        self.sign = 1
        self.flip_at = 0
        self.hist = []
        self.stuck_until = -1
        self.stuck_dir = (0.0, 0.0)
        self.path = None
        self.pi = 0
        self.path_goal = None
        self.path_tick = -99
        self.hit_tick = -999
        self.hit_bearing = 0.0
        self.last_ang = None
        self.last_desired = (1.0, 0.0)
        self.hint = None
        self.recover_state = "FIGHT"  # FIGHT -> SCAN -> HOLD; threat/zone forces exit
        self.recover_scan_start = 0
        self.recover_scan_yaw = 0.0
        self.recover_scan_threat = False
        self.recover_last_hit = -999
        self.recover_retry_after = -1
        self.recover_hold_start = -1
        self.recover_escape_start = -1

    # ---------------- navigation ----------------
    def next_wp(self, x, y):
        p = self.path
        while self.pi < len(p) - 1 and hdist(p[self.pi], (x, y)) < 5:
            self.pi += 1
        best = self.pi
        for k in range(min(len(p) - 1, self.pi + 8), self.pi, -1):
            if self.grid.ray_free(x, y, p[k][0], p[k][1], 4.0):
                best = k
                break
        return p[best]

    def waypoint(self, x, y, goal, tick):
        g = self.grid
        if g is None:
            return goal
        try:
            if g.ray_free(x, y, goal[0], goal[1], 4.0):
                self.path = None
                return goal
            if (self.path is None or self.path_goal is None or hdist(self.path_goal, goal) > 10
                    or tick - self.path_tick > 12):
                self.path = g.find_path(x, y, goal[0], goal[1])
                self.pi = 0
                self.path_goal = goal
                self.path_tick = tick
            if self.path and len(self.path) > 1:
                return self.next_wp(x, y)
        except Exception:
            mf.log("navigation failed:\n" + traceback.format_exc())
            self.grid = None
        return goal

    def steer(self, x, y, dvec, zc, zr):
        n = math.hypot(dvec[0], dvec[1])
        if n < 0.05:
            return (0.0, 0.0)
        m = min(1.0, n)
        ux, uy = dvec[0] / n, dvec[1] / n
        g = self.grid
        if g is None:
            return (ux * m, uy * m)
        best, besta = -1e9, None
        for k in range(24):
            a = k * TAU / 24
            c, s = math.cos(a), math.sin(a)
            sc = c * ux + s * uy
            fd = g.free_dist(x, y, a, 20.0, 5.0, 3.0)
            if fd < 20:
                sc -= (20 - fd) / 15.0 * 1.6
            if zr > 0 and math.hypot(x + c * 10 - zc[0], y + s * 10 - zc[1]) > zr - 2:
                sc -= 0.8
            if self.last_ang is not None and abs(wrap(a - self.last_ang)) < 0.2:
                sc += 0.15
            if sc > best:
                best, besta = sc, a
        self.last_ang = besta
        return (math.cos(besta) * m, math.sin(besta) * m)

    # ---------------- zone ----------------
    @staticmethod
    def inside_point(nc, nr, x, y, f):
        d = math.hypot(x - nc[0], y - nc[1])
        if d <= nr * f:
            return (x, y)
        ux, uy = unit(x - nc[0], y - nc[1])
        return (nc[0] + ux * nr * f, nc[1] + uy * nr * f)

    def zone_goal(self, obs, me, x, y, t, speed):
        z = obs["zone"]
        c, r, nc, nr = z["center"], z["radius"], z["next_center"], z["next_radius"]
        if math.hypot(x - c[0], y - c[1]) > r - 6 or not me["in_zone"]:
            return self.inside_point(nc, nr, x, y, 0.6), 2
        d_next = math.hypot(x - nc[0], y - nc[1])
        if d_next > nr - 8:
            travel = (d_next - nr * 0.6) / max(speed, 5.0)
            if FEATURES["zone_eta"] and self.grid is not None:
                g_ = self.inside_point(nc, nr, x, y, 0.6)
                if not self.grid.ray_free(x, y, g_[0], g_[1], 4.0):
                    travel *= 1.5          # buildings in the way: the real walk is longer than a straight line
            if t >= z["shrink_start"] or z["shrink_start"] - t < travel + 15:
                return self.inside_point(nc, nr, x, y, 0.6), 1
        return None, 0

    # ---------------- weapons ----------------
    def shoot(self, me, tpos, tvel, special, target_h=4.5):
        """Aim and decide which triggers to hold. Every weapon is tested on its own, so gun + lightning can fire together."""
        x, y, z = me["pos"]
        mh = me["stats"].get("muzzle_height", 7.0)
        o = (x, y, z + mh)
        cur = me["aim"]
        tc = (tpos[0], tpos[1], tpos[2] + target_h)
        d = hdist(tpos, (x, y))
        hpf = me["hp"] / max(1.0, me["max_hp"])
        held = {}
        for i, w in enumerate(me["weapons"]):
            if w:
                held[w["kind"]] = (i, w)
        aims = {}            # kind -> (yaw, pitch, slot, usable)
        fires = [False, False]

        def err(yaw, pit):
            return abs(wrap(cur[0] - yaw)), abs(cur[1] - pit)

        if "gun" in held:
            i, w = held["gun"]
            st = w.get("stats", {})
            yaw, pit = angles(o, lead(o, tc, tvel, st.get("speed", 252), 0.05))
            ok = d < st.get("range", 230) * 0.92 and w.get("ammo", 1) > 0 and w.get("reload", 0) <= 0
            aims["gun"] = (yaw, pit, i, ok)
            if ok:
                spread = math.radians(st.get("spread_deg", 0.6))
                tol = clamp(math.atan2(2.5, max(d, 5.0)) + spread + 0.01, 0.025, 0.3)   # the enemy's size at this range
                ye, pe = err(yaw, pit)
                if ye < tol and pe < tol * 1.6:
                    fires[i] = True

        if special and "laser" in held and hpf > 0.25:       # charging halves our speed: not when weak
            i, w = held["laser"]
            ch, cd = w.get("charge", 0), w.get("cooldown", 0)
            yaw, pit = angles(o, lead(o, tc, tvel, 1e9, 0.5 * max(0.1, 0.65 - ch) + 0.1))
            ok = 10 < d < 260
            aims["laser"] = (yaw, pit, i, ok)
            ye, pe = err(yaw, pit)
            tol = clamp(math.atan2(2.5, max(d, 5.0)) + 0.02, 0.02, 0.2)
            aligned = ye < tol and pe < tol * 1.6
            if ch > 0:
                if not ok:
                    pass                                   # target gone: release
                elif ch < 0.65:
                    fires[i] = True                        # keep charging
                else:
                    fires[i] = not aligned                 # fully charged: release the moment we are lined up
            elif ok and cd <= 0.15 and ye < 0.25:
                fires[i] = True                            # start a charge

        if special and "lightning" in held:
            i, w = held["lightning"]
            rng_ = w.get("stats", {}).get("range", 45)
            yaw, pit = angles(o, tc)
            ok = d < rng_ - 2 and me["energy"] > 8
            aims["lightning"] = (yaw, pit, i, ok)
            if ok and w.get("cooldown", 0) <= 0.05 and err(yaw, pit)[0] < 0.4:   # the arc has a wide cone
                fires[i] = True

        if special and "mortar" in held:
            i, w = held["mortar"]
            st = w.get("stats", {})
            v, g = st.get("speed", 72), st.get("gravity", 22)
            o2 = (x, y, z + mh + 2.5)
            pred, pit_m = tc, None
            for _ in range(3):
                pit_m = mortar_pitch(hdist(pred, o2), pred[2] - o2[2], v, g)
                if pit_m is None:
                    break
                tf = hdist(pred, o2) / max(1.0, v * math.cos(pit_m))
                pred = (tc[0] + tvel[0] * (tf + 0.1), tc[1] + tvel[1] * (tf + 0.1), tc[2])
            if pit_m is not None:
                yaw = math.atan2(pred[1] - y, pred[0] - x)
                slow = math.hypot(tvel[0], tvel[1]) < 5
                ok = 25 < d < 200 and w.get("cooldown", 0) <= 0.1 and (slow or d < 80)
                aims["mortar"] = (yaw, pit_m, i, ok)
                if ok:
                    ye, pe = err(yaw, pit_m)
                    if ye < 0.15 and pe < 0.3:
                        fires[i] = True

        # The torso has ONE aim. Priority: a laser mid-charge, a long-range laser shot, then the gun's lead aim.
        primary = None
        if "laser" in aims and held["laser"][1].get("charge", 0) > 0:
            primary = aims["laser"]
        elif "laser" in aims and aims["laser"][3] and d > 90 and held["laser"][1].get("cooldown", 0) <= 0.3:
            primary = aims["laser"]
        if primary is None:
            for k in ("gun", "mortar", "lightning", "laser"):
                if k in aims and aims[k][3]:
                    primary = aims[k]
                    break
        if primary is None:
            for k in ("gun", "lightning", "mortar", "laser"):
                if k in aims:
                    primary = aims[k]
                    break
        if primary is None:
            yaw, pit = angles(o, tc)
            return [yaw, pit], fires
        return [primary[0], primary[1]], fires

    def strafe_sign(self, e, x, y, d):
        """Which way to circle the enemy: toward its back when close (rear hits do x1.5), else a random side."""
        if FEATURES["back_circle"] and d < 70 and e.get("aim"):
            rel = wrap(math.atan2(y - e["pos"][1], x - e["pos"][0]) - e["aim"][0])
            if abs(rel) < 2.4 and self.rng.random() < 0.75:
                return 1.0 if rel < 0 else -1.0
        return float(self.sign)

    def pref_dist(self, held_kinds, enemy):
        if "lightning" in held_kinds:
            pref = 28.0
        elif "laser" in held_kinds:
            pref = 110.0
        elif "mortar" in held_kinds:
            pref = 120.0
        else:
            pref = 85.0
        if "lightning" not in held_kinds and "lightning" in (enemy.get("weapons") or []):
            pref = max(pref, 70.0)
        return pref

    # ---------------- where might enemies be? ----------------
    def update_hints(self, obs, pos, tick):
        """Remember the most recent sound / hit direction as a rough guess of where a fight is."""
        for s in obs["sounds"]:
            if s["kind"] in ("gun", "mortar", "laser", "lightning", "explosion", "laser_charge", "step", "boost", "jet"):
                dist = {"near": 40.0, "mid": 110.0, "far": 220.0}.get(s["dist"], 110.0)
                self.hint = (pos[0] + math.cos(s["bearing"]) * dist, pos[1] + math.sin(s["bearing"]) * dist, tick)
        for ev in obs["events"]:
            if ev.get("type") == "damaged" and ev.get("bearing") is not None and ev.get("weapon") != "zone":
                self.hint = (pos[0] + math.cos(ev["bearing"]) * 60.0, pos[1] + math.sin(ev["bearing"]) * 60.0, tick)

    def search_hint(self, pos, tick):
        h = self.hint
        if h is not None and tick - h[2] < 80 and hdist(pos, h) > 15:
            return (h[0], h[1])
        return None

    # ---------------- main decision ----------------
    def act(self, obs):
        tick, t = obs["tick"], obs["t"]
        me = obs["self"]
        x, y, z = me["pos"]
        pos = (x, y)
        st = me["stats"]
        walk = st.get("walk_speed", 11.7)
        hpf = me["hp"] / max(1.0, me["max_hp"])
        zone = obs["zone"]
        act = {"tick": tick}
        self.update_hints(obs, pos, tick)

        for ev in obs["events"]:
            if ev.get("type") == "damaged":
                self.hit_tick = tick
                self.recover_last_hit = tick
                if ev.get("bearing") is not None:
                    self.hit_bearing = ev["bearing"]
                if FEATURES["stagger_hit"]:
                    self.sign = self.rng.choice((-1, 1))                    # pick a side and hold it
                    self.flip_at = tick + self.rng.randint(4, 8)
                elif self.rng.random() < 0.5:
                    self.sign = -self.sign

        # upgrade choice
        offer = obs["offer"]
        if offer:
            best_i, best_r = 0, 99
            for i, ch in enumerate(offer["choices"]):
                key = (ch["id"] + " " + ch["name"]).lower()
                for r, word in enumerate(UPGRADE_PRIORITY):
                    if word in key:
                        if r < best_r:
                            best_i, best_r = i, r
                        break
            act["pick"] = best_i

        if tick >= self.flip_at:
            self.sign = -self.sign
            self.flip_at = tick + self.rng.randint(6, 18)

        held = {w["kind"]: i for i, w in enumerate(me["weapons"]) if w}
        enemies = []
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                p3 = math.sqrt((m["pos"][0] - x) ** 2 + (m["pos"][1] - y) ** 2 + (m["pos"][2] - z) ** 2)
                enemies.append((p3, m))
        npcs = [(hdist(m["pos"], pos), m) for m in obs["mechs"] if m["kind"] == "npc"]
        crates = [(hdist(c["pos"], pos), c) for c in obs["crates"]]
        near_enemies = [(d, m) for d, m in enemies if d < 130]
        n_close = len(near_enemies)
        flee = n_close >= 3 or (hpf < 0.3 and n_close >= 1)

        # ---- pick up a weapon we are standing next to
        free_slot = next((i for i, w in enumerate(me["weapons"]) if w is None), None)
        worst_slot, worst_rank = None, 99.0
        for i, w in enumerate(me["weapons"]):
            if w and RANK.get(w["kind"], 0) < worst_rank and w.get("charge", 0) <= 0:
                worst_slot, worst_rank = i, RANK.get(w["kind"], 0)
        want_rank = 0.0 if free_slot is not None else worst_rank
        loot_wp = None
        best_loot_score = -1e9
        for p in obs["pickups"]:
            if p["kind"] == "gun" or p["kind"] in held:
                continue
            r = RANK.get(p["kind"], 0)
            if r <= want_rank:
                continue
            d = hdist(p["pos"], pos)
            if d < self.reach - 0.4 and abs(p["pos"][2] - z) < 3.5:
                if free_slot is not None:
                    act["take"] = p["id"]
                elif worst_slot is not None and not (near_enemies and near_enemies[0][0] < 30):
                    act["take"] = p["id"]
                    act["drop"] = worst_slot
                break
            if d < 120 and (loot_wp is None or r * 50 - d > best_loot_score):
                best_loot_score = r * 50 - d
                loot_wp = (d, p["pos"])

        # ---- targets
        tgt = None
        if enemies:
            tgt = min(enemies, key=lambda e: e[0] * (0.6 + 0.8 * (e[1].get("hp", 1000) / max(1.0, e[1].get("max_hp", 1000)))))
        farm = None
        if tgt is None and hpf > 0.5:
            opts = [(d, m, "npc") for d, m in npcs if d < 90] + [(d, c, "crate") for d, c in crates if d < 60]
            if opts:
                farm = min(opts, key=lambda o: o[0])

        aim, fire = None, [False, False]
        if tgt and tgt[0] < 235:
            aim, fire = self.shoot(me, tgt[1]["pos"], tgt[1]["vel"], True, 4.5 * (1 + 0.05 * (tgt[1].get("cubes", 0) or 0)))
        elif farm:
            tv = farm[1].get("vel", (0, 0, 0))
            aim, fire = self.shoot(me, farm[1]["pos"], tv, False)
        elif tick - self.hit_tick < 15:
            aim = [self.hit_bearing, 0.0]
        else:
            for s in obs["sounds"]:
                if s["kind"] in ("gun", "mortar", "laser", "lightning", "laser_charge") and s["dist"] != "far" and s["age"] < 0.6:
                    aim = [s["bearing"], 0.0]
                    break

        # Direct-fire LOS gate. The grid is coarse (4 m cells), so this is a
        # conservative heuristic, not an exact projectile collision test.
        los_clear = True
        if FEATURES.get("los_gate", True) and tgt is not None and self.grid is not None:
            los_clear = self.grid.ray_free(x, y, tgt[1]["pos"][0], tgt[1]["pos"][1], 4.0)
            if not los_clear:
                for wi, weapon in enumerate(me["weapons"]):
                    if weapon and weapon.get("kind") in ("gun", "laser", "lightning"):
                        fire[wi] = False

        # ---- where do we want to go?
        zgoal, zurg = self.zone_goal(obs, me, x, y, t, walk)
        desired = [0.0, 0.0]

        def add(v, w):
            desired[0] += v[0] * w
            desired[1] += v[1] * w

        boost_wanted = False
        mode = "hold"
        hold_ring = False
        goal_pt = None
        engaged = tgt is not None and tgt[0] < 70

        if zgoal is not None:
            goal_pt = zgoal
            wp = self.waypoint(x, y, zgoal, tick)
            add(unit(wp[0] - x, wp[1] - y), 2.0 if zurg == 2 else 1.0)
            mode = "zone%d" % zurg
            if zurg == 2 and me["energy"] > 8:
                boost_wanted = True
        elif not flee and not engaged:
            # supply drop
            nc, nr = zone["next_center"], zone["next_radius"]
            drop = None
            for dr in obs["drops"]:
                dd = hdist(dr["pos"], pos)
                if dd > 240 or hdist(dr["pos"], nc) > nr - 6:
                    continue
                if dr["lands_in"] < dd / max(walk * 0.85, 5.0) + 10 and (drop is None or dd < drop[0]):
                    drop = (dd, dr)
            if drop is not None:
                dd, dr = drop
                if dd < dr["radius"] * 0.5:
                    hold_ring = True
                else:
                    goal_pt = dr["pos"]
                    wp = self.waypoint(x, y, goal_pt, tick)
                    add(unit(wp[0] - x, wp[1] - y), 1.0)
                    mode = "drop"
                    if dd > 80 and me["energy"] > 70:
                        boost_wanted = True
            elif loot_wp is not None:
                goal_pt = loot_wp[1]
                wp = self.waypoint(x, y, goal_pt, tick)
                add(unit(wp[0] - x, wp[1] - y), 1.0)
                mode = "loot"
            else:
                cube = None
                for c in obs["cubes"]:
                    dd = hdist(c["pos"], pos)
                    if dd < 45 and (cube is None or dd < cube[0]):
                        cube = (dd, c["pos"])
                if cube is not None:
                    goal_pt = cube[1]
                    wp = self.waypoint(x, y, goal_pt, tick)
                    add(unit(wp[0] - x, wp[1] - y), 1.0)
                    mode = "cube"
                elif farm is not None and farm[0] > 35:
                    goal_pt = farm[1]["pos"]
                    wp = self.waypoint(x, y, goal_pt, tick)
                    add(unit(wp[0] - x, wp[1] - y), 0.9)
                    mode = "farm"

        # ---- fight movement
        if tgt is not None:
            e = tgt[1]
            ux, uy = unit(e["pos"][0] - x, e["pos"][1] - y)
            d = hdist(e["pos"], pos)
            pref = self.pref_dist(held, e)
            radial = clamp((d - pref) / 25.0, -1.0, 1.0)
            ehp = e.get("hp", 1000) / max(1.0, e.get("max_hp", 1000))
            aggressive = hpf > 0.6 and ehp < hpf + 0.15 and n_close <= 1
            if radial > 0 and not aggressive:
                radial = 0.0
            add((ux, uy), radial)
            if d < 170:
                add((-uy, ux), 0.9 * self.sign * (0.75 + 0.25 * self.rng.random()))
            if mode == "hold":
                mode = "fight"
        if flee:
            fx = fy = 0.0
            for d, e in near_enemies:
                ax, ay = unit(x - e["pos"][0], y - e["pos"][1])
                w = 1.0 / max(d, 20.0)
                fx += ax * w
                fy += ay * w
            add(unit(fx, fy), 1.5)
            mode = "flee"
            if me["energy"] > 25:
                boost_wanted = True


        # ================================================================
        # STRATEGY-SPECIFIC MOVEMENT POLICY
        # Shared combat, aiming, navigation and projectile dodging remain.
        # ================================================================

        if STRATEGY == "agg":

            # AGGRESSIVE: pursue and fight almost everything.
            # Retreat only at critically low health.
            if tgt is not None and zgoal is None:

                e = tgt[1]
                d = hdist(e["pos"], pos)
                ux, uy = unit(e["pos"][0] - x, e["pos"][1] - y)

                flee_at = FEATURES["flee_hp"]
                if FEATURES["regen_retreat"] and e.get("hp", 1000) / max(1.0, e.get("max_hp", 1000)) > hpf + 0.2:
                    flee_at = 0.30

                if (FEATURES["loot_dash"] and loot_wp is not None and free_slot is not None
                        and loot_wp[0] < 30 and d > 50 and hpf >= flee_at):
                    # an empty weapon slot and a weapon close by: grab it first
                    desired = [0.0, 0.0]
                    wpd = self.waypoint(x, y, loot_wp[1], tick)
                    add(unit(wpd[0] - x, wpd[1] - y), 1.5)
                    goal_pt = loot_wp[1]
                    mode = "LOOT"

                elif hpf < flee_at:
                    desired = [0.0, 0.0]
                    fx, fy = 0.0, 0.0
                    for ed, enemy in near_enemies:
                        ax, ay = unit(x - enemy["pos"][0],
                                      y - enemy["pos"][1])
                        fx += ax / max(ed, 15.0)
                        fy += ay / max(ed, 15.0)
                    add(unit(fx, fy), 2.0)
                    mode = "FLEE"
                    boost_wanted = me["energy"] > 15

                else:
                    desired = [0.0, 0.0]

                    # Close distance instead of maintaining the baseline's
                    # more cautious preferred range.
                    if FEATURES["pref_fix"]:
                        preferred = 28.0 if "lightning" in held else 62.0   # lightning range is ~45 m
                    else:
                        preferred = 48.0 if "lightning" in held else 62.0
                    radial = clamp((d - preferred) / 25.0, -0.25, 1.0)
                    add((ux, uy), radial)

                    # Keep moving around the target while closing in.
                    add((-uy, ux), 0.55 * self.strafe_sign(e, x, y, d))

                    mode = "FIGHT"

                    en_min = 60 if ("lightning" in held and FEATURES["energy_reserve"]) else 25
                    if d > 75 and me["energy"] > en_min:
                        boost_wanted = True

            elif tgt is None and zgoal is None and hold_ring and FEATURES["drop_fix"]:
                # Standing in a supply-drop ring: stay put until the 4 s capture finishes.
                desired = [0.0, 0.0]
                mode = "CAPTURE"

            elif tgt is None and zgoal is None:
                # No visible enemy: actively search instead of standing still.
                if not hasattr(self, "agg_angle"):
                    self.agg_angle = self.rng.uniform(-math.pi, math.pi)
                    self.agg_turn = tick

                if tick - self.agg_turn > self.rng.randint(45, 85):
                    self.agg_angle += self.rng.uniform(-1.4, 1.4)
                    self.agg_turn = tick

                if math.hypot(desired[0], desired[1]) < 0.1:
                    desired = [math.cos(self.agg_angle),
                               math.sin(self.agg_angle)]
                    mode = "SEARCH"
                    if FEATURES["smart_search"]:
                        hint = self.search_hint(pos, tick)
                        nc_, nr_ = zone["next_center"], zone["next_radius"]
                        if hint is not None:
                            desired = list(unit(hint[0] - x, hint[1] - y))
                            mode = "HUNT"
                        elif hdist(pos, nc_) > 0.45 * nr_:
                            # everyone is pushed toward the middle of the next circle: go and meet them
                            toc = unit(nc_[0] - x, nc_[1] - y)
                            desired = [toc[0] * 0.8 + math.cos(self.agg_angle) * 0.4,
                                       toc[1] * 0.8 + math.sin(self.agg_angle) * 0.4]
                            mode = "SEEK"

        elif STRATEGY == "run":

            # RUNNER: keep moving, avoid enemies, and stay comfortably
            # inside the current safe zone.
            desired = [0.0, 0.0]

            if zgoal is not None:
                wp = self.waypoint(x, y, zgoal, tick)
                add(unit(wp[0] - x, wp[1] - y), 2.0)
                mode = "ZONE"
                boost_wanted = me["energy"] > 12

            else:
                if not hasattr(self, "run_angle"):
                    self.run_angle = self.rng.uniform(-math.pi, math.pi)
                    self.run_turn = tick

                # Change direction periodically, not every frame.
                if tick - self.run_turn > self.rng.randint(45, 90):
                    self.run_angle += self.rng.uniform(-1.0, 1.0)
                    self.run_turn = tick

                desired = [math.cos(self.run_angle),
                           math.sin(self.run_angle)]

                # Bias movement toward the interior when approaching
                # the edge of the current circle.
                zc = zone["center"]
                zr = zone["radius"]
                dc = hdist(pos, zc)

                if me["in_zone"] and dc > zr * 0.62:
                    inward = unit(zc[0] - x, zc[1] - y)
                    add(inward, 2.0)

                # Repel nearby enemies. The closer they are, the stronger
                # the avoidance response.
                closest = 9999.0
                for ed, enemy in near_enemies:
                    closest = min(closest, ed)
                    if ed < 115:
                        away = unit(x - enemy["pos"][0],
                                    y - enemy["pos"][1])
                        weight = 1.5 * (1.0 - ed / 115.0)
                        add(away, weight)

                if closest < 38:
                    mode = "FLEE"
                    boost_wanted = me["energy"] > 12
                else:
                    mode = "RUN"

                if closest < 65 and me["energy"] > 25:
                    boost_wanted = True

        elif STRATEGY == "def":

            # DEFENDER: choose a position, hold it, and relocate when
            # compromised or forced by the zone.
            desired = [0.0, 0.0]

            if not hasattr(self, "def_goal"):
                self.def_goal = None
                self.def_last_relocate = -10000

            if zgoal is not None:
                # Zone movement always takes priority.
                self.def_goal = tuple(zgoal)
                wp = self.waypoint(x, y, zgoal, tick)
                add(unit(wp[0] - x, wp[1] - y), 2.0)
                mode = "ZONE"
                boost_wanted = me["energy"] > 15

            else:
                # Has the current position been compromised?
                discovered = (tick - self.hit_tick <= 8)

                for ed, enemy in enemies:
                    if ed < 55:
                        discovered = True
                        break

                # Also relocate if the current position is becoming
                # too close to the edge of the circle.
                dc = hdist(pos, zone["center"])
                if me["in_zone"] and dc > zone["radius"] * 0.72:
                    discovered = True

                relocate = (
                    self.def_goal is None
                    or (
                        discovered
                        and tick - self.def_last_relocate > 40
                    )
                )

                if relocate:
                    # Sample a new position reasonably far from the
                    # current one, but comfortably inside the circle.
                    if me["in_zone"]:
                        center = zone["center"]
                        radius = zone["radius"]
                    else:
                        center = zone["next_center"]
                        radius = zone["next_radius"]

                    candidate = pos

                    for attempt in range(20):
                        angle = self.rng.uniform(-math.pi, math.pi)
                        dist = radius * math.sqrt(
                            self.rng.uniform(0.05, 0.30)
                        )
                        px = center[0] + math.cos(angle) * dist
                        py = center[1] + math.sin(angle) * dist

                        if hdist((px, py), pos) < 28:
                            continue

                        if self.grid is not None and self.grid.blocked(px, py):
                            continue

                        candidate = (px, py)
                        break

                    if self.grid is not None:
                        free = self.grid.nearest_free(
                            candidate[0], candidate[1]
                        )
                        if free is not None:
                            candidate = self.grid.center(free)

                    self.def_goal = candidate
                    self.def_last_relocate = tick
                    self.path = None

                # Move to the selected position, then remain stationary.
                if hdist(pos, self.def_goal) > 7:
                    wp = self.waypoint(
                        x, y, self.def_goal, tick
                    )
                    add(unit(wp[0] - x, wp[1] - y), 1.3)
                    mode = "RELOCATE"
                else:
                    mode = "HOLD"

        # ================================================================
        # RECOVERY: disengage, rotate the torso through a full 360-degree
        # scan, then hold still only if the scan and ongoing observations look safe.
        # Zone movement always wins. HUNT_MODE explicitly disables recovery.
        # ================================================================
        if FEATURES.get("recover_mode", True) and not FEATURES.get("hunt_mode", False):
            visible_threats = []
            for ed, enemy in enemies:
                if ed <= RECOVER_THREAT_RANGE:
                    has_los = True
                    if self.grid is not None:
                        has_los = self.grid.ray_free(x, y, enemy["pos"][0], enemy["pos"][1], 4.0)
                    if has_los or ed < 45:
                        visible_threats.append((ed, enemy))
            recent_damage = tick - self.recover_last_hit <= 8
            threat_now = bool(visible_threats) or recent_damage

            if zgoal is not None:
                self.recover_state = "FIGHT"
                self.recover_retry_after = tick + 10
            else:
                # Scale the disengagement threshold with the number of threats.
                n_threat = len(visible_threats)
                threshold = RECOVER_HP_THREE_THREATS if n_threat >= 3 else (RECOVER_HP_TWO_THREATS if n_threat >= 2 else RECOVER_HP_ONE_THREAT)
                should_start = hpf < threshold and tick >= self.recover_retry_after

                if self.recover_state == "FIGHT" and should_start:
                    self.path = None
                    self.recover_hold_start = -1
                    if threat_now:
                        self.recover_state = "ESCAPE"
                        self.recover_escape_start = tick
                    else:
                        self.recover_state = "SCAN"
                        self.recover_scan_start = tick
                        self.recover_scan_yaw = me["aim"][0]
                        self.recover_scan_threat = False

                if self.recover_state == "ESCAPE":
                    if threat_now:
                        fx = fy = 0.0
                        for ed, enemy in visible_threats:
                            ax, ay = unit(x - enemy["pos"][0], y - enemy["pos"][1])
                            weight = 1.0 / max(ed, 12.0)
                            fx += ax * weight
                            fy += ay * weight
                        if math.hypot(fx, fy) < 1e-6:
                            fx, fy = math.cos(self.hit_bearing + math.pi), math.sin(self.hit_bearing + math.pi)
                        desired = [0.0, 0.0]
                        add(unit(fx, fy), 2.0)
                        boost_wanted = me["energy"] > 12
                        mode = "RECOVER_ESCAPE"
                    elif tick - self.recover_escape_start >= 8:
                        self.recover_state = "SCAN"
                        self.recover_scan_start = tick
                        self.recover_scan_yaw = me["aim"][0]
                        self.recover_scan_threat = False

                if self.recover_state == "SCAN":
                    self.recover_scan_threat = self.recover_scan_threat or threat_now
                    elapsed = max(0, tick - self.recover_scan_start)
                    progress = min(1.0, elapsed / float(RECOVER_SCAN_OBSERVATIONS * 2))
                    scan_yaw = wrap(self.recover_scan_yaw + TAU * progress)
                    desired = [0.0, 0.0]
                    boost_wanted = False
                    mode = "RECOVER_SCAN"
                    aim = [scan_yaw, 0.0]
                    fire = [False, False]
                    if self.recover_scan_threat:
                        # A threat was observed during the sweep: move away before retrying.
                        self.recover_state = "ESCAPE"
                        self.recover_escape_start = tick
                        mode = "RECOVER_ABORT"
                    elif progress >= 1.0:
                        self.recover_state = "HOLD"
                        self.recover_hold_start = tick
                        mode = "RECOVER_HOLD"

                elif self.recover_state == "HOLD":
                    # Keep checking every observation; a scan is not a guarantee.
                    if threat_now:
                        self.recover_state = "ESCAPE"
                        self.recover_escape_start = tick
                        fx = fy = 0.0
                        for ed, enemy in visible_threats:
                            ax, ay = unit(x - enemy["pos"][0], y - enemy["pos"][1])
                            weight = 1.0 / max(ed, 12.0)
                            fx += ax * weight
                            fy += ay * weight
                        if math.hypot(fx, fy) < 1e-6:
                            fx, fy = math.cos(self.hit_bearing + math.pi), math.sin(self.hit_bearing + math.pi)
                        desired = [0.0, 0.0]
                        add(unit(fx, fy), 2.0)
                        boost_wanted = me["energy"] > 12
                        fire = [False, False]
                        mode = "RECOVER_ABORT"
                    elif hpf >= RECOVER_TO_HP and tick - self.recover_last_hit >= 50:
                        self.recover_state = "FIGHT"
                        mode = "RECOVER_DONE"
                    else:
                        desired = [0.0, 0.0]
                        boost_wanted = False
                        aim = [me["aim"][0], 0.0]
                        fire = [False, False]
                        mode = "RECOVER_HOLD"

        # Optional all-out hunt mode: override ordinary retreat logic and
        # keep pressing the selected target. Zone safety and projectile dodging
        # still take precedence; this mode is intentionally high risk.
        if FEATURES.get("hunt_mode", False) and tgt is not None and zgoal is None:
            enemy = tgt[1]
            dx, dy = unit(enemy["pos"][0] - x, enemy["pos"][1] - y)
            dd = hdist(enemy["pos"], pos)
            desired = [0.0, 0.0]
            # Close aggressively; maintain a small lateral component to avoid
            # becoming a stationary target. Do not invoke low-HP recovery.
            radial = clamp((dd - (24.0 if "lightning" in held else 42.0)) / 20.0, 0.25, 1.0)
            add((dx, dy), radial)
            add((-dy, dx), 0.65 * self.strafe_sign(enemy, x, y, dd))
            mode = "HUNT"
            if me["energy"] > 12 and dd > 35:
                boost_wanted = True

        # ---- dodging
        for d, e in enemies:
            if e.get("charging"):
                to_me = math.atan2(y - e["pos"][1], x - e["pos"][0])
                if abs(wrap(to_me - e["aim"][0])) < 0.2:
                    ux, uy = unit(e["pos"][0] - x, e["pos"][1] - y)
                    add((-uy, ux), 2.5 * self.sign)
                    if me["energy"] > 20:
                        boost_wanted = True
        cx, cy, cz = x, y, z + 4.5
        for p in obs["projectiles"]:
            vx, vy, vz = p["vel"]
            if p["kind"] == "mortar":
                disc = vz * vz + 44.0 * (p["pos"][2] - z)
                if disc < 0:
                    continue
                tt = (vz + math.sqrt(disc)) / 22.0
                if 0 < tt < 2.5:
                    ix, iy = p["pos"][0] + vx * tt, p["pos"][1] + vy * tt
                    if hdist((ix, iy), pos) < 12:
                        add(unit(x - ix, y - iy), 1.5)
                continue
            v2 = vx * vx + vy * vy + vz * vz
            if v2 < 1:
                continue
            rx, ry, rz = p["pos"][0] - cx, p["pos"][1] - cy, p["pos"][2] - cz
            tc = -(rx * vx + ry * vy + rz * vz) / v2
            if 0 < tc < 0.8:
                qx, qy, qz = rx + vx * tc, ry + vy * tc, rz + vz * tc
                if math.sqrt(qx * qx + qy * qy + qz * qz) < 5.0:
                    nx_, ny_ = unit(-vy, vx)
                    sgn = -1.0 if (nx_ * qx + ny_ * qy) > 0 else 1.0
                    add((nx_, ny_), 1.5 * sgn)

        # ---- stuck handling
        stuck = False
        if tick < self.stuck_until:
            stuck = True
        elif len(self.hist) >= 10:
            h0 = self.hist[-10]
            if hdist((h0[1], h0[2]), pos) < 1.5 and min(hh[3] for hh in self.hist[-10:]) > 0.4:
                ang = math.atan2(self.last_desired[1], self.last_desired[0]) + self.rng.choice((-1, 1)) * self.rng.uniform(1.2, 2.2)
                self.stuck_dir = (math.cos(ang), math.sin(ang))
                self.stuck_until = tick + 8
                self.path = None
                self.hist = []
                stuck = True
        if stuck:
            move = self.stuck_dir
            act["jump"] = True
            if me["energy"] > 10:
                boost_wanted = True
        else:
            move = self.steer(x, y, desired, zone["center"], zone["radius"] if me["in_zone"] else 0)
        mag = math.hypot(move[0], move[1])
        if mag < 0.12:
            move = (0.0, 0.0)
            mag = 0.0
        else:
            self.last_desired = unit(desired[0], desired[1]) if not stuck else self.last_desired
        self.hist.append((tick, x, y, mag if not hold_ring else 0.0))
        self.hist = self.hist[-12:]

        act["move"] = [move[0], move[1]]
        act["boost"] = bool(boost_wanted and mag > 0 and me["energy"] > 5)
        if aim is None and mag > 0.3:
            aim = [math.atan2(move[1], move[0]), 0.0]
        if aim is not None:
            act["aim"] = aim
        if fire[0] or fire[1]:
            act["fire"] = fire
        gun = me["weapons"][held["gun"]] if "gun" in held else None
        if gun and not (tgt and tgt[0] < 120) and gun.get("reload", 0) <= 0 and gun.get("ammo", 99) <= 3:
            act["reload"] = True

        # ---- debug drawing (only your team sees it)
        if tgt:
            mf.draw.line([x, y, z + 7], [tgt[1]["pos"][0], tgt[1]["pos"][1], tgt[1]["pos"][2] + 4.5], "#ff5040")
        if goal_pt is not None:
            mf.draw.point([goal_pt[0], goal_pt[1], z], "#40c0ff", 6)
        mf.draw.label("%s hp%.0f e%d" % (mode, me["hp"], n_close))
        if FEATURES["record"] and tick % 2 == 0:
            line = "%d %s hp%.0f en%.0f p%.0f,%.0f t%s d%s n%d los%d r%s" % (
                tick, mode, me["hp"], me["energy"], x, y, tgt[1]["id"] if tgt else "-", "%.0f" % tgt[0] if tgt else "-", n_close, int(los_clear), self.recover_state[:1])
            mf.draw.text(line[:78])
            if os.environ.get("MECH_LOG"):
                mf.log(line)
        return act


if __name__ == "__main__":
    mf.run(Bot)
