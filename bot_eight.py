"""bot_eight.py -- your bot_six, kept intact, plus a set of low-risk additions.

Practice scoring = 16 - place = roughly the number of house bots you kill before you die (the squad of 15
never fights itself and hunts you). So this bot tries to (1) kill faster, (2) get stronger earlier, (3) survive
the fights it cannot win, and (4) never waste time on dead ends.

EVERYTHING FROM bot_six IS UNCHANGED: every old FEATURES flag keeps the value you had (early_loot, lowhp_hold and
rear_circle stay OFF; flee_hp 0.13; kite thresholds; focus fire; cube_range 70; ...). New behaviour is additive and
each new piece is fail-safe: if any new code raises, the bot drops into SAFE MODE (all new features off, i.e. plain
bot_six behaviour) for 10 s and carries on. If the map model proves wrong (damage from enemies it says cannot
see us), the line-of-sight features switch off by themselves.

NEW
  grid        same walkability as bot_six (verified equal), built ~10x faster with a scanline fill, plus eye-level
              solids (for line of sight) and building corners (for cover). Falls back to bot_six behaviour if it fails.
  sorted      enemies sorted by distance (bot_six checked "nearest enemy < 30 m" on an unsorted list).
  chase_path  chasing an enemy behind a building uses A* around it (bot_six pushed straight at the wall).
  los_gate    never holds the trigger without line of sight (no more shots into walls; no wasted laser charges).
  cover_retreat  when bot_six would kite/flee (same triggers), run to the nearest cell no visible enemy can see
              instead of straight away over open ground, then AMBUSH: stand still facing the enemy (regenerating
              45 HP/s after 5 s unhurt, reloading), hop again if hit while hidden. Cap 20 s.
  kill_loot   for 8 s after a kill: grab cubes within 35 m (a dead mech drops all its cubes + 2) and weapons within
              40 m even with enemies around.
  upgrades    chooses upgrades from the real effect paths in hello.config.upgrades (damage, fire rate, armor, HP,
              regen, turn rate...) with bot_six's keyword order as fallback.
  zone_centre in the last circles aim nearer the centre (the centre is the last place the zone reaches).
  reload_hidden  reload while behind cover.
  logging     START (grid facts), state changes, damage bursts, 20 s SUMMARY on stderr; a JSON decision record in
              the debug drawing (decode with analyze_log.py).
"""
import heapq
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
TPS = 20
NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
      (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
RANK = {"laser": 3.0, "lightning": 2.0, "mortar": 1.0, "gun": 9.0}
UPGRADE_PRIORITY = ["armor", "plating", "shield", "hp", "health", "regen", "repair",
                    "gun_rate", "rate", "damage", "speed", "boost", "energy", "sensor"]

FEATURES = {
    # ---- bot_six flags, values exactly as you had them
    "focus": True, "focus_stick": 1.35, "early_loot": False, "loot_range": 130.0,
    "lowhp_hold": False, "lowhp": 0.45, "kite": True, "kite_hp_crowd": 0.55, "kite_hp_one": 0.25,
    "kite_exit": 0.60, "idle_reload": True, "cube_range": 70.0, "rear_circle": False, "lightning_pref": True,
    "energy_reserve": True, "capture_fix": True, "reload_fix": True, "hunt": True, "cube_in_fight": True,
    "flee_hp": 0.13, "grid_budget": 10.0,
    # ---- new in bot_eight
    "sorted": True,           # enemies sorted by distance
    "chase_path": True,       # A* around buildings when chasing a target we cannot walk straight to
    "los_gate": True,         # no trigger without line of sight (self-checking)
    "cover_retreat": True,    # kite/flee to a cell nobody can see, then ambush-regenerate
    "ambush_cap_s": 20.0,     # longest ambush hold per retreat
    "kill_loot": True,        # after a kill: cubes within 35 m, weapons within 40 m
    "kill_loot_s": 8.0,
    "upgrade_by_effect": True,
    "zone_centre": True,      # smaller hide factor in the last circles
    "reload_hidden": True,
    "safe_mode_s": 10.0,      # seconds with every new feature off after an unexpected exception
    "record": True,
}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def wrap(a):
    return (a + math.pi) % TAU - math.pi


def unit(vx, vy):
    n = math.hypot(vx, vy)
    return (vx / n, vy / n) if n > 1e-9 else (0.0, 0.0)


def hdist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def angles(o, p):
    dx, dy, dz = p[0] - o[0], p[1] - o[1], p[2] - o[2]
    return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))


def lead(o, p, v, speed, extra=0.1):
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


# ----------------------------------------------------------------------------------------------
# Map grid (4 m cells). `inf` (obstacles grown one cell) is exactly bot_six's walkability. Extra: lo/hi = lowest
# base / highest top of any solid in the cell (line of sight), and building corner cells (cover candidates).
# ----------------------------------------------------------------------------------------------
class Grid:
    CELL = 4.0

    def __init__(self, map_dir, budget=8.0):
        t0 = time.perf_counter()
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
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
        N = W * H
        blk = bytearray(N)
        lo = array("f", [1e9]) * N
        hi = array("f", [-1e9]) * N
        self.lo, self.hi = lo, hi
        minx, miny = self.minx, self.miny
        self.truncated = False
        self.n_prisms = 0
        corner_pts = []

        for pi, p in enumerate(world.get("prisms", [])):
            if (pi & 7) == 0 and time.perf_counter() - t0 > budget:
                self.truncated = True
                break
            ring = p.get("ring")
            if not ring or len(ring) < 6:
                continue
            self.n_prisms += 1
            xs, ys = ring[0::2], ring[1::2]
            base = p.get("base", 0.0)
            top = p.get("top", base + 10.0)
            walk_block = (base - self.h(xs[0], ys[0])) < 9.5     # a 9 m mech walks under higher bridges
            edges = []
            for rx, ry in [(xs, ys)] + [(hh[0::2], hh[1::2]) for hh in (p.get("holes") or []) if len(hh) >= 6]:
                m_ = len(rx)
                for i in range(m_):
                    edges.append((rx[i], ry[i], rx[(i + 1) % m_], ry[(i + 1) % m_]))
            gx0 = max(0, int((min(xs) - minx) / C))
            gx1 = min(W - 1, int((max(xs) - minx) / C))
            gy0 = max(0, int((min(ys) - miny) / C))
            gy1 = min(H - 1, int((max(ys) - miny) / C))
            for gy in range(gy0, gy1 + 1):         # scanline fill (even-odd over outer ring + holes)
                yc = miny + (gy + 0.5) * C
                cr = []
                for (x1, y1, x2, y2) in edges:
                    if (y1 > yc) != (y2 > yc):
                        cr.append(x1 + (yc - y1) * (x2 - x1) / (y2 - y1))
                cr.sort()
                for k in range(0, len(cr) - 1, 2):
                    a = max(gx0, int(math.ceil((cr[k] - minx) / C - 0.5)))
                    b = min(gx1, int(math.floor((cr[k + 1] - minx) / C - 0.5)))
                    for gx in range(a, b + 1):
                        i = gy * W + gx
                        if base < lo[i]:
                            lo[i] = base
                        if top > hi[i]:
                            hi[i] = top
                        if walk_block:
                            blk[i] = 1
            n = len(xs)
            for i in range(n):  # thin walls: also mark the outline
                ax, ay, bx, by = xs[i], ys[i], xs[(i + 1) % n], ys[(i + 1) % n]
                m = max(1, int(math.hypot(bx - ax, by - ay) / 2))
                for s in range(m + 1):
                    gx = int((ax + (bx - ax) * s / m - minx) / C)
                    gy = int((ay + (by - ay) * s / m - miny) / C)
                    if 0 <= gx < W and 0 <= gy < H:
                        k = gy * W + gx
                        if base < lo[k]:
                            lo[k] = base
                        if top > hi[k]:
                            hi[k] = top
                        if walk_block:
                            blk[k] = 1
            if walk_block and top - base >= 6.0 and n >= 3:
                corner_pts.extend(self._corners(xs, ys))

        for gy in range(H):  # slopes too steep to walk
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

        inf = bytearray(blk)  # grow every obstacle by one cell (the mech is 2.5 m wide on each side)
        for gy in range(H):
            for gx in range(W):
                if blk[gy * W + gx]:
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            a, b = gx + dx, gy + dy
                            if 0 <= a < W and 0 <= b < H:
                                inf[b * W + a] = 1
        self.inf = inf
        self.blocked_frac = sum(inf) / float(N)
        if self.blocked_frac > 0.5:
            raise ValueError("grid looks wrong (over half blocked): navigating without it")
        seen, cells = set(), []
        for (px, py) in corner_pts:
            gx = int((px - minx) / C)
            gy = int((py - miny) / C)
            if 0 <= gx < W and 0 <= gy < H:
                k = gy * W + gx
                if not inf[k] and k not in seen:
                    seen.add(k)
                    cells.append(self.center(k))
        self.corners = cells
        self.cbuckets = {}
        for (cx, cy) in cells:
            self.cbuckets.setdefault((int(cx // 40), int(cy // 40)), []).append((cx, cy))
        self.build_s = time.perf_counter() - t0

    @staticmethod
    def _corners(xs, ys):
        """Candidate standing points around the convex corners of one building footprint."""
        n = len(xs)
        area = 0.0
        for i in range(n):
            area += xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
        s = 1.0 if area > 0 else -1.0
        out = []
        for i in range(n):
            px, py = xs[i - 1], ys[i - 1]
            vx, vy = xs[i], ys[i]
            qx, qy = xs[(i + 1) % n], ys[(i + 1) % n]
            l1, l2 = math.hypot(vx - px, vy - py), math.hypot(qx - vx, qy - vy)
            if l1 < 3.0 or l2 < 3.0:
                continue
            e1 = ((vx - px) / l1, (vy - py) / l1)
            e2 = ((qx - vx) / l2, (qy - vy) / l2)
            if (e1[0] * e2[1] - e1[1] * e2[0]) * s < 0.64:      # keep convex corners sharper than ~40 degrees
                continue
            n1 = (s * e1[1], -s * e1[0])
            n2 = (s * e2[1], -s * e2[0])
            bis = unit(n1[0] + n2[0], n1[1] + n2[1])
            for o in (8.0, 11.0):
                out.append((vx + o * n1[0], vy + o * n1[1]))
                out.append((vx + o * n2[0], vy + o * n2[1]))
                out.append((vx + o * bis[0], vy + o * bis[1]))
                out.append((vx + o * n1[0] - 7.0 * e1[0], vy + o * n1[1] - 7.0 * e1[1]))
                out.append((vx + o * n2[0] + 7.0 * e2[0], vy + o * n2[1] + 7.0 * e2[1]))
        return out

    def corners_near(self, x, y, R, limit=60):
        out = []
        R2 = R * R
        for bx in range(int((x - R) // 40), int((x + R) // 40) + 1):
            for by in range(int((y - R) // 40), int((y + R) // 40) + 1):
                for (cx, cy) in self.cbuckets.get((bx, by), ()):
                    d2 = (cx - x) ** 2 + (cy - y) ** 2
                    if d2 <= R2:
                        out.append((d2, cx, cy))
        out.sort()
        return [(cx, cy) for _, cx, cy in out[:limit]]

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

    def los(self, a, b, skip=5.0, tail=3.0, step=3.0):
        """Can a bullet/beam fly from 3D point a to 3D point b? (trees never stop shots)"""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        L = math.hypot(dx, dy)
        if L < skip + tail:
            return True
        s = skip
        lo, hi = self.lo, self.hi
        while s <= L - tail:
            f = s / L
            x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            i = self.idx(x, y)
            if lo[i] <= z <= hi[i] or z < self.h(x, y) - 0.5:
                return False
            s += step
        return True

    def arc_clear(self, o, yaw, pit, v, g, hd):
        vx, vy, vz = v * math.cos(pit) * math.cos(yaw), v * math.cos(pit) * math.sin(yaw), v * math.sin(pit)
        t = 0.25
        while t < 14:
            x, y, z = o[0] + vx * t, o[1] + vy * t, o[2] + vz * t - 0.5 * g * t * t
            if math.hypot(x - o[0], y - o[1]) > hd - 8:
                return True
            i = self.idx(x, y)
            if self.lo[i] <= z <= self.hi[i] or z < self.h(x, y) - 0.5:
                return False
            t += 0.25
        return True

    def find_path(self, sx, sy, tx, ty, max_exp=1500, tlimit=0.012):
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
            hi_ = f - c
            if hi_ < besth:
                best, besth = i, hi_
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
                    continue       # a diagonal step would clip a building's corner
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


# ----------------------------------------------------------------------------------------------
# The bot
# ----------------------------------------------------------------------------------------------
class Bot:
    name = "bot-eight"

    def __init__(self, hello):
        cfg = hello["config"]
        self.reach = cfg["loot"]["pickup_radius"]
        self.mortar_g = cfg.get("weapons", {}).get("mortar", {}).get("gravity", 22.0)
        self.rng = random.Random(hello.get("seed", 0))
        self.log_all = bool(os.environ.get("MECH_LOG"))
        self.upg = {}
        try:
            for u in cfg.get("upgrades", []) or []:
                self.upg[u.get("id")] = u
        except Exception:
            self.upg = {}
        self.grid = None
        t0 = time.perf_counter()
        try:
            self.grid = Grid(hello["map"]["dir"], budget=FEATURES["grid_budget"])
            g = self.grid
            mf.log("START %s slot=%s GRID ok %dx%d prisms=%d corners=%d blocked=%.2f truncated=%s build=%.2fs" % (
                self.name, hello.get("slot"), g.W, g.H, g.n_prisms, len(g.corners), g.blocked_frac, g.truncated, time.perf_counter() - t0))
        except Exception:
            mf.log("START %s GRID FAILED after %.1fs, steering without map:\n%s" % (self.name, time.perf_counter() - t0, traceback.format_exc()))
        self.T0 = time.perf_counter()
        # bot_six state
        self.sign = 1
        self.flip_at = 0
        self.hist = []
        self.stuck_until = -1
        self.stuck_dir = (0.0, 0.0)
        self.stuck_events = 0
        self.path = None
        self.pi = 0
        self.path_goal = None
        self.path_tick = -99
        self.hit_tick = -999
        self.hit_bearing = 0.0
        self.tgt_id = None
        self.kiting = False
        self.last_ang = None
        self.last_desired = (1.0, 0.0)
        # bot_eight state
        self.safe_until = -1
        self.errors = 0
        self.hit_tick_enemy = -999
        self.last_kill = -999
        self.kite_since = -999
        self.cover = None
        self.cover_next = -1
        self.hops = 0
        self.last_known = None
        self.last_seen_tick = -999
        self.los_miss = 0
        self.los_miss_tick = -999
        self.los_bad_until = -1
        self.state = "INIT"
        self.state_ticks = {}
        self.n_act = 0
        self.ms_sum = 0.0
        self.ms_max = 0.0
        self.dbg_bytes = 0
        self.dmg_burst_tick = -999
        self.tm = {"dealt": 0.0, "taken": 0.0, "kills": 0, "retreats": 0, "ambush_ticks": 0, "chase_paths": 0, "gated": 0}

    # ---------------- navigation ----------------
    def elapsed(self):
        return time.perf_counter() - self.T0

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
                if self.elapsed() < 0.026:
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

    def zone_goal(self, obs, me, x, y, t, speed, new_on):
        z = obs["zone"]
        c, r, nc, nr = z["center"], z["radius"], z["next_center"], z["next_radius"]
        f = 0.6
        if new_on and FEATURES["zone_centre"]:
            f = 0.6 if nr >= 140 else (0.45 if nr >= 70 else (0.3 if nr >= 25 else 0.12))
        if math.hypot(x - c[0], y - c[1]) > r - 6 or not me["in_zone"]:
            return self.inside_point(nc, nr, x, y, f), 2
        d_next = math.hypot(x - nc[0], y - nc[1])
        if d_next > nr - 8:
            travel = (d_next - nr * f) / max(speed, 5.0)
            if t >= z["shrink_start"] or z["shrink_start"] - t < travel + 15:
                return self.inside_point(nc, nr, x, y, f), 1
        return None, 0

    # ---------------- weapons (bot_six shoot() + line-of-sight gate) ----------------
    def shoot(self, me, tpos, tvel, special, los_t=True, use_arc=False):
        x, y, z = me["pos"]
        mh = me["stats"].get("muzzle_height", 7.0)
        o = (x, y, z + mh)
        cur = me["aim"]
        tc = (tpos[0], tpos[1], tpos[2] + 4.5)
        d = hdist(tpos, (x, y))
        held = {}
        for i, w in enumerate(me["weapons"]):
            if w:
                held[w["kind"]] = (i, w)
        cand = {}
        if "gun" in held:
            i, w = held["gun"]
            st = w["stats"]
            yaw, pit = angles(o, lead(o, tc, tvel, st.get("speed", 252)))
            ok = d < st.get("range", 230) * 0.92 and w.get("ammo", 1) > 0 and w.get("reload", 0) <= 0 and los_t
            cand["gun"] = (yaw, pit, ok, i)
        if special and "laser" in held:
            i, w = held["laser"]
            yaw, pit = angles(o, lead(o, tc, tvel, 1e9, 0.25))
            ok = d < 170 and (w["cooldown"] <= 0.35 or w.get("charge", 0) > 0) and los_t
            cand["laser"] = (yaw, pit, ok, i)
        if special and "lightning" in held:
            i, w = held["lightning"]
            yaw, pit = angles(o, tc)
            ok = d < w["stats"].get("range", 45) - 3 and me["energy"] > 12 and los_t
            cand["lightning"] = (yaw, pit, ok, i)
        if special and "mortar" in held:
            i, w = held["mortar"]
            st = w["stats"]
            v, g = st.get("speed", 72), st.get("gravity", 22)
            o2 = (x, y, z + mh + 2.5)
            pred, pit = tc, None
            for _ in range(3):
                pit = mortar_pitch(hdist(pred, o2), pred[2] - o2[2], v, g)
                if pit is None:
                    break
                tf = hdist(pred, o2) / (v * math.cos(pit))
                pred = (tc[0] + tvel[0] * (tf + 0.1), tc[1] + tvel[1] * (tf + 0.1), tc[2])
            if pit is not None:
                yaw = math.atan2(pred[1] - y, pred[0] - x)
                slow = math.hypot(tvel[0], tvel[1]) < 5 or d < 80
                clear = True
                if use_arc and self.grid is not None:
                    clear = self.grid.arc_clear(o2, yaw, pit, v, g, hdist(pred, o2))
                cand["mortar"] = (yaw, pit, 22 < d < 200 and w["cooldown"] <= 0.5 and slow and clear, i)
        primary = None
        for k in ("lightning", "laser", "mortar", "gun"):
            if k in cand and cand[k][2]:
                primary = k
                break
        if primary is None:
            primary = "gun" if "gun" in cand else (next(iter(cand)) if cand else None)
        if primary is None:
            yaw, pit = angles(o, tc)
        else:
            yaw, pit = cand[primary][0], cand[primary][1]
        fire = [False, False]
        tol0 = clamp(math.atan2(2.3, max(d, 5.0)) + 0.01, 0.025, 0.22)
        for k, (cy, cp, ok, i) in cand.items():
            tol = tol0
            if k == "laser" and held["laser"][1].get("charge", 0) > 0:
                tol = tol0 * 1.8
            ye, pe = abs(wrap(cur[0] - cy)), abs(cur[1] - cp)
            if k == "lightning":
                aligned = ye < 0.4
            else:
                aligned = ye < tol and pe < tol * 1.6
            if k == "laser":
                ch = held["laser"][1].get("charge", 0)
                if ch >= 0.7:
                    continue
                if ch > 0 and (d > 190 or not ok and ch < 0.15):
                    continue
            if ok and aligned:
                fire[i] = True
        return [yaw, pit], fire

    def strafe_sign(self, e, x, y, d):
        if FEATURES["rear_circle"] and d < 70 and e.get("aim"):
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

    # ---------------- upgrades ----------------
    EFFECT_W = [("armor", 3.0), ("taken", 3.0), ("max_hp", 2.8), ("regen", 2.2), ("damage", 3.0), ("cooldown", 2.8),
                ("rate", 2.8), ("splash", 1.8), ("chain", 2.0), ("charge", 1.8), ("reload", 1.8), ("mag", 1.5),
                ("turn", 1.8), ("energy", 1.8), ("speed", 1.4), ("jet", 1.0), ("range", 1.2), ("knock", 1.0),
                ("sens", 0.8), ("vision", 0.8), ("sound", 0.8), ("hp", 2.8)]

    def pick_upgrade(self, offer, held, new_on):
        best_i, best_s = 0, -1e9
        for i, ch in enumerate(offer["choices"]):
            key = (ch.get("id", "") + " " + ch.get("name", "")).lower()
            ks = None
            for r, word in enumerate(UPGRADE_PRIORITY):       # bot_six order, as a score from 3.2 downwards
                if word in key:
                    ks = 3.2 - 0.15 * r
                    break
            es = None
            if new_on and FEATURES["upgrade_by_effect"]:
                u = self.upg.get(ch.get("id"))
                if u and u.get("effects"):
                    es = 0.0
                    for ef in u["effects"]:
                        path = (ef.get("path") or "").lower()
                        w = 1.0
                        for tok, wt in self.EFFECT_W:
                            if tok in path:
                                w = wt
                                break
                        if any(k in path for k in held):
                            w += 0.3
                        es = max(es, w)
            sc = es if es is not None else (ks if ks is not None else 1.0)
            if sc > best_s:
                best_i, best_s = i, sc
        return best_i

    # ---------------- cover ----------------
    def zone_ok(self, cx, cy, zone, nc, nr):
        return hdist((cx, cy), zone["center"]) <= zone["radius"] - 5 and hdist((cx, cy), nc) <= nr * 0.97

    def find_cover(self, x, y, eyes, zone, nc, nr, tl=0.008, away_from=None, gain=0.0):
        """Nearest walkable cell, straight-line reachable, that none of the enemy eye points can see."""
        g = self.grid
        if g is None or not eyes:
            return None
        t0 = time.perf_counter()
        mx = sum(e[0] for e in eyes) / len(eyes)
        my = sum(e[1] for e in eyes) / len(eyes)
        away = unit(x - mx, y - my)
        cands = [c for c in g.corners_near(x, y, 40.0, 40) if hdist(c, (x, y)) >= 5.0]
        base = self.rng.random() * TAU / 12
        for r in (8.0, 14.0, 22.0, 32.0):
            for k in range(12):
                a = base + k * TAU / 12
                cands.append((x + math.cos(a) * r, y + math.sin(a) * r))
        scored = []
        for (cx, cy) in cands:                    # cheap filters + score first, expensive line-of-sight tests best-first
            if g.blocked(cx, cy) or not self.zone_ok(cx, cy, zone, nc, nr):
                continue
            r = hdist((cx, cy), (x, y))
            if r > 45.0:
                continue
            if away_from is not None and hdist((cx, cy), away_from) < hdist((x, y), away_from) + gain:
                continue
            scored.append((r - 8.0 * (((cx - x) * away[0] + (cy - y) * away[1]) / max(r, 1.0)), cx, cy, r))
        scored.sort()
        for sc, cx, cy, r in scored:
            if time.perf_counter() - t0 > tl:
                break
            if not g.ray_free(x, y, cx, cy, 3.0):
                continue
            tgt = (cx, cy, g.h(cx, cy) + 4.5)
            if any(g.los(e, tgt, 5.0, 2.0, 3.0) for e in eyes):
                continue
            return (cx, cy, r)
        return None

    def retreat_policy(self, C):
        """kite/flee to a cell no visible enemy can see, then ambush-regenerate. None -> bot_six's run-away code."""
        tick, x, y = C["tick"], C["x"], C["y"]
        threats, me, hpf = C["threats"], C["me"], C["hpf"]
        zone, nc, nr = C["zone"], C["nc"], C["nr"]
        eyes = [(e["pos"][0], e["pos"][1], e["pos"][2] + 7.0) for d, e in threats if d < 220]
        compromised = 0 <= tick - self.hit_tick_enemy <= 10
        if not eyes and compromised and self.hit_bearing is not None:
            eyes = [(x + math.cos(self.hit_bearing) * 80.0, y + math.sin(self.hit_bearing) * 80.0, C["z"] + 7.0)]
        lk = self.last_known
        recent = lk is not None and tick - self.last_seen_tick < 60 and hdist((x, y), lk) < 90.0
        if self.cover is not None and hdist((x, y), self.cover[:2]) < 2.5:
            self.cover = None
        if threats or compromised or (recent and self.hops < 2):
            if self.cover is None and tick >= self.cover_next and self.elapsed() < 0.024:
                breaking = not (threats or compromised)
                eyes2 = eyes if eyes else ([(lk[0], lk[1], C["z"] + 7.0)] if lk else [])
                self.cover = self.find_cover(x, y, eyes2, zone, nc, nr, away_from=(lk if breaking else None),
                                             gain=(12.0 if breaking else 0.0))
                self.cover_next = tick + 6
                if breaking:
                    self.hops += 1
                if self.cover is not None:
                    self.tm["retreats"] += 1
            if self.cover is not None:
                cx, cy, cr = self.cover
                if threats and min(d for d, e in threats) < 22.0 and cr > 25.0:
                    return None                              # an enemy on top of us and the cover is far: bot_six code decides
                wp = self.waypoint(x, y, (cx, cy), tick)
                return {"mode": "RETREAT" if (threats or compromised) else "BREAK",
                        "vec": [v * 1.2 for v in unit(wp[0] - x, wp[1] - y)],
                        "boost": me["energy"] > 25 and hpf < 0.4, "goal": (cx, cy)}
            if threats or compromised:
                return None
        # hidden: stand still, facing the enemy, regenerating
        if tick - self.kite_since < int(FEATURES["ambush_cap_s"] * TPS):
            self.tm["ambush_ticks"] += 1
            return {"mode": "AMBUSH", "vec": [0.0, 0.0], "boost": False, "goal": None}
        return None

    # ---------------- logging ----------------
    def note_damage(self, tick, t, dm, me, pos, hidden):
        if tick - self.dmg_burst_tick > 40:
            mf.log("DAMAGE t=%.1f k=%d hp=%d pos=(%d,%d) hidden=%d %s" % (t, tick, me["hp"], pos[0], pos[1], 1 if hidden else 0, json.dumps(dm, separators=(",", ":"))))
        self.dmg_burst_tick = tick

    def draw(self, tick, rec, tgt, goal_pt, x, y, z, notable):
        d = mf.draw
        d.label("%s hp%d" % (rec["s"], rec["hp"]))
        if tgt:
            d.line([x, y, z + 7], [tgt[1]["pos"][0], tgt[1]["pos"][1], tgt[1]["pos"][2] + 4.5], "#ff5040")
        if goal_pt is not None:
            d.point([goal_pt[0], goal_pt[1], z], "#40c0ff", 6)
        if FEATURES["record"] and (tick % 2 == 0 or notable):
            if self.dbg_bytes > 6.0e6 and not rec.get("dm"):
                return
            s = json.dumps(rec, separators=(",", ":"))
            n = 0
            for i in range(0, min(len(s), 12 * 70), 70):
                d.text("#%d.%d:%s" % (tick, i // 70, s[i:i + 70]))
                n += 1
            self.dbg_bytes += len(s) + 24 * n

    # ---------------- main decision ----------------
    def act(self, obs):
        try:
            return self._act(obs, tick_new_on=obs["tick"] >= self.safe_until)
        except Exception:
            self.errors += 1
            if self.errors <= 5 or self.errors % 50 == 0:
                mf.log("ACT ERROR #%d tick %s -> SAFE MODE (bot_six behaviour) for %.0fs:\n%s" % (
                    self.errors, obs.get("tick"), FEATURES["safe_mode_s"], traceback.format_exc()))
            self.safe_until = obs["tick"] + int(FEATURES["safe_mode_s"] * TPS)
            self.cover = None
            try:
                return self._act(obs, tick_new_on=False)
            except Exception:
                mf.log("ACT ERROR in safe mode too:\n" + traceback.format_exc())
                return {"tick": obs["tick"], "move": [0.0, 0.0]}

    def _act(self, obs, tick_new_on=True):
        self.T0 = time.perf_counter()
        new_on = tick_new_on
        tick, t = obs["tick"], obs["t"]
        me = obs["self"]
        x, y, z = me["pos"]
        pos = (x, y)
        st = me["stats"]
        walk = st.get("walk_speed", 11.7)
        hpf = me["hp"] / max(1.0, me["max_hp"])
        zone = obs["zone"]
        alive = obs.get("alive", 16)
        g = self.grid
        map_ok = (new_on and g is not None and not g.truncated and tick >= self.los_bad_until)
        act = {"tick": tick}

        # ---- enemies, and who can see us (map line of sight, nearest 5 only)
        enemies = []
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                p3 = math.sqrt((m["pos"][0] - x) ** 2 + (m["pos"][1] - y) ** 2 + (m["pos"][2] - z) ** 2)
                enemies.append((p3, m))
        if new_on and FEATURES["sorted"]:
            enemies.sort(key=lambda a: a[0])
        vis = {}
        if map_ok:
            eye_me = (x, y, z + 4.5)
            for i, (d, e) in enumerate(enemies):
                if d > 240:
                    break
                if i < 5 and self.elapsed() < 0.015:
                    vis[e["id"]] = g.los((e["pos"][0], e["pos"][1], e["pos"][2] + 7.0), eye_me, 5.0, 3.0, 3.0)
                else:
                    vis[e["id"]] = d < 60
        threats = [(d, e) for d, e in enemies if vis.get(e["id"]) and d < 220] if map_ok else []
        if threats:
            self.last_known = (threats[0][1]["pos"][0], threats[0][1]["pos"][1])
            self.last_seen_tick = tick
        hidden = map_ok and not threats

        # ---- events
        dm = []
        for ev in obs["events"]:
            ty = ev.get("type")
            if ty == "damaged":
                w_, amt = ev.get("weapon"), ev.get("amount", 0)
                dm.append([w_, round(amt), 1 if ev.get("rear") else 0])
                self.hit_tick = tick
                if ev.get("bearing") is not None:
                    self.hit_bearing = ev["bearing"]
                if w_ != "zone":
                    self.hit_tick_enemy = tick
                    self.tm["taken"] += amt
                    by = ev.get("by")
                    if map_ok and w_ in ("gun", "laser", "lightning") and by in vis and not vis[by]:
                        if tick - self.los_miss_tick > 300:
                            self.los_miss = 0
                        self.los_miss += 1
                        self.los_miss_tick = tick
                        mf.log("LOSCHK t=%.1f %s hit us for %d but the map says no line of sight (miss #%d)" % (t, w_, amt, self.los_miss))
                        if self.los_miss >= 4:
                            self.los_bad_until = tick + 600
                            mf.log("LOSCHK map line-of-sight model looks wrong: LOS features OFF for 30 s")
                if self.rng.random() < 0.5:
                    self.sign = -self.sign
            elif ty == "hit":
                self.tm["dealt"] += ev.get("amount", 0)
            elif ty == "kill":
                self.last_kill = tick
                self.tm["kills"] += 1
                mf.log("KILL t=%.1f k=%d hp=%d alive=%d" % (t, tick, me["hp"], alive))
        if dm:
            self.note_damage(tick, t, dm, me, pos, hidden)

        held = {w["kind"]: i for i, w in enumerate(me["weapons"]) if w}

        # upgrade choice
        offer = obs["offer"]
        if offer:
            act["pick"] = self.pick_upgrade(offer, held, new_on)

        if tick >= self.flip_at:
            self.sign = -self.sign
            self.flip_at = tick + self.rng.randint(6, 18)

        npcs = [(hdist(m["pos"], pos), m) for m in obs["mechs"] if m["kind"] == "npc"]
        crates = [(hdist(c["pos"], pos), c) for c in obs["crates"]]
        near_enemies = [(d, m) for d, m in enemies if d < 130]
        n_close = len(near_enemies)
        flee = n_close >= 3 or (hpf < 0.3 and n_close >= 1)
        killing_spree = new_on and FEATURES["kill_loot"] and (tick - self.last_kill) < FEATURES["kill_loot_s"] * TPS

        # ---- pick up a weapon we are standing next to
        free_slot = next((i for i, w in enumerate(me["weapons"]) if w is None), None)
        worst_slot, worst_rank = None, 99.0
        for i, w in enumerate(me["weapons"]):
            if w and RANK.get(w["kind"], 0) < worst_rank and w.get("charge", 0) <= 0:
                worst_slot, worst_rank = i, RANK.get(w["kind"], 0)
        want_rank = 0.0 if free_slot is not None else worst_rank
        loot_wp = None
        best_loot = -1e9
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
            if FEATURES["early_loot"]:
                if d < FEATURES["loot_range"] and (loot_wp is None or r * 50 - d > best_loot):
                    best_loot = r * 50 - d
                    loot_wp = (d, p["pos"])
            elif d < 70 and (loot_wp is None or d < loot_wp[0]):
                loot_wp = (d, p["pos"])

        # ---- targets
        tgt = None
        if enemies:
            sc_ = lambda e: e[0] * (0.6 + 0.8 * (e[1].get("hp", 1000) / max(1.0, e[1].get("max_hp", 1000))))
            tgt = min(enemies, key=sc_)
            if FEATURES["focus"] and self.tgt_id is not None:
                prev = next((e for e in enemies if e[1].get("id") == self.tgt_id), None)
                if prev is not None and sc_(prev) <= sc_(tgt) * FEATURES["focus_stick"]:
                    tgt = prev
            self.tgt_id = tgt[1].get("id")
        farm = None
        if tgt is None and hpf > 0.5:
            opts = [(d, m, "npc") for d, m in npcs if d < 90] + [(d, c, "crate") for d, c in crates if d < 60]
            if opts:
                farm = min(opts, key=lambda o: o[0])

        mh = st.get("muzzle_height", 7.0)
        gate_on = map_ok and FEATURES["los_gate"]

        def los_to(p3):
            if not gate_on or self.elapsed() > 0.03:
                return True
            ok = g.los((x, y, z + mh), p3, 5.0, 3.0, 3.0)
            if not ok:
                self.tm["gated"] += 1
            return ok

        aim, fire = None, [False, False]
        if tgt and tgt[0] < 235:
            tp_ = tgt[1]["pos"]
            aim, fire = self.shoot(me, tp_, tgt[1]["vel"], True, los_to((tp_[0], tp_[1], tp_[2] + 4.5)), gate_on)
        elif farm:
            tv = farm[1].get("vel", (0, 0, 0))
            fp = farm[1]["pos"]
            aim, fire = self.shoot(me, fp, tv, False, los_to((fp[0], fp[1], fp[2] + 3.0)), gate_on)
        elif tick - self.hit_tick < 15:
            aim = [self.hit_bearing, 0.0]
        else:
            for s in obs["sounds"]:
                if s["kind"] in ("gun", "mortar", "laser", "lightning", "laser_charge") and s["dist"] != "far" and s["age"] < 0.6:
                    aim = [s["bearing"], 0.0]
                    break

        # ---- where do we want to go?
        zgoal, zurg = self.zone_goal(obs, me, x, y, t, walk, new_on)
        desired = [0.0, 0.0]

        def add(v, w):
            desired[0] += v[0] * w
            desired[1] += v[1] * w

        boost_wanted = False
        mode = "hold"
        hold_ring = False
        goal_pt = None
        engaged = tgt is not None and tgt[0] < 70
        nc, nr = zone["next_center"], zone["next_radius"]

        if zgoal is not None:
            goal_pt = zgoal
            wp = self.waypoint(x, y, zgoal, tick)
            add(unit(wp[0] - x, wp[1] - y), 2.0 if zurg == 2 else 1.0)
            mode = "zone%d" % zurg
            if zurg == 2 and me["energy"] > 8:
                boost_wanted = True
        elif not flee and not engaged:
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
                    if dd < FEATURES["cube_range"] and (cube is None or dd < cube[0]):
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

        # ================= movement policy (the 'agg' policy of bot_six + additions) =================
        C = {"tick": tick, "x": x, "y": y, "z": z, "me": me, "hpf": hpf, "threats": threats, "zone": zone, "nc": nc, "nr": nr}
        if tgt is not None and zgoal is None:
            e = tgt[1]
            d = hdist(e["pos"], pos)
            ux, uy = unit(e["pos"][0] - x, e["pos"][1] - y)

            if FEATURES["kite"]:
                if self.kiting:
                    if (hpf > FEATURES["kite_exit"] and n_close < 3) or n_close == 0:
                        self.kiting = False
                        self.cover, self.hops = None, 0
                elif ((n_close >= 3 and hpf < FEATURES["kite_hp_crowd"]) or (n_close >= 1 and hpf < FEATURES["kite_hp_one"])):
                    self.kiting = True
                    self.kite_since = tick
                    self.hops = 0
            if hpf < FEATURES["flee_hp"] or (FEATURES["kite"] and self.kiting):
                desired = [0.0, 0.0]
                cov = None
                if map_ok and FEATURES["cover_retreat"]:
                    cov = self.retreat_policy(C)
                if cov is not None:
                    desired = list(cov["vec"])
                    mode = cov["mode"]
                    boost_wanted = cov["boost"]
                    goal_pt = cov.get("goal")
                else:
                    fx, fy = 0.0, 0.0
                    for ed, enemy in near_enemies:
                        ax, ay = unit(x - enemy["pos"][0], y - enemy["pos"][1])
                        fx += ax / max(ed, 15.0)
                        fy += ay / max(ed, 15.0)
                    add(unit(fx, fy), 2.0)
                    mode = "FLEE"
                    boost_wanted = me["energy"] > 15
            else:
                desired = [0.0, 0.0]
                preferred = (28.0 if FEATURES["lightning_pref"] else 48.0) if "lightning" in held else 62.0
                radial = clamp((d - preferred) / 25.0, -0.25, 1.0)
                dirx, diry = ux, uy
                if new_on and FEATURES["chase_path"] and g is not None and radial > 0.05 and d < 200 \
                        and not g.ray_free(x, y, e["pos"][0], e["pos"][1], 4.0):
                    wpc = self.waypoint(x, y, (e["pos"][0], e["pos"][1]), tick)    # walk around the building, not into it
                    dirx, diry = unit(wpc[0] - x, wpc[1] - y)
                    self.tm["chase_paths"] += 1
                add((dirx, diry), radial)
                add((-uy, ux), 0.55 * self.strafe_sign(e, x, y, d))
                mode = "FIGHT"
                en_min = 60 if ("lightning" in held and FEATURES["energy_reserve"]) else 25
                if d > 75 and me["energy"] > en_min:
                    boost_wanted = True
                if FEATURES["cube_in_fight"] and hpf > 0.4 and n_close <= (3 if killing_spree else 2):
                    rng_c = 35.0 if killing_spree else 14.0
                    for c in obs["cubes"]:
                        if hdist(c["pos"], pos) < rng_c:
                            wpc = self.waypoint(x, y, c["pos"], tick)
                            add(unit(wpc[0] - x, wpc[1] - y), 1.6 if killing_spree else 1.2)
                            mode = "FIGHT+CUBE"
                            break
                    else:
                        if killing_spree and loot_wp is not None and loot_wp[0] < 40 and n_close <= 1:
                            wpl = self.waypoint(x, y, loot_wp[1], tick)
                            add(unit(wpl[0] - x, wpl[1] - y), 1.6)
                            goal_pt = loot_wp[1]
                            mode = "FIGHT+LOOT"

        elif tgt is None and zgoal is None and hold_ring and FEATURES["capture_fix"]:
            desired = [0.0, 0.0]
            mode = "CAPTURE"

        elif (tgt is None and zgoal is None and new_on and FEATURES["cover_retreat"] and self.kiting
              and hpf < FEATURES["kite_exit"] and tick - self.kite_since < int(FEATURES["ambush_cap_s"] * TPS) and map_ok):
            # the enemies lost us: keep regenerating where we are (hop again if hit), reload, then rejoin the hunt
            cov = self.retreat_policy(C)
            if cov is not None:
                desired = list(cov["vec"])
                mode = cov["mode"]
                boost_wanted = cov["boost"]
                goal_pt = cov.get("goal")
            else:
                self.kiting = False

        elif (tgt is None and zgoal is None and FEATURES["lowhp_hold"] and hpf < FEATURES["lowhp"]
              and math.hypot(desired[0], desired[1]) < 0.1):
            if tick - self.hit_tick < 20:
                desired = [-math.cos(self.hit_bearing), -math.sin(self.hit_bearing)]
                mode = "REGEN_RUN"
                boost_wanted = me["energy"] > 15
            else:
                desired = [0.0, 0.0]
                mode = "REGEN"

        elif tgt is None and zgoal is None:
            if self.kiting:
                self.kiting = False
            if not hasattr(self, "agg_angle"):
                self.agg_angle = self.rng.uniform(-math.pi, math.pi)
                self.agg_turn = tick
            if tick - self.agg_turn > self.rng.randint(45, 85):
                self.agg_angle += self.rng.uniform(-1.4, 1.4)
                self.agg_turn = tick
            if math.hypot(desired[0], desired[1]) < 0.1:
                desired = [math.cos(self.agg_angle), math.sin(self.agg_angle)]
                mode = "SEARCH"
                if FEATURES["hunt"]:
                    hb = None
                    if tick - self.hit_tick < 40:
                        hb = self.hit_bearing
                    else:
                        for s_ in obs["sounds"]:
                            if s_["kind"] in ("gun", "mortar", "laser", "lightning", "laser_charge") and s_["dist"] != "far" and s_["age"] < 1.5:
                                hb = s_["bearing"]
                                break
                    if hb is not None:
                        desired = [math.cos(hb), math.sin(hb)]
                        mode = "HUNT"

        # ---- dodging (bot_six)
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

        # ---- stuck handling (bot_six)
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
                self.stuck_events += 1
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
        if aim is None and mode in ("AMBUSH", "RETREAT", "BREAK") and self.last_known is not None:
            aim = [math.atan2(self.last_known[1] - y, self.last_known[0] - x), 0.0]     # keep the barrel on the corner they come around
        if aim is None and mag > 0.3:
            aim = [math.atan2(move[1], move[0]), 0.0]
        if aim is not None:
            act["aim"] = aim
        if fire[0] or fire[1]:
            act["fire"] = fire
        gun = me["weapons"][held["gun"]] if "gun" in held else None
        if gun and gun.get("reload", 0) <= 0:
            am = gun.get("ammo", 99)
            if FEATURES["reload_fix"] and am <= 0:
                act["reload"] = True
            elif am <= 3 and not (tgt and tgt[0] < 120):
                act["reload"] = True
            elif FEATURES["idle_reload"] and tgt is None and am < 12:
                act["reload"] = True
            elif new_on and FEATURES["reload_hidden"] and mode in ("AMBUSH", "BREAK") and am < 12:
                act["reload"] = True

        # ---- logging
        ms = (time.perf_counter() - self.T0) * 1000
        self.n_act += 1
        self.ms_sum += ms
        self.ms_max = max(self.ms_max, ms)
        self.state_ticks[mode] = self.state_ticks.get(mode, 0) + 1
        if mode != self.state:
            mf.log("t=%.1f k=%d %s->%s hp=%d pos=(%d,%d) al=%d th=%d" % (t, tick, self.state, mode, me["hp"], x, y, alive, len(threats)))
            self.state = mode
        if tick % (20 * TPS) == 0:
            mf.log("SUMMARY t=%.0f hp=%d al=%d pos=(%d,%d) ms_avg=%.1f ms_max=%.1f stuck=%d los_miss=%d safe=%d states=%s tm=%s" % (
                t, me["hp"], alive, x, y, self.ms_sum / max(1, self.n_act), self.ms_max, self.stuck_events, self.los_miss,
                1 if not new_on else 0, json.dumps(self.state_ticks, separators=(",", ":")),
                json.dumps({k: round(v) for k, v in self.tm.items()}, separators=(",", ":"))))
        rec = {"k": tick, "t": round(t, 1), "hp": round(me["hp"]), "e": round(me["energy"]), "p": [round(x), round(y)], "al": alive,
               "s": mode, "th": len(threats), "ne": len(enemies), "kt": 1 if self.kiting else 0,
               "tg": tgt[1]["id"] if tgt else None, "td": round(tgt[0]) if tgt else None,
               "iz": 1 if me["in_zone"] else 0, "ms": round(ms, 1), "h": min(99, tick - self.hit_tick),
               "ws": "".join(k[0] for k in held)}
        if enemies and (threats or tick % 10 == 0):
            snap = []
            for d_, e_ in enemies[:3]:
                to_me = math.atan2(y - e_["pos"][1], x - e_["pos"][0])
                snap.append([e_["id"], round(e_["pos"][0]), round(e_["pos"][1]), round(e_.get("hp", -1)),
                             "".join(w[0] for w in (e_.get("weapons") or [])), 1 if vis.get(e_["id"]) else 0,
                             round(abs(wrap(e_["aim"][0] - to_me)), 2), 1 if e_.get("charging") else 0])
            rec["ns"] = snap
        if fire[0] or fire[1]:
            rec["fi"] = "".join(str(int(v)) for v in fire)
        if dm:
            rec["dm"] = dm
        if stuck:
            rec["stk"] = self.stuck_events
        if self.log_all:
            mf.log(json.dumps(rec, separators=(",", ":")))
        self.draw(tick, rec, tgt, goal_pt, x, y, z, bool(dm or threats))
        return act


if __name__ == "__main__":
    mf.run(Bot)
