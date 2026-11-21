"""bot_four.py -- the 3.3 'agg' bot, plus a working map, line-of-sight-aware shooting and corner/cover play.

WHY THIS EXISTS
  The 3.3 bot builds its walkability grid with Grid(map_dir, budget=0.20) and the slope pass raises TimeoutError
  the moment 0.20 s have passed (json.load of world.json alone takes longer). Unless that was fast enough, the grid
  was None on every run: no A*, no wall-aware steering, no line-of-sight. This file loads a PRECOMPUTED grid
  (grid_cache.bin, made once by build_grid_cache.py from your real world.json) so the map really works.

WHAT IS NEW (every item has a switch in FEATURES; flip one off to ablate it)
  map_cache   load grid_cache.bin (walls, eye-level solids, convex building corners); fall back to a slow build
  los_gate    never hold the trigger without line of sight (the old bot shot into walls)
  cover       the new combat layer:
                RECOVER  below a HP threshold (or 3+ enemies see us): break line of sight, hold still facing
                         the enemy, regenerate (45 HP/s after 5 s unhurt) up to recover_to, then re-engage
                PEEK/BACK against a target farther than 55 m (no lightning): pick a fire spot next to a building
                         corner that sees the target and a hidden spot <= 14 m away; peek ~1.5 s, step back
                         while reloading / when hurt / when the weapon is on cooldown
  telemetry   one JSON decision record in the debug drawing, state changes + damage bursts + a 20 s summary on
              stderr, and a self-check of the map model (LOSCHK lines: damage taken from an enemy the map says
              cannot see us). If the map model proves wrong the cover layer and the LOS gate switch themselves off.

Everything else (shoot(), loot, drops, zone, strafing, dodging, stuck handling) is the 3.3 code.
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
TPS = 20                                   # sim ticks per second; a decision happens every 2 ticks
NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
      (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
RANK = {"laser": 3.0, "lightning": 2.0, "mortar": 1.0, "gun": 9.0}
UPGRADE_PRIORITY = ["armor", "plating", "shield", "hp", "health", "regen", "repair",
                    "gun_rate", "rate", "damage", "speed", "boost", "energy", "sensor"]
CACHE_NAME = "grid_cache.bin"

STRATEGY = "agg"

FEATURES = {
    # ---- from the 3.3 bot (unchanged)
    "pref_fix": True, "drop_fix": True, "energy_reserve": True, "smart_search": True, "lightning_first": True,
    "flee_hp": 0.22, "back_circle": True, "loot_dash": True, "zone_eta": True, "stagger_hit": True,
    "regen_retreat": False, "max_pursuit": 260.0, "record": True,
    # ---- new
    "map_cache": True,        # load grid_cache.bin
    "runtime_build": True,    # if no valid cache: try to build the grid at start-up (needs up to ~8 s)
    "los_gate": True,         # do not fire without line of sight
    "cover": True,            # master switch for RECOVER and PEEK/BACK
    "peek": True,             # PEEK/BACK against far targets (needs cover)
    "recover_hp_one": 0.40,   # enter RECOVER below this HP fraction when one enemy sees us ...
    "recover_hp_multi": 0.60, # ... or below this when two or more do
    "recover_crowd": 3,       # 3+ enemies with line of sight and HP < 0.85 also triggers RECOVER
    "recover_to": 0.78,       # leave RECOVER at this HP fraction
    "peek_min_dist": 55.0,    # PEEK/BACK only against targets at least this far
    "peek_ticks": 30,         # longest single peek (1.5 s)
    "back_ticks": 14,         # shortest time behind cover between peeks (0.7 s)
}
if FEATURES["lightning_first"]:
    RANK["lightning"] = 3.5


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
# The map. Cells are 4 m. Per cell: inf (1 = a mech cannot stand here, obstacles grown by one cell),
# lo/hi (lowest base / highest top of any solid in the cell: used for line of sight), plus a list of
# "corner cells": walkable cells 8-11 m from the convex corners of blocking buildings.
# ----------------------------------------------------------------------------------------------
class Grid:
    CELL = 4.0

    def __init__(self):
        self.build_s = 0.0
        self.source = "none"
        self.corners = []
        self.cbuckets = {}

    # ---- terrain access
    def _load_terrain(self, world, map_dir):
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
        self.W = int((self.nx - 1) * self.step / C) + 1
        self.H = int((self.ny - 1) * self.step / C) + 1

    @staticmethod
    def signature(map_dir):
        wp = os.path.join(map_dir, "v2", "world.json")
        tp = os.path.join(map_dir, "v2", "terrain.i16")
        import hashlib
        with open(wp, "rb") as f:
            head = f.read(65536)
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read(65536)
        return {"world_size": size, "terrain_size": os.path.getsize(tp),
                "head": hashlib.md5(head).hexdigest(), "tail": hashlib.md5(tail).hexdigest()}

    # ---- build from world.json (scanline polygon fill: much faster than testing every cell)
    @classmethod
    def build(cls, map_dir, budget=None):
        t0 = time.perf_counter()
        self = cls()
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            world = json.load(f)
        self._load_terrain(world, map_dir)
        W, H, C = self.W, self.H, self.CELL
        N = W * H
        minx, miny = self.minx, self.miny
        blk = bytearray(N)
        lo = array("f", [1e9]) * N
        hi = array("f", [-1e9]) * N
        self.lo, self.hi = lo, hi
        corner_pts = []
        for pi, p in enumerate(world.get("prisms", [])):
            if budget is not None and (pi & 15) == 0 and time.perf_counter() - t0 > budget:
                raise TimeoutError("grid build exceeded %.1fs at prism %d" % (budget, pi))
            ring = p.get("ring")
            if not ring or len(ring) < 6:
                continue
            xs, ys = ring[0::2], ring[1::2]
            base = p.get("base", 0.0)
            top = p.get("top", base + 10.0)
            walk_block = (base - self.h(xs[0], ys[0])) < 9.5      # a 9 m mech walks under higher bridges
            edges = []
            for rx, ry in [(xs, ys)] + [(hh[0::2], hh[1::2]) for hh in (p.get("holes") or []) if len(hh) >= 6]:
                m_ = len(rx)
                for i in range(m_):
                    edges.append((rx[i], ry[i], rx[(i + 1) % m_], ry[(i + 1) % m_]))
            gx0 = max(0, int((min(xs) - minx) / C))
            gx1 = min(W - 1, int((max(xs) - minx) / C))
            gy0 = max(0, int((min(ys) - miny) / C))
            gy1 = min(H - 1, int((max(ys) - miny) / C))
            for gy in range(gy0, gy1 + 1):
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
            for i in range(n):   # thin walls: also mark the outline
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
                corner_pts.extend(cls._corners(xs, ys))
        for gy in range(H):  # slopes too steep to walk
            if budget is not None and (gy & 15) == 0 and time.perf_counter() - t0 > budget:
                raise TimeoutError("grid build exceeded %.1fs in slope pass" % budget)
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
        inf = bytearray(blk)   # grow every obstacle by one cell: the mech is 5 m wide
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
            raise ValueError("grid looks wrong (over half blocked)")
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
        self._index_corners()
        self.build_s = time.perf_counter() - t0
        self.source = "built"
        return self

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
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            if cross * s < 0.64:                  # not a convex corner sharper than ~40 degrees
                continue
            n1 = (s * e1[1], -s * e1[0])           # outward normals of the two walls
            n2 = (s * e2[1], -s * e2[0])
            bis = unit(n1[0] + n2[0], n1[1] + n2[1])
            for o in (8.0, 11.0):
                out.append((vx + o * n1[0], vy + o * n1[1]))
                out.append((vx + o * n2[0], vy + o * n2[1]))
                out.append((vx + o * bis[0], vy + o * bis[1]))
                out.append((vx + o * n1[0] - 7.0 * e1[0], vy + o * n1[1] - 7.0 * e1[1]))    # hugging wall 1, behind the corner
                out.append((vx + o * n2[0] + 7.0 * e2[0], vy + o * n2[1] + 7.0 * e2[1]))    # hugging wall 2, behind the corner
        return out

    def _index_corners(self):
        b = {}
        for (cx, cy) in self.corners:
            b.setdefault((int(cx // 40), int(cy // 40)), []).append((cx, cy))
        self.cbuckets = b

    def corners_near(self, x, y, R, limit=60):
        bx0, bx1 = int((x - R) // 40), int((x + R) // 40)
        by0, by1 = int((y - R) // 40), int((y + R) // 40)
        out = []
        R2 = R * R
        for bx in range(bx0, bx1 + 1):
            for by in range(by0, by1 + 1):
                for (cx, cy) in self.cbuckets.get((bx, by), ()):
                    d2 = (cx - x) ** 2 + (cy - y) ** 2
                    if d2 <= R2:
                        out.append((d2, cx, cy))
        out.sort()
        return [(cx, cy) for _, cx, cy in out[:limit]]

    # ---- cache
    def save(self, path, map_dir):
        hdr = {"v": 1, "W": self.W, "H": self.H, "cell": self.CELL, "nc": len(self.corners),
               "nx": self.nx, "ny": self.ny, "minx": self.minx, "miny": self.miny, "step": self.step,
               "sig": self.signature(map_dir), "blocked": self.blocked_frac, "build_s": self.build_s}
        with open(path, "wb") as f:
            f.write(b"MFG1\n" + json.dumps(hdr).encode() + b"\n")
            f.write(bytes(self.inf))
            f.write(self.lo.tobytes())
            f.write(self.hi.tobytes())
            f.write(array("f", [v for c in self.corners for v in c]).tobytes())

    @classmethod
    def load(cls, path, map_dir):
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            if f.readline() != b"MFG1\n":
                raise ValueError("bad cache magic")
            hdr = json.loads(f.readline().decode())
            if hdr.get("sig") != cls.signature(map_dir):
                raise ValueError("cache was built from a different world.json/terrain.i16 (signature mismatch)")
            self = cls()
            W, H = hdr["W"], hdr["H"]
            N = W * H
            self.inf = bytearray(f.read(N))
            self.lo = array("f")
            self.lo.frombytes(f.read(4 * N))
            self.hi = array("f")
            self.hi.frombytes(f.read(4 * N))
            cr = array("f")
            cr.frombytes(f.read(8 * hdr["nc"]))
        if sys.byteorder == "big":
            self.lo.byteswap()
            self.hi.byteswap()
            cr.byteswap()
        self._load_terrain_light(map_dir, hdr)       # terrain.i16 itself is read from the mounted map (fast)
        self.corners = [(cr[i], cr[i + 1]) for i in range(0, len(cr), 2)]
        self._index_corners()
        self.blocked_frac = hdr.get("blocked", -1.0)
        self.build_s = time.perf_counter() - t0
        self.source = "cache"
        return self

    def _load_terrain_light(self, map_dir, hdr):
        self.nx, self.ny = hdr["nx"], hdr["ny"]
        self.minx, self.miny, self.step = hdr["minx"], hdr["miny"], hdr["step"]
        ter = array("h")
        with open(os.path.join(map_dir, "v2", "terrain.i16"), "rb") as f:
            ter.frombytes(f.read())
        if sys.byteorder == "big":
            ter.byteswap()
        if len(ter) != self.nx * self.ny:
            raise ValueError("terrain size mismatch")
        self.ter = ter
        self.W, self.H = hdr["W"], hdr["H"]

    # ---- terrain and cells
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

    # ---- shooting checks (trees never stop shots)
    def los(self, a, b, skip=5.0, tail=3.0, step=3.0):
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

    def find_path(self, sx, sy, tx, ty, max_exp=1500, tlimit=0.008):
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
                    continue
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


def load_grid(map_dir):
    """Cache first, then a slow build. Returns (grid or None, human readable status)."""
    here = os.path.dirname(os.path.abspath(__file__))
    notes = []
    if FEATURES["map_cache"]:
        try:
            g = Grid.load(os.path.join(here, CACHE_NAME), map_dir)
            return g, "cache ok (%d corner cells, blocked %.2f, %.2fs)" % (len(g.corners), g.blocked_frac, g.build_s)
        except Exception as ex:
            notes.append("cache: %s" % ex)
    if FEATURES["runtime_build"]:
        try:
            g = Grid.build(map_dir, budget=8.0)
            return g, "BUILT AT START-UP (%.1fs); run build_grid_cache.py. %s" % (g.build_s, "; ".join(notes))
        except Exception as ex:
            notes.append("build: %s" % ex)
    return None, "NO GRID (%s)" % "; ".join(notes)


# ----------------------------------------------------------------------------------------------
# The bot
# ----------------------------------------------------------------------------------------------
class Bot:
    name = "four-cover"

    def __init__(self, hello):
        cfg = hello["config"]
        self.reach = cfg["loot"]["pickup_radius"]
        self.mortar_g = cfg.get("weapons", {}).get("mortar", {}).get("gravity", 22.0)
        self.rng = random.Random(hello.get("seed", 0))
        self.log_all = bool(os.environ.get("MECH_LOG"))
        self.grid = None
        t0 = time.perf_counter()
        try:
            self.grid, status = load_grid(hello["map"]["dir"])
        except Exception:
            status = "NO GRID (crash)\n" + traceback.format_exc()
        mf.log("START %s slot=%s seed=%s GRID: %s (init %.2fs)" % (self.name, hello.get("slot"), hello.get("seed"), status, time.perf_counter() - t0))
        self.T0 = time.perf_counter()
        # navigation (3.3)
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
        self.last_ang = None
        self.last_desired = (1.0, 0.0)
        self.hint = None
        # cover layer
        self.cf = None               # None | "RECOVER" | "PEEK" | "BACK"
        self.cf_since = 0
        self.cf_block_until = -1
        self.cover = None            # (x, y, r) hidden cell to run to
        self.cover_next = -1
        self.plan = None             # {"F":(x,y),"H":(x,y),"tick":..,"tid":..,"tpos":..}
        self.plan_next = -1
        self.dmg_hist = []           # (tick, amount) non-zone damage
        self.last_known = None
        self.los_miss = 0
        self.los_miss_tick = -999
        self.los_bad_until = -1
        # telemetry
        self.state = "INIT"
        self.state_ticks = {}
        self.n_act = 0
        self.ms_sum = 0.0
        self.ms_max = 0.0
        self.errors = 0
        self.dbg_bytes = 0
        self.dmg_burst_tick = -999
        self.tm = {"hid": 0, "exp": 0, "dt_exp": 0.0, "dt_hid": 0.0, "dealt": 0.0, "peeks": 0, "recov": 0,
                   "relocs": 0, "hp_regen": 0.0, "kills": 0, "fired_blind_blocked": 0}
        self.prev_hp = None

    # ---------------- navigation (3.3) ----------------
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
                if time.perf_counter() - self.T0 < 0.024:
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

    # ---------------- zone (3.3) ----------------
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
                    travel *= 1.5
            if t >= z["shrink_start"] or z["shrink_start"] - t < travel + 15:
                return self.inside_point(nc, nr, x, y, 0.6), 1
        return None, 0

    # ---------------- weapons (3.3 shoot(), now line-of-sight aware) ----------------
    def shoot(self, me, tpos, tvel, special, target_h=4.5, los_t=True):
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
        aims = {}
        fires = [False, False]

        def err(yaw, pit):
            return abs(wrap(cur[0] - yaw)), abs(cur[1] - pit)

        if "gun" in held:
            i, w = held["gun"]
            st = w.get("stats", {})
            yaw, pit = angles(o, lead(o, tc, tvel, st.get("speed", 252), 0.05))
            ok = d < st.get("range", 230) * 0.92 and w.get("ammo", 1) > 0 and w.get("reload", 0) <= 0 and los_t
            aims["gun"] = (yaw, pit, i, ok)
            if ok:
                spread = math.radians(st.get("spread_deg", 0.6))
                tol = clamp(math.atan2(2.5, max(d, 5.0)) + spread + 0.01, 0.025, 0.3)
                ye, pe = err(yaw, pit)
                if ye < tol and pe < tol * 1.6:
                    fires[i] = True

        if special and "laser" in held and hpf > 0.25:
            i, w = held["laser"]
            ch, cd = w.get("charge", 0), w.get("cooldown", 0)
            yaw, pit = angles(o, lead(o, tc, tvel, 1e9, 0.5 * max(0.1, 0.65 - ch) + 0.1))
            ok = 10 < d < 260 and los_t
            aims["laser"] = (yaw, pit, i, ok)
            ye, pe = err(yaw, pit)
            tol = clamp(math.atan2(2.5, max(d, 5.0)) + 0.02, 0.02, 0.2)
            aligned = ye < tol and pe < tol * 1.6
            if ch > 0:
                if not ok:
                    pass
                elif ch < 0.65:
                    fires[i] = True
                else:
                    fires[i] = not aligned
            elif ok and cd <= 0.15 and ye < 0.25:
                fires[i] = True

        if special and "lightning" in held:
            i, w = held["lightning"]
            rng_ = w.get("stats", {}).get("range", 45)
            yaw, pit = angles(o, tc)
            ok = d < rng_ - 2 and me["energy"] > 8 and los_t
            aims["lightning"] = (yaw, pit, i, ok)
            if ok and w.get("cooldown", 0) <= 0.05 and err(yaw, pit)[0] < 0.4:
                fires[i] = True

        if special and "mortar" in held:
            i, w = held["mortar"]
            st = w.get("stats", {})
            v, g = st.get("speed", 72), st.get("gravity", self.mortar_g)
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
                clear = True
                if self.grid is not None and FEATURES["los_gate"] and tick_ok(self):
                    clear = self.grid.arc_clear(o2, yaw, pit_m, v, g, hdist(pred, o2))
                ok = 25 < d < 200 and w.get("cooldown", 0) <= 0.1 and (slow or d < 80) and clear
                aims["mortar"] = (yaw, pit_m, i, ok)
                if ok:
                    ye, pe = err(yaw, pit_m)
                    if ye < 0.15 and pe < 0.3:
                        fires[i] = True

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
        if FEATURES["back_circle"] and d < 70 and e.get("aim"):
            rel = wrap(math.atan2(y - e["pos"][1], x - e["pos"][0]) - e["aim"][0])
            if abs(rel) < 2.4 and self.rng.random() < 0.75:
                return 1.0 if rel > 0 else -1.0
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

    # ---------------- hints (3.3) ----------------
    def update_hints(self, obs, pos, tick):
        candidates = []
        sound_dist = {"near": 40.0, "mid": 110.0, "far": 220.0}
        useful = ("gun", "mortar", "laser", "lightning", "explosion", "laser_charge", "step", "boost", "jet", "npc_gun")
        for s in obs.get("sounds", []):
            age = float(s.get("age", 99.0))
            if s.get("kind") not in useful or age > 1.5:
                continue
            dist = sound_dist.get(s.get("dist"), 110.0)
            priority = 2.0 if s.get("kind") in ("gun", "mortar", "laser", "lightning", "explosion", "laser_charge") else 0.5
            score = priority - age - (0.35 if s.get("dist") == "far" else 0.0)
            candidates.append((score, s["bearing"], dist))
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged" and ev.get("bearing") is not None and ev.get("weapon") != "zone":
                candidates.append((4.0, ev["bearing"], 60.0))
        if candidates:
            _, bearing, dist = max(candidates, key=lambda q: q[0])
            self.hint = (pos[0] + math.cos(bearing) * dist, pos[1] + math.sin(bearing) * dist, tick)

    def search_hint(self, pos, tick):
        h = self.hint
        if h is not None and tick - h[2] < 80 and hdist(pos, h) > 15:
            return (h[0], h[1])
        return None

    # ---------------- cover layer ----------------
    def dmg_recent(self, tick, window):
        return sum(a for (tk, a) in self.dmg_hist if tick - tk <= window)

    @staticmethod
    def weapon_ready(me):
        for w in me["weapons"]:
            if not w:
                continue
            k = w["kind"]
            if k == "gun" and w.get("reload", 0) <= 0 and w.get("ammo", 1) > 0 and w.get("cooldown", 0) <= 0.3:
                return True
            if k == "laser" and (w.get("cooldown", 0) <= 0.5 or w.get("charge", 0) > 0):
                return True
            if k == "mortar" and w.get("cooldown", 0) <= 0.5:
                return True
        return False

    def zone_ok(self, cx, cy, zone, nc, nr):
        zc, zr = zone["center"], zone["radius"]
        return hdist((cx, cy), zc) <= zr - 5 and hdist((cx, cy), nc) <= nr * 0.97

    def find_cover(self, x, y, eyes, zone, nc, nr, tl=0.008):
        """Nearest walkable cell, reachable in a straight line, that none of the given enemy eye points can see."""
        g = self.grid
        if g is None or not eyes:
            return None
        t0 = time.perf_counter()
        mx = sum(e[0] for e in eyes) / len(eyes)
        my = sum(e[1] for e in eyes) / len(eyes)
        away = unit(x - mx, y - my)
        cands = []
        for (cx, cy) in g.corners_near(x, y, 40.0, 40):
            if hdist((cx, cy), (x, y)) < 5.0:
                continue
            cands.append((cx, cy))
        base = self.rng.random() * TAU / 12
        for r in (8.0, 14.0, 22.0, 32.0):
            for k in range(12):
                a = base + k * TAU / 12
                cands.append((x + math.cos(a) * r, y + math.sin(a) * r))
        best = None
        for (cx, cy) in cands:
            if time.perf_counter() - t0 > tl:
                break
            if g.blocked(cx, cy) or not self.zone_ok(cx, cy, zone, nc, nr):
                continue
            r = hdist((cx, cy), (x, y))
            if r > 45.0 or not g.ray_free(x, y, cx, cy, 3.0):
                continue
            tgt = (cx, cy, g.h(cx, cy) + 4.5)
            if any(g.los(e, tgt, 5.0, 2.0, 3.0) for e in eyes):
                continue
            sc = r - 8.0 * (((cx - x) * away[0] + (cy - y) * away[1]) / max(r, 1.0))
            if best is None or sc < best[0]:
                best = (sc, cx, cy, r)
        return None if best is None else (best[1], best[2], best[3])

    def plan_fire(self, x, y, tgt_e, others, zone, nc, nr, tl=0.008):
        """Fire spot F (sees the target) and hidden spot H (sees nobody we know of) <= 14 m apart, near building corners."""
        g = self.grid
        if g is None:
            return None
        eT = (tgt_e["pos"][0], tgt_e["pos"][1], tgt_e["pos"][2] + 7.0)
        tp = (tgt_e["pos"][0], tgt_e["pos"][1])
        oe = [(o["pos"][0], o["pos"][1], o["pos"][2] + 7.0) for o in others][:2]
        cands = g.corners_near(x, y, 45.0, 36)
        t0 = time.perf_counter()
        F, Hs = [], []
        for (cx, cy) in cands:
            if time.perf_counter() - t0 > tl:
                break
            if g.blocked(cx, cy) or not self.zone_ok(cx, cy, zone, nc, nr):
                continue
            # line of sight must hold (or fail) at the cell centre AND 2.5 m to each side, so a spot that is only
            # "visible" at one exact point is not chosen (we never stand exactly on a cell centre)
            ux_, uy_ = unit(cx - tp[0], cy - tp[1])
            pts = [(cx, cy), (cx - uy_ * 2.5, cy + ux_ * 2.5), (cx + uy_ * 2.5, cy - ux_ * 2.5)]
            nvis = sum(1 for (px, py) in pts if g.los(eT, (px, py, g.h(px, py) + 4.5), 5.0, 2.0, 3.0))
            tgt = (cx, cy, g.h(cx, cy) + 4.5)
            ex = sum(1 for e in oe if g.los(e, tgt, 5.0, 2.0, 3.0))
            dT = hdist((cx, cy), tp)
            if nvis == 3:
                if 25.0 < dT < 200.0:
                    F.append((ex, cx, cy, dT))
            elif nvis == 0 and ex == 0:
                Hs.append((cx, cy))
        best = None
        for (ex, fx, fy, dT) in F:
            bh = None
            for (hx, hy) in Hs:
                dd = hdist((fx, fy), (hx, hy))
                if 3.0 <= dd <= 14.0 and (bh is None or dd < bh[0]) and g.ray_free(fx, fy, hx, hy, 3.0):
                    bh = (dd, hx, hy)
            if bh is None:
                continue
            sc = 0.5 * hdist((fx, fy), (x, y)) + 25.0 * ex + bh[0] + 0.15 * abs(dT - 90.0)
            if best is None or sc < best[0]:
                best = (sc, (fx, fy), (bh[1], bh[2]))
        if best is None:
            return None
        return {"F": best[1], "H": best[2]}

    def cover_policy(self, C):
        """Returns {"mode","vec","boost","goal"} or None (then the 3.3 fight/flee code runs)."""
        tick, g = C["tick"], self.grid
        hpf, threats, tgt, me, pos = C["hpf"], C["threats"], C["tgt"], C["me"], C["pos"]
        x, y = pos
        zone, nc, nr = C["zone"], C["nc"], C["nr"]
        exposed_n = sum(1 for d, e in threats if d < 170)
        dmg_burst = self.dmg_recent(tick, 10)
        eyes = [(e["pos"][0], e["pos"][1], e["pos"][2] + 7.0) for d, e in threats if d < 220]
        if not eyes and tick - self.hit_tick <= 10 and self.cf is not None:
            eyes = [(x + math.cos(self.hit_bearing) * 80.0, y + math.sin(self.hit_bearing) * 80.0, C["z"] + 7.0)]

        def set_state(s):
            if s != self.cf:
                if s == "RECOVER":
                    self.tm["recov"] += 1
                if s == "PEEK":
                    self.tm["peeks"] += 1
                self.cf, self.cf_since = s, tick
                if s != "RECOVER":
                    self.cover = None

        # ---- RECOVER: entry
        thr_hp = FEATURES["recover_hp_one"] if exposed_n <= 1 else FEATURES["recover_hp_multi"]
        danger_now = bool(threats) or tick - self.hit_tick < 30
        if (self.cf != "RECOVER" and tick >= self.cf_block_until and danger_now
                and (hpf < thr_hp or (exposed_n >= FEATURES["recover_crowd"] and hpf < 0.85))):
            set_state("RECOVER")
        # ---- RECOVER: behaviour / exit
        if self.cf == "RECOVER":
            close = [d for d, e in threats if d < 22.0]
            if hpf >= FEATURES["recover_to"] or tick - self.cf_since > 20 * TPS or close:
                if close:
                    self.cf_block_until = tick + 60          # cornered: fight, do not bounce straight back in
                set_state(None)
            elif threats:                                    # somebody sees us: break line of sight
                if self.cover is None or tick >= self.cover_next:
                    if time.perf_counter() - self.T0 < 0.022:
                        self.cover = self.find_cover(x, y, eyes, zone, nc, nr)
                        self.cover_next = tick + 6
                        self.tm["relocs"] += 1
                if self.cover is not None:
                    cx, cy, cr = self.cover
                    wp = self.waypoint(x, y, (cx, cy), tick)
                    return {"mode": "COVER", "vec": [v * 1.2 for v in unit(wp[0] - x, wp[1] - y)],
                            "boost": hpf < 0.35 and me["energy"] > 20, "goal": (cx, cy)}
                set_state(None)                              # no cover anywhere: let the 3.3 flee code decide
                self.cf_block_until = tick + 40
                return None
            else:
                return {"mode": "RECOVER", "vec": [0.0, 0.0], "boost": False, "goal": None}

        # ---- PEEK / BACK (ranged targets only)
        if not (FEATURES["peek"] and tgt is not None):
            if self.cf in ("PEEK", "BACK"):
                set_state(None)
            return None
        d_t, e_t = tgt
        if d_t < FEATURES["peek_min_dist"] or "lightning" in C["held"]:
            if self.cf in ("PEEK", "BACK"):
                set_state(None)
            return None
        need = (self.plan is None or self.plan["tid"] != e_t["id"] or tick - self.plan["tick"] > 30
                or hdist(self.plan["tpos"], (e_t["pos"][0], e_t["pos"][1])) > 25.0)
        if need and tick >= self.plan_next and time.perf_counter() - self.T0 < 0.020:
            others = [e for d, e in threats if e["id"] != e_t["id"]]
            pl = self.plan_fire(x, y, e_t, others, zone, nc, nr)
            self.plan_next = tick + 8
            if pl is not None:
                pl.update({"tid": e_t["id"], "tick": tick, "tpos": (e_t["pos"][0], e_t["pos"][1])})
                self.plan = pl
            elif self.plan is not None and self.plan["tid"] != e_t["id"]:
                self.plan = None
        if self.plan is None or self.plan["tid"] != e_t["id"]:
            if self.cf in ("PEEK", "BACK"):
                set_state(None)
            return None
        F, H = self.plan["F"], self.plan["H"]
        ready = self.weapon_ready(me)
        if self.cf not in ("PEEK", "BACK"):
            set_state("PEEK" if (ready and hpf >= 0.5) else "BACK")
        elif self.cf == "PEEK":
            if (tick - self.cf_since > FEATURES["peek_ticks"] or dmg_burst >= 100 or not ready):
                set_state("BACK")
        else:
            if tick - self.cf_since > FEATURES["back_ticks"] and ready and hpf >= 0.5 and dmg_burst < 40:
                set_state("PEEK")
        goal = F if self.cf == "PEEK" else H
        dd = hdist(pos, goal)
        if dd > 2.0:
            wp = self.waypoint(x, y, goal, tick)
            return {"mode": self.cf + "_GO", "vec": list(unit(wp[0] - x, wp[1] - y)), "boost": False, "goal": goal}
        return {"mode": self.cf, "vec": [0.0, 0.0], "boost": False, "goal": goal}

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
        if self.plan is not None and tick % 10 == 0:
            d.point([self.plan["F"][0], self.plan["F"][1], z], "#ffd060", 8)
            d.point([self.plan["H"][0], self.plan["H"][1], z], "#60ff90", 8)
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
            return self._act(obs)
        except Exception:
            self.errors += 1
            if self.errors <= 5 or self.errors % 100 == 0:
                mf.log("ACT ERROR #%d tick %s:\n%s" % (self.errors, obs.get("tick"), traceback.format_exc()))
            return {"tick": obs["tick"], "move": [0.0, 0.0]}

    def _act(self, obs):
        self.T0 = time.perf_counter()
        g = self.grid
        tick, t = obs["tick"], obs["t"]
        me = obs["self"]
        x, y, z = me["pos"]
        pos = (x, y)
        st = me["stats"]
        walk = st.get("walk_speed", 11.7)
        hpf = me["hp"] / max(1.0, me["max_hp"])
        zone = obs["zone"]
        alive = obs.get("alive", 16)
        act = {"tick": tick}
        self.update_hints(obs, pos, tick)
        map_trusted = g is not None and tick >= self.los_bad_until
        cover_on = FEATURES["cover"] and map_trusted
        gate_on = FEATURES["los_gate"] and map_trusted

        # ---- enemies and who can see us
        enemies = []
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                p3 = math.sqrt((m["pos"][0] - x) ** 2 + (m["pos"][1] - y) ** 2 + (m["pos"][2] - z) ** 2)
                enemies.append((p3, m))
        enemies.sort(key=lambda a: a[0])
        eye_me = (x, y, z + 4.5)
        vis = {}
        for i, (d, e) in enumerate(enemies):
            if d > 240:
                break
            if g is not None and i < 5 and time.perf_counter() - self.T0 < 0.015:
                vis[e["id"]] = g.los((e["pos"][0], e["pos"][1], e["pos"][2] + 7.0), eye_me, 5.0, 3.0, 3.0)
            else:
                vis[e["id"]] = d < 60
        threats = [(d, e) for d, e in enemies if vis.get(e["id"]) and d < 220]
        if threats:
            self.last_known = (threats[0][1]["pos"][0], threats[0][1]["pos"][1])
        hidden = not threats

        # ---- events
        dm = []
        for ev in obs["events"]:
            ty = ev.get("type")
            if ty == "damaged":
                w, amt = ev.get("weapon"), ev.get("amount", 0)
                dm.append([w, round(amt), 1 if ev.get("rear") else 0])
                if w != "zone":
                    self.hit_tick = tick
                    self.dmg_hist.append((tick, amt))
                    self.tm["dt_hid" if hidden else "dt_exp"] += amt
                    if ev.get("bearing") is not None:
                        self.hit_bearing = ev["bearing"]
                    by = ev.get("by")
                    if w in ("gun", "laser", "lightning") and by in vis and not vis[by] and g is not None:
                        if tick - self.los_miss_tick > 300:
                            self.los_miss = 0
                        self.los_miss += 1
                        self.los_miss_tick = tick
                        mf.log("LOSCHK t=%.1f %s hit us for %d but the map says no line of sight (miss #%d)" % (t, w, amt, self.los_miss))
                        if self.los_miss >= 4:
                            self.los_bad_until = tick + 600
                            mf.log("LOSCHK map model looks wrong: cover layer + LOS gate OFF for 30 s")
                if FEATURES["stagger_hit"]:
                    self.sign = self.rng.choice((-1, 1))
                    self.flip_at = tick + self.rng.randint(4, 8)
                elif self.rng.random() < 0.5:
                    self.sign = -self.sign
            elif ty == "hit":
                self.tm["dealt"] += ev.get("amount", 0)
            elif ty == "kill":
                self.tm["kills"] += 1
        self.dmg_hist = [(tk, a) for (tk, a) in self.dmg_hist if tick - tk <= 60]
        if dm:
            self.note_damage(tick, t, dm, me, pos, hidden)
        if self.prev_hp is not None and me["hp"] > self.prev_hp and self.cf in ("RECOVER", "BACK"):
            self.tm["hp_regen"] += me["hp"] - self.prev_hp
        self.prev_hp = me["hp"]
        self.tm["hid" if hidden else "exp"] += 1

        # ---- upgrade choice
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
        npcs = [(hdist(m["pos"], pos), m) for m in obs["mechs"] if m["kind"] == "npc"]
        crates = [(hdist(c["pos"], pos), c) for c in obs["crates"]]
        near_enemies = [(d, m) for d, m in enemies if d < 130]
        n_close = len(near_enemies)
        flee = n_close >= 3 or (hpf < 0.3 and n_close >= 1)

        # ---- weapon pickups
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
        pursue_limit = float(FEATURES.get("max_pursuit", 260.0))
        eligible = [e for e in enemies if e[0] <= pursue_limit]
        if eligible:
            tgt = min(eligible, key=lambda e: e[0] * (0.6 + 0.8 * (e[1].get("hp", 1000) / max(1.0, e[1].get("max_hp", 1000)))))
        farm = None
        if tgt is None and hpf > 0.5:
            opts = [(d, m, "npc") for d, m in npcs if d < 90] + [(d, c, "crate") for d, c in crates if d < 60]
            if opts:
                farm = min(opts, key=lambda o: o[0])

        mh = st.get("muzzle_height", 7.0)

        def los_to(p3):
            if not gate_on:
                return True
            return g.los((x, y, z + mh), p3, 5.0, 3.0, 3.0)

        aim, fire = None, [False, False]
        if tgt and tgt[0] < 235:
            th_ = 4.5 * (1 + 0.05 * (tgt[1].get("cubes", 0) or 0))
            tp_ = tgt[1]["pos"]
            ls = los_to((tp_[0], tp_[1], tp_[2] + th_))
            aim, fire = self.shoot(me, tp_, tgt[1]["vel"], True, th_, ls)
        elif farm:
            tv = farm[1].get("vel", (0, 0, 0))
            fp = farm[1]["pos"]
            aim, fire = self.shoot(me, fp, tv, False, 4.5, los_to((fp[0], fp[1], fp[2] + 3.0)))
        elif tick - self.hit_tick < 15:
            aim = [self.hit_bearing, 0.0]
        else:
            for s in obs["sounds"]:
                if s["kind"] in ("gun", "mortar", "laser", "lightning", "laser_charge") and s["dist"] != "far" and s["age"] < 0.6:
                    aim = [s["bearing"], 0.0]
                    break

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

        # ================= movement policy (agg + cover layer) =================
        C = {"tick": tick, "me": me, "pos": pos, "z": z, "hpf": hpf, "threats": threats, "tgt": tgt,
             "held": held, "zone": zone, "nc": nc, "nr": nr}
        cov = None
        if cover_on and zgoal is None and (tgt is not None or self.cf == "RECOVER"):
            cov = self.cover_policy(C)
        elif self.cf is not None and (not cover_on or zgoal is not None or (tgt is None and self.cf != "RECOVER")):
            self.cf, self.cover = None, None

        if cov is not None:
            desired = list(cov["vec"])
            mode = cov["mode"]
            boost_wanted = cov.get("boost", False)
            if cov.get("goal") is not None:
                goal_pt = cov["goal"]
        elif tgt is not None and zgoal is None:
            e = tgt[1]
            d = hdist(e["pos"], pos)
            ux, uy = unit(e["pos"][0] - x, e["pos"][1] - y)
            flee_at = FEATURES["flee_hp"]
            if FEATURES["regen_retreat"] and e.get("hp", 1000) / max(1.0, e.get("max_hp", 1000)) > hpf + 0.2:
                flee_at = 0.30
            must_flee = (n_close >= 3 or hpf < flee_at or (hpf < 0.30 and n_close >= 1))
            if (not must_flee and FEATURES["loot_dash"] and loot_wp is not None and free_slot is not None
                    and loot_wp[0] < 30 and d > 50 and hpf >= flee_at):
                desired = [0.0, 0.0]
                wpd = self.waypoint(x, y, loot_wp[1], tick)
                add(unit(wpd[0] - x, wpd[1] - y), 1.5)
                goal_pt = loot_wp[1]
                mode = "LOOT"
            elif must_flee:
                desired = [0.0, 0.0]
                fx, fy = 0.0, 0.0
                for ed, enemy in near_enemies:
                    ax, ay = unit(x - enemy["pos"][0], y - enemy["pos"][1])
                    weight = 1.0 / max(ed, 15.0)
                    fx += ax * weight
                    fy += ay * weight
                if math.hypot(fx, fy) < 1e-6:
                    ax, ay = unit(x - e["pos"][0], y - e["pos"][1])
                    fx, fy = ax, ay
                if math.hypot(fx, fy) < 1e-6 and near_enemies:
                    _, nearest = min(near_enemies, key=lambda q: q[0])
                    fx, fy = unit(x - nearest["pos"][0], y - nearest["pos"][1])
                add(unit(fx, fy), 2.0)
                mode = "FLEE"
                boost_wanted = me["energy"] > 15
            else:
                desired = [0.0, 0.0]
                if FEATURES["pref_fix"]:
                    preferred = 28.0 if "lightning" in held else 62.0
                else:
                    preferred = 48.0 if "lightning" in held else 62.0
                radial = clamp((d - preferred) / 25.0, -0.25, 1.0)
                add((ux, uy), radial)
                add((-uy, ux), 0.55 * self.strafe_sign(e, x, y, d))
                mode = "FIGHT"
                en_min = 60 if ("lightning" in held and FEATURES["energy_reserve"]) else 25
                if d > 75 and me["energy"] > en_min:
                    boost_wanted = True

        elif tgt is None and zgoal is None and (n_close >= 3 or (hpf < 0.30 and n_close >= 1)):
            fx = fy = 0.0
            for ed, enemy in near_enemies:
                ax, ay = unit(x - enemy["pos"][0], y - enemy["pos"][1])
                weight = 1.0 / max(ed, 15.0)
                fx += ax * weight
                fy += ay * weight
            if math.hypot(fx, fy) < 1e-6 and near_enemies:
                _, nearest = min(near_enemies, key=lambda q: q[0])
                fx, fy = unit(x - nearest["pos"][0], y - nearest["pos"][1])
            desired = [fx, fy]
            mode = "FLEE"
            boost_wanted = me["energy"] > 15

        elif tgt is None and zgoal is None and hold_ring and FEATURES["drop_fix"]:
            desired = [0.0, 0.0]
            mode = "CAPTURE"

        elif tgt is None and zgoal is None:
            if not hasattr(self, "agg_angle"):
                self.agg_angle = self.rng.uniform(-math.pi, math.pi)
                self.agg_turn = tick
            if tick - self.agg_turn > self.rng.randint(45, 85):
                self.agg_angle += self.rng.uniform(-1.4, 1.4)
                self.agg_turn = tick
            if math.hypot(desired[0], desired[1]) < 0.1:
                desired = [math.cos(self.agg_angle), math.sin(self.agg_angle)]
                mode = "SEARCH"
                if FEATURES["smart_search"]:
                    hint = self.search_hint(pos, tick)
                    nc_, nr_ = zone["next_center"], zone["next_radius"]
                    if hint is not None:
                        desired = list(unit(hint[0] - x, hint[1] - y))
                        mode = "HUNT"
                    elif hdist(pos, nc_) > 0.45 * nr_:
                        toc = unit(nc_[0] - x, nc_[1] - y)
                        desired = [toc[0] * 0.8 + math.cos(self.agg_angle) * 0.4, toc[1] * 0.8 + math.sin(self.agg_angle) * 0.4]
                        mode = "SEEK"

        # ---- dodging (3.3)
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
                disc = vz * vz + 2 * self.mortar_g * (p["pos"][2] - z)
                if disc < 0:
                    continue
                tt = (vz + math.sqrt(disc)) / self.mortar_g
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

        # ---- stuck handling (3.3)
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
        if aim is None and self.cf in ("RECOVER", "BACK", "PEEK") and self.last_known is not None:
            aim = [math.atan2(self.last_known[1] - y, self.last_known[0] - x), 0.0]     # keep the barrel on the corner they come around
        if aim is None and mag > 0.3:
            aim = [math.atan2(move[1], move[0]), 0.0]
        if aim is not None:
            act["aim"] = aim
        if fire[0] or fire[1]:
            act["fire"] = fire
        gun = me["weapons"][held["gun"]] if "gun" in held else None
        if gun and gun.get("reload", 0) <= 0:
            ammo = gun.get("ammo", 99)
            if ammo <= 0 or (not (tgt and tgt[0] < 120) and ammo <= 3):
                act["reload"] = True

        # ---- logging
        ms = (time.perf_counter() - self.T0) * 1000
        self.n_act += 1
        self.ms_sum += ms
        self.ms_max = max(self.ms_max, ms)
        self.state_ticks[mode] = self.state_ticks.get(mode, 0) + 1
        if mode != self.state:
            mf.log("t=%.1f k=%d %s->%s hp=%d pos=(%d,%d) al=%d th=%d cf=%s" % (t, tick, self.state, mode, me["hp"], x, y, alive, len(threats), self.cf))
            self.state = mode
        if tick % (20 * TPS) == 0:
            mf.log("SUMMARY t=%.0f hp=%d al=%d pos=(%d,%d) ms_avg=%.1f ms_max=%.1f stuck=%d los_miss=%d states=%s tm=%s" % (
                t, me["hp"], alive, x, y, self.ms_sum / max(1, self.n_act), self.ms_max, self.stuck_events, self.los_miss,
                json.dumps(self.state_ticks, separators=(",", ":")), json.dumps({k: round(v) for k, v in self.tm.items()}, separators=(",", ":"))))
        rec = {"k": tick, "t": round(t, 1), "hp": round(me["hp"]), "e": round(me["energy"]), "p": [round(x), round(y)], "al": alive,
               "s": mode, "c": self.cf, "th": len(threats), "ne": len(enemies),
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
            rec["ns"] = snap      # [id, x, y, hp, weapons, map-says-LOS, aim error toward me (rad), charging]
        if fire[0] or fire[1]:
            rec["fi"] = "".join(str(int(v)) for v in fire)
        rec["pz"] = [round(zone["radius"]), round(nr), round(hdist(pos, zone["center"]))]
        if dm:
            rec["dm"] = dm
        if stuck:
            rec["stk"] = self.stuck_events
        if self.log_all:
            mf.log(json.dumps(rec, separators=(",", ":")))
        self.draw(tick, rec, tgt, goal_pt, x, y, z, bool(dm or threats))
        return act


def tick_ok(bot):
    """True while this decision is still well inside its time budget (50 ms hard limit)."""
    return time.perf_counter() - bot.T0 < 0.030


if __name__ == "__main__":
    mf.run(Bot)
