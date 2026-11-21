"""bot_common.py: the shared engine for the three mechfight bots (aggressive / defensive / balanced).

Every bot file only supplies a CONFIG dict (its "personality") and calls bot_common.main(CONFIG).
All the real skills live here, so the three bots differ only in strategy, never in technique:

  * Map: walkability grid (A* with no corner cutting), line-of-sight and mortar-arc checks.
  * Travel time: measured along the real A* path, not a straight line.
  * Threat model: a score per enemy, summed into a "danger" number that decides fight or flee.
  * Evasion: dodges bullets/shells, and runs "staggered" (random side, held 0.4-0.8 s) when shot at,
    while the torso keeps facing the attacker.
  * Weapons: picks the weapon by matchup (laser > mortar > lightning > laser), range and line of sight.
  * Supply drops: an explicit APPROACH -> CAPTURE state that stays inside the ring.
  * Logging: one compact JSON record per decision (stderr locally, debug drawing in real matches).
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
NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
      (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
BEATS = {"laser": "mortar", "mortar": "lightning", "lightning": "laser"}  # key beats value
COUNTER = {v: k for k, v in BEATS.items()}                                # COUNTER[x] = what beats x

# ----------------------------------------------------------------------------------------------
# Default personality. Each bot file overrides some of these.
# ----------------------------------------------------------------------------------------------
BASE = {
    "name": "mech",
    "flee_danger": 2.4,     # run away when (danger x weakness) goes above this
    "flee_hp": 0.30,        # ...or when HP fraction is below this and someone dangerous is near
    "engage_min": 0.2,      # a target must score above this before we walk toward it
    "chase": 130.0,         # farthest we will walk after an enemy (m)
    "pref_mul": 1.0,        # multiplies the preferred fighting distance
    "strafe": 0.9,          # sideways movement while fighting
    "avoid_dist": 0.0,      # keep this far from visible enemies when not fighting (0 = ignore)
    "cube_range": 50.0, "loot_range": 75.0, "farm_range": 70.0,
    "drop_range": 220.0,    # how far we will walk for a supply drop
    "drop_danger": 1.2,     # max danger to go for / stay on a drop
    "zone_margin": 15.0,    # start moving to the next safe circle this many seconds early
    "evade_weight": 1.2,
    "upgrades": ["def", "off", "mob", "sense"],   # upgrade categories, best first
    "adaptive": False,      # balanced bot: shift thresholds with HP and enemies alive
    "record_debug": True,   # write the decision log into the debug drawing (downloadable by your team)
}


# ----------------------------------------------------------------------------------------------
# Small math helpers
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


def inside_poly(x, y, xs, ys):
    res = False
    j = len(xs) - 1
    for i in range(len(xs)):
        if (ys[i] > y) != (ys[j] > y) and x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i]) + xs[i]:
            res = not res
        j = i
    return res


def angles(o, p):
    dx, dy, dz = p[0] - o[0], p[1] - o[1], p[2] - o[2]
    return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))


def lead(o, p, v, speed, extra=0.1):
    """Where will a moving target be when a shot of this speed arrives?"""
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
# The map: walkability grid (4 m cells), line of sight, A*
# ----------------------------------------------------------------------------------------------
class Grid:
    CELL = 4.0

    def __init__(self, map_dir, budget=9.0):
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
        blk = bytearray(N)                  # 1 = a mech cannot walk here
        self.lo = array("f", [1e9]) * N     # lowest solid height in the cell (for shots)
        self.hi = array("f", [-1e9]) * N    # highest solid height in the cell
        minx, miny = self.minx, self.miny
        lo, hi = self.lo, self.hi

        for p in world.get("prisms", []):
            if time.perf_counter() - t0 > budget:
                break
            ring = p.get("ring")
            if not ring or len(ring) < 6:
                continue
            xs, ys = ring[0::2], ring[1::2]
            base = p.get("base", 0.0)
            top = p.get("top", base + 10.0)
            walk_block = (base - self.h(xs[0], ys[0])) < 9.5   # a 9 m mech walks under higher bridges
            holes = [(hh[0::2], hh[1::2]) for hh in (p.get("holes") or []) if len(hh) >= 6]
            gx0 = max(0, int((min(xs) - minx) / C))
            gx1 = min(W - 1, int((max(xs) - minx) / C))
            gy0 = max(0, int((min(ys) - miny) / C))
            gy1 = min(H - 1, int((max(ys) - miny) / C))

            def mark(k):
                if base < lo[k]:
                    lo[k] = base
                if top > hi[k]:
                    hi[k] = top
                if walk_block:
                    blk[k] = 1

            for gy in range(gy0, gy1 + 1):
                y = miny + (gy + 0.5) * C
                for gx in range(gx0, gx1 + 1):
                    x = minx + (gx + 0.5) * C
                    if inside_poly(x, y, xs, ys) and not any(inside_poly(x, y, hx, hy) for hx, hy in holes):
                        mark(gy * W + gx)
            n = len(xs)
            for i in range(n):  # thin walls: also mark the outline
                ax, ay, bx, by = xs[i], ys[i], xs[(i + 1) % n], ys[(i + 1) % n]
                m = max(1, int(math.hypot(bx - ax, by - ay) / 2))
                for s in range(m + 1):
                    gx = int((ax + (bx - ax) * s / m - minx) / C)
                    gy = int((ay + (by - ay) * s / m - miny) / C)
                    if 0 <= gx < W and 0 <= gy < H:
                        mark(gy * W + gx)

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

        inf = bytearray(blk)  # grow every obstacle by one cell: the mech is 5 m wide
        for gy in range(H):
            for gx in range(W):
                if blk[gy * W + gx]:
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            a, b = gx + dx, gy + dy
                            if 0 <= a < W and 0 <= b < H:
                                inf[b * W + a] = 1
        self.inf = inf
        if sum(inf) > 0.5 * N:
            raise ValueError("grid looks wrong (over half blocked): navigating without it")

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

    # ---- walking checks
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

    # ---- shooting checks
    def los(self, a, b, skip=5.0, tail=3.0, step=3.0):
        """Can a bullet/beam fly from 3D point a to 3D point b? Trees do not count (they never stop shots)."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        L = math.hypot(dx, dy)
        if L < skip + tail:
            return True
        s = skip
        while s <= L - tail:
            f = s / L
            x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            i = self.idx(x, y)
            if self.lo[i] <= z <= self.hi[i] or z < self.h(x, y) - 0.5:
                return False
            s += step
        return True

    def arc_clear(self, o, yaw, pit, v, g, hd):
        """Does a mortar shell launched from o fly clear of buildings until it is near horizontal distance hd?"""
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

    # ---- A*: returns (path of (x, y) points, reached_goal). No corner cutting.
    def find_path(self, sx, sy, tx, ty, max_exp=1400, tlimit=0.008):
        W, H, inf = self.W, self.H, self.inf
        s = self.idx(sx, sy)
        g = self.nearest_free(tx, ty)
        if g is None:
            return None, False
        gx_, gy_ = g % W, g // W

        def hf(i):
            dx, dy = abs(i % W - gx_), abs(i // W - gy_)
            return 1.25 * (dx + dy - 0.586 * min(dx, dy))

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
                    continue  # diagonal step would cut a building's corner
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
        return path, best == g


# ----------------------------------------------------------------------------------------------
# The bot brain
# ----------------------------------------------------------------------------------------------
class Brain:
    def __init__(self, hello, cfg):
        self.p = dict(BASE)
        self.p.update(cfg)
        self.name = self.p["name"]
        c = hello["config"]
        self.slot = hello.get("slot", -1)
        self.reach = c["loot"]["pickup_radius"]
        self.mortar_g = c.get("weapons", {}).get("mortar", {}).get("gravity", 22.0)
        self.rng = random.Random(hello.get("seed", 0))
        self.log_stderr = bool(os.environ.get("MECH_LOG"))
        self.grid = None
        try:
            self.grid = Grid(hello["map"]["dir"])
        except Exception:
            mf.log("grid build failed, steering without map:\n" + traceback.format_exc())
        self.T0 = time.perf_counter()
        # navigation
        self.paths = {}
        self.draw_path = None
        self.last_ang = None
        self.last_desired = (1.0, 0.0)
        self.hist = []
        self.stuck_until = -1
        self.stuck_dir = (0.0, 0.0)
        self.stuck_events = 0
        # combat memory
        self.sign = 1
        self.flip_at = 0
        self.last_hit = {"tick": -999, "bearing": None, "by": None, "rear": False, "weapon": None}
        self.dmg_mem = {}          # attacker id -> recent damage (decays)
        self.enemy_w = {}          # enemy id -> (tick seen, weapons)
        self.evade_dir = None
        self.evade_until = -1
        self.evade_speed = 1.0
        self.flee_until = -1
        self.last_jump = -99
        self.kills = 0
        # supply drops
        self.drop_id = None
        self.drop_mode = None
        self.drop_black = {}
        self.drop_contest = 0
        self.drop_abandon = ""

    # ------------------------------------------------------------------ navigation
    def route(self, key, x, y, goal, tick):
        """Next waypoint toward goal and the remaining walking distance along the real path."""
        g = self.grid
        d0 = hdist((x, y), goal)
        if g is None:
            return goal, d0
        try:
            if g.ray_free(x, y, goal[0], goal[1], 4.0):
                self.paths.pop(key, None)
                return goal, d0
            c = self.paths.get(key)
            stale = c is None or hdist(c["goal"], goal) > 10 or tick - c["tick"] > 12
            if stale and time.perf_counter() - self.T0 < 0.022:
                path, reached = g.find_path(x, y, goal[0], goal[1])
                c = {"path": path, "pi": 0, "goal": goal, "tick": tick, "reached": reached}
                self.paths[key] = c
            if not c or not c["path"] or len(c["path"]) < 2:
                return goal, d0 * 1.3
            p = c["path"]
            while c["pi"] < len(p) - 1 and hdist(p[c["pi"]], (x, y)) < 5:
                c["pi"] += 1
            best = c["pi"]
            for k in range(min(len(p) - 1, c["pi"] + 8), c["pi"], -1):
                if g.ray_free(x, y, p[k][0], p[k][1], 4.0):
                    best = k
                    break
            rem = hdist((x, y), p[c["pi"]])
            for i in range(c["pi"], len(p) - 1):
                rem += hdist(p[i], p[i + 1])
            if not c["reached"]:
                rem += 1.3 * hdist(p[-1], goal)
            if key == "main":
                self.draw_path = p[c["pi"]:]
            return p[best], rem
        except Exception:
            mf.log("navigation failed:\n" + traceback.format_exc())
            self.grid = None
            return goal, d0

    def steer(self, x, y, dvec, zc, zr, cap=1.0):
        """Turn a wish (dvec) into a safe direction: avoid walls and the zone edge."""
        n = math.hypot(dvec[0], dvec[1])
        if n < 0.05:
            return (0.0, 0.0)
        m = min(cap, n)
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

    # ------------------------------------------------------------------ zone
    @staticmethod
    def inside_point(nc, nr, x, y, f):
        d = math.hypot(x - nc[0], y - nc[1])
        if d <= nr * f:
            return (x, y)
        ux, uy = unit(x - nc[0], y - nc[1])
        return (nc[0] + ux * nr * f, nc[1] + uy * nr * f)

    def zone_exit_time(self, zone, x, y, t):
        """Seconds until the shrinking edge reaches (x, y). 0 if already outside, 1e9 if never."""
        c, r, nc, nr = zone["center"], zone["radius"], zone["next_center"], zone["next_radius"]
        if math.hypot(x - c[0], y - c[1]) > r:
            return 0.0
        s0, s1 = zone["shrink_start"], zone["shrink_end"]
        T0 = max(t, s0)
        if s1 <= T0:
            return 1e9
        for k in range(9):
            s = T0 + (s1 - T0) * k / 8.0
            f = clamp((s - s0) / max(s1 - s0, 1e-6), 0.0, 1.0)
            cx, cy = c[0] + (nc[0] - c[0]) * f, c[1] + (nc[1] - c[1]) * f
            rr = r + (nr - r) * f
            if math.hypot(x - cx, y - cy) > rr - 2:
                return max(0.0, s - t)
        return 1e9

    # ------------------------------------------------------------------ threat model
    def threat(self, e, d, pos, myv, tick):
        """How dangerous is this enemy to me right now? (about 0 .. 3)"""
        w = e.get("weapons") or []
        dist_term = clamp(1.0 - d / 160.0, 0.0, 1.0)
        wt = 0.35
        if "lightning" in w:
            wt = max(wt, 0.9 if d < 50 else 0.25)
        if "laser" in w:
            wt = max(wt, 0.8 if d < 220 else 0.3)
        if "mortar" in w:
            wt = max(wt, 0.6 if d < 200 else 0.2)
        ex, ey = e["pos"][0], e["pos"][1]
        to_me = math.atan2(pos[1] - ey, pos[0] - ex)
        diff = abs(wrap(e["aim"][0] - to_me))
        aim_term = 1.0 if diff < 0.2 else 0.5 if diff < 0.5 else 0.0
        charge_term = 1.0 if (e.get("charging") and aim_term > 0) else 0.0
        ux, uy = unit(ex - pos[0], ey - pos[1])
        closing = -((e["vel"][0] - myv[0]) * ux + (e["vel"][1] - myv[1]) * uy)
        appr = clamp(closing / 15.0, 0.0, 1.0)
        mem = clamp(self.dmg_mem.get(e["id"], 0.0) / 300.0, 0.0, 1.0)
        return dist_term * (0.6 + wt) + 0.5 * aim_term + 0.8 * charge_term + 0.3 * appr + 0.6 * mem

    def enemy_weapon_count(self, kind, tick):
        return sum(1 for (tk, ws) in self.enemy_w.values() if tick - tk < 400 and kind in ws)

    def weapon_adv(self, held, e):
        ew = e.get("weapons") or []
        adv = 0
        for k in held:
            if BEATS.get(k) in ew:
                adv += 1
            if COUNTER.get(k) in ew:
                adv -= 1
        return adv

    def target_util(self, e, d, los, held, enemies, tick):
        ehp = e.get("hp", 1000.0) / max(1.0, e.get("max_hp", 1000.0))
        u = 1.4 * (1.0 - ehp)                                  # weak enemies are tempting
        u += 0.5 * self.weapon_adv(held, e)                    # good weapon matchup
        u += 0.4 if los else -0.3                              # can we actually hit it?
        u += 0.3 if d < 120 else 0.0
        others = sum(1 for d2, e2 in enemies if e2["id"] != e["id"] and hdist(e2["pos"], e["pos"]) < 60)
        u -= 0.5 * others                                      # do not dive into a crowd
        u -= d / self.p["chase"]                               # chase cost
        u += 0.3 * clamp(self.dmg_mem.get(e["id"], 0.0) / 200.0, 0.0, 1.5)   # it is already hurting us
        if self.last_hit["by"] == e["id"] and tick - self.last_hit["tick"] < 20:
            u += 0.8                                           # turn on whoever just shot us
        return u

    def pref_dist(self, held, e):
        if "lightning" in held:
            pref = 26.0
        elif "laser" in held:
            pref = 110.0
        elif "mortar" in held:
            pref = 120.0
        else:
            pref = 85.0
        ew = e.get("weapons") or []
        if "lightning" not in held and "lightning" in ew:
            pref = max(pref, 75.0)            # keep out of lightning range
        if "lightning" in held and ("laser" in ew or "gun" in ew):
            pref = 24.0                       # close in: lightning beats laser, and the gun is weak up close
        if "mortar" in held and "lightning" in ew:
            pref = max(pref, 100.0)
        return pref * self.p["pref_mul"]

    # ------------------------------------------------------------------ weapons
    def solve_weapons(self, me, e, d, special, danger):
        """Choose a weapon for this target. Returns (aim, fire[2], name, los)."""
        x, y, z = me["pos"]
        o = (x, y, z + me["stats"].get("muzzle_height", 7.0))
        cub = e.get("cubes", 0) or 0
        tc = (e["pos"][0], e["pos"][1], e["pos"][2] + 4.5 * (1 + 0.05 * cub))
        ev = e.get("vel", (0, 0, 0))
        ew = e.get("weapons") or []
        cur = me["aim"]
        g = self.grid
        cand = {}
        for i, w in enumerate(me["weapons"]):
            if not w:
                continue
            k = w["kind"]
            ws = w.get("stats", {})
            if k == "gun":
                pred = lead(o, tc, ev, ws.get("speed", 252))
                yaw, pit = angles(o, pred)
                los = g.los(o, pred) if g else True
                ok = d < ws.get("range", 230) * 0.92 and w.get("ammo", 1) > 0 and w.get("reload", 0) <= 0 and los
                util = 1.0 + (0.2 if d < 100 else 0.0)
                cand[k] = (yaw, pit, ok, i, util, los)
            elif k == "laser" and special:
                ch = w.get("charge", 0)
                pred = lead(o, tc, ev, 1e9, 0.5 * max(0.1, 0.7 - ch) + 0.1)
                yaw, pit = angles(o, pred)
                los = g.los(o, pred) if g else True
                ready = w["cooldown"] <= 0.3 or ch > 0
                ok = 10 < d < 250 and ready and los
                util = 1.5 + (0.8 if "mortar" in ew else 0.0) - (1.2 if ("lightning" in ew and d < 65) else 0.0)
                util -= 0.4 if danger > 1.5 else 0.0   # charging slows us to half speed: risky when crowded
                cand[k] = (yaw, pit, ok, i, util, los)
            elif k == "lightning" and special:
                yaw, pit = angles(o, tc)
                los = g.los(o, tc) if g else True
                rng_ = ws.get("range", 45)
                ok = d < rng_ - 3 and me["energy"] > 15 and los
                util = 1.6 + (0.7 if "laser" in ew else 0.0) - (0.9 if "mortar" in ew else 0.0)
                cand[k] = (yaw, pit, ok, i, util, los)
            elif k == "mortar" and special:
                v = ws.get("speed", 72.0)
                gr = ws.get("gravity", self.mortar_g)
                o2 = (x, y, z + me["stats"].get("muzzle_height", 7.0) + 2.5)
                pred, pit = tc, None
                for _ in range(3):
                    pit = mortar_pitch(hdist(pred, o2), pred[2] - o2[2], v, gr)
                    if pit is None:
                        break
                    tf = hdist(pred, o2) / (v * math.cos(pit))
                    pred = (tc[0] + ev[0] * (tf + 0.1), tc[1] + ev[1] * (tf + 0.1), tc[2])
                if pit is not None:
                    yaw = math.atan2(pred[1] - y, pred[0] - x)
                    clear = g.arc_clear(o2, yaw, pit, v, gr, hdist(pred, o2)) if g else True
                    slow = math.hypot(ev[0], ev[1]) < 8
                    ok = 25 < d < 215 and w["cooldown"] <= 0.3 and clear
                    util = 1.1 + (0.7 if "lightning" in ew else 0.0) + (0.3 if slow else -0.5)
                    cand[k] = (yaw, pit, ok, i, util, clear)
        best = None
        for k, c in cand.items():
            if c[2] and (best is None or c[4] > cand[best][4]):
                best = k
        if best is None:
            best = "gun" if "gun" in cand else (next(iter(cand)) if cand else None)
        if best is None:
            yaw, pit = angles(o, tc)
            return [yaw, pit], [False, False], "none", True
        yaw, pit = cand[best][0], cand[best][1]
        fire = [False, False]
        tol0 = clamp(math.atan2(2.3, max(d, 5.0)) + 0.01, 0.025, 0.22)
        for k, (cy, cp, ok, i, util, los) in cand.items():
            if k == "laser":
                ch = me["weapons"][i].get("charge", 0)
                if ch >= 0.7:
                    continue              # fully charged: release the trigger so the beam fires now
                if ch > 0 and not ok:
                    continue              # target lost while charging: release
            tol = tol0 * (1.8 if (k == "laser" and me["weapons"][i].get("charge", 0) > 0) else 1.0)
            ye, pe = abs(wrap(cur[0] - cy)), abs(cur[1] - cp)
            aligned = ye < 0.4 if k == "lightning" else (ye < tol and pe < tol * 1.6)
            if ok and aligned:
                fire[i] = True
        return [yaw, pit], fire, best, cand[best][5]

    def weapon_value(self, kind, tick):
        base = {"laser": 1.6, "lightning": 1.4, "mortar": 1.0}.get(kind, 0.0)
        beaten = BEATS.get(kind)
        beater = COUNTER.get(kind)
        return base + 0.25 * min(3, self.enemy_weapon_count(beaten, tick)) - 0.25 * min(3, self.enemy_weapon_count(beater, tick))

    # ------------------------------------------------------------------ evasion
    def evade(self, me, x, y, z, vx, vy, tick, enemies, projectiles, sounds):
        """Dodge bullets/shells; when shot at, run sideways in a randomised, held (staggered) pattern."""
        vec = [0.0, 0.0]
        reasons = []
        jump = False
        cz = z + 4.5
        for pr in projectiles:
            pp, pv = pr["pos"], pr["vel"]
            if pr["kind"] == "mortar":
                disc = pv[2] ** 2 + 2 * self.mortar_g * (pp[2] - z)
                if disc < 0:
                    continue
                tt = (pv[2] + math.sqrt(disc)) / self.mortar_g
                if 0 < tt < 2.5:
                    ix, iy = pp[0] + pv[0] * tt, pp[1] + pv[1] * tt
                    if hdist((ix, iy), (x, y)) < 12:
                        ax, ay = unit(x - ix, y - iy)
                        vec[0] += ax * 1.5
                        vec[1] += ay * 1.5
                        reasons.append("shell")
                continue
            r = (pp[0] - x, pp[1] - y, pp[2] - cz)
            v = (pv[0] - vx, pv[1] - vy, pv[2])
            v2 = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
            if v2 < 1:
                continue
            tc = -(r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) / v2   # time of closest approach
            if 0 < tc < 0.9:
                q = (r[0] + v[0] * tc, r[1] + v[1] * tc, r[2] + v[2] * tc)
                if math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2) < 5.5:
                    nx, ny = unit(-pv[1], pv[0])
                    s = nx * q[0] + ny * q[1]
                    sgn = (-1.0 if s > 0 else 1.0) if abs(s) > 0.5 else float(self.sign)
                    w = min(4.0, 1.8 / (0.3 + tc))
                    vec[0] += nx * sgn * w
                    vec[1] += ny * sgn * w
                    reasons.append("bullet")
                    if tc < 0.35:
                        jump = True
        # staggered evasion: only when something is really aiming at / hitting us
        base = None
        hit = self.last_hit
        if tick - hit["tick"] <= 12 and hit["weapon"] != "zone" and hit["bearing"] is not None:
            base = hit["bearing"]
            reasons.append("hit")
        aimed = None
        for d, e in enemies:
            if d < 170:
                to_me = math.atan2(y - e["pos"][1], x - e["pos"][0])
                if abs(wrap(e["aim"][0] - to_me)) < 0.25 and (aimed is None or d < aimed[0]):
                    aimed = (d, e)
        if aimed is not None:
            base = math.atan2(aimed[1]["pos"][1] - y, aimed[1]["pos"][0] - x)
            reasons.append("aimed")
        else:
            for s_ in sounds:
                if s_.get("kind") == "laser_charge" and s_.get("dist") in ("near", "mid") and s_.get("age", 9) < 0.6:
                    base = s_["bearing"]
                    reasons.append("laser_snd")
                    break
        if base is not None:
            if tick >= self.evade_until or self.evade_dir is None:
                best, besta = -9.0, None
                for side in (1, -1):
                    a = base + side * math.pi / 2 + self.rng.uniform(-0.45, 0.45)
                    sc = (self.grid.free_dist(x, y, a, 25.0, 4.0, 3.0) / 25.0) if self.grid else 1.0
                    sc += 0.3 if side == self.sign else 0.0
                    sc += self.rng.uniform(0.0, 0.25)       # a little randomness so we are not predictable
                    if sc > best:
                        best, besta = sc, a
                self.evade_dir = besta
                self.evade_until = tick + self.rng.randint(4, 8)   # hold 0.4 - 0.8 s, then re-roll
                self.evade_speed = self.rng.choice((1.0, 0.85, 0.7))
                self.sign = 1 if math.sin(besta - base) > 0 else -1
            vec[0] += math.cos(self.evade_dir) * self.evade_speed * 1.2
            vec[1] += math.sin(self.evade_dir) * self.evade_speed * 1.2
        else:
            self.evade_dir = None
        return vec, reasons, jump

    # ------------------------------------------------------------------ supply drops
    def update_drop(self, obs, pos, my_id, tick, walk, danger_eff, zone_urgent):
        z = obs["zone"]
        nc, nr = z["next_center"], z["next_radius"]
        byid = {d["id"]: d for d in obs["drops"]}
        if self.drop_id is not None and self.drop_id not in byid:
            self.drop_id = None
            self.drop_mode = None
        if self.drop_id is not None:
            dr = byid[self.drop_id]
            reason = None
            if danger_eff > self.p["drop_danger"] * 1.3:
                reason = "danger"
            elif zone_urgent == 2:
                reason = "zone"
            else:
                holder = dr.get("holder")
                if holder is not None and holder != my_id:
                    self.drop_contest += 1
                    if self.drop_contest > 50:
                        reason = "contested"
                else:
                    self.drop_contest = 0
            if reason:
                self.drop_black[self.drop_id] = tick + 200
                self.drop_abandon = reason
                self.drop_id = None
                self.drop_mode = None
                return None
        if self.drop_id is None:
            best = None
            for dr in obs["drops"]:
                if self.drop_black.get(dr["id"], 0) > tick:
                    continue
                dd = hdist(dr["pos"], pos)
                if dd > self.p["drop_range"] or hdist(dr["pos"], nc) > nr + 8:
                    continue
                if dr["lands_in"] > dd * 1.3 / max(walk, 5.0) + 25:
                    continue
                if danger_eff > self.p["drop_danger"]:
                    continue
                if best is None or dd < best[0]:
                    best = (dd, dr)
            if best is not None:
                self.drop_id = best[1]["id"]
                self.drop_mode = "approach"
                self.drop_contest = 0
        if self.drop_id is None:
            return None
        dr = byid[self.drop_id]
        dd = hdist(dr["pos"], pos)
        if dd < dr["radius"] - 2.0:
            self.drop_mode = "capture"
        elif self.drop_mode == "capture" and dd > dr["radius"] + 1.0:
            self.drop_mode = "approach"
        return {"mode": self.drop_mode, "drop": dr, "dist": dd}

    # ------------------------------------------------------------------ upgrades
    def pick_upgrade(self, offer):
        order = self.p["upgrades"]
        best_i, best_r = 0, 99
        for i, ch in enumerate(offer["choices"]):
            txt = (ch.get("id", "") + " " + ch.get("name", "") + " " + ch.get("desc", "")).lower()
            if any(w in txt for w in ("taken", "armor", "hp", "health", "regen", "repair")):
                cat = "def"
            elif any(w in txt for w in ("energy", "speed", "boost", "jet", "jump")):
                cat = "mob"
            elif any(w in txt for w in ("sensor", "vision", "sound", "range")):
                cat = "sense"
            else:
                cat = "off"
            r = order.index(cat) if cat in order else 50
            if r < best_r:
                best_i, best_r = i, r
        return best_i

    # ------------------------------------------------------------------ main decision
    def act(self, obs):
        self.T0 = time.perf_counter()
        tick, t = obs["tick"], obs["t"]
        me = obs["self"]
        x, y, z = me["pos"]
        vx, vy = me["vel"][0], me["vel"][1]
        pos = (x, y)
        st = me["stats"]
        walk = st.get("walk_speed", 11.7)
        hpf = me["hp"] / max(1.0, me["max_hp"])
        en = me["energy"]
        zone = obs["zone"]
        my_id = me.get("id", self.slot)
        p = self.p
        act = {"tick": tick}

        # ---- events
        dm, evlog = [], []
        for k in list(self.dmg_mem):
            self.dmg_mem[k] *= 0.985
            if self.dmg_mem[k] < 5:
                del self.dmg_mem[k]
        for ev in obs["events"]:
            ty = ev.get("type")
            if ty == "damaged":
                dm.append([ev.get("weapon"), round(ev.get("amount", 0)), 1 if ev.get("rear") else 0])
                self.last_hit = {"tick": tick, "bearing": ev.get("bearing"), "by": ev.get("by"),
                                 "rear": bool(ev.get("rear")), "weapon": ev.get("weapon")}
                if ev.get("by") is not None:
                    self.dmg_mem[ev["by"]] = self.dmg_mem.get(ev["by"], 0.0) + ev.get("amount", 0)
            elif ty == "kill":
                self.kills += 1
                evlog.append("kill")
            elif ty in ("take_failed", "took", "drop", "upgrade", "effect", "dropped"):
                evlog.append(ty + (":" + str(ev.get("reason") or ev.get("effect") or ev.get("id") or "")))

        # ---- upgrade offer
        if obs["offer"]:
            act["pick"] = self.pick_upgrade(obs["offer"])

        # ---- who is around
        held = [w["kind"] for w in me["weapons"] if w]
        enemies = []
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                enemies.append((hdist(m["pos"], pos), m))
                self.enemy_w[m["id"]] = (tick, m.get("weapons") or [])
        enemies.sort(key=lambda a: a[0])
        npcs = [(hdist(m["pos"], pos), m) for m in obs["mechs"] if m["kind"] == "npc"]
        crates = [(hdist(c["pos"], pos), c) for c in obs["crates"]]
        near = [(d, e) for d, e in enemies if d < 130]
        alive = obs.get("alive", 16)

        # ---- danger: sum of threats, scaled up when we are weak
        danger = 0.0
        for d, e in enemies:
            if d < 200:
                danger += self.threat(e, d, pos, (vx, vy), tick)
        danger_eff = danger * (1.7 - hpf)
        thr = p["flee_danger"]
        if p["adaptive"]:
            thr *= 1.15 if (hpf > 0.8 and alive > 6) else (0.85 if alive <= 3 else 1.0)
        want_flee = danger_eff > thr or (hpf < p["flee_hp"] and near and danger > 0.4)
        if want_flee:
            self.flee_until = tick + 8        # once we run, keep running a moment (no dithering)
        flee = tick < self.flee_until

        # ---- zone: how long until the edge reaches us, and can we get to safety in time (real path length)?
        nc, nr = zone["next_center"], zone["next_radius"]
        texit = self.zone_exit_time(zone, x, y, t)
        zone_urgent, zone_margin, eta = 0, None, None
        zgoal = self.inside_point(nc, nr, x, y, 0.6)
        wp_z = zgoal
        if texit < 150 or not me["in_zone"]:
            wp_z, rem = self.route("zone", x, y, zgoal, tick)
            eta = rem / max(walk * 0.9, 5.0) + 1.0
            zone_margin = texit - eta
            if not me["in_zone"] or zone_margin < 0:
                zone_urgent = 2
            elif zone_margin < p["zone_margin"]:
                zone_urgent = 1

        # ---- choose a target and a weapon
        tgt, tgt_u, tgt_los = None, -9.0, True
        for i, (d, e) in enumerate(enemies):
            if d > 235:
                break
            los = (self.grid.los((x, y, z + 7), (e["pos"][0], e["pos"][1], e["pos"][2] + 4.5)) if (self.grid and i < 3) else True)
            u = self.target_util(e, d, los, held, enemies, tick)
            if u > tgt_u:
                tgt, tgt_u, tgt_los = (d, e), u, los
        farm = None
        if tgt is None and hpf > 0.5 and danger_eff < 0.5:
            opts = [(d, m, "npc") for d, m in npcs if d < p["farm_range"]] + [(d, c, "crate") for d, c in crates if d < p["farm_range"] * 0.8]
            if opts:
                farm = min(opts, key=lambda o: o[0])
        aim, fire, wname = None, [False, False], "-"
        if tgt:
            aim, fire, wname, _ = self.solve_weapons(me, tgt[1], tgt[0], True, danger)
        elif farm:
            fe = dict(farm[1])
            fe.setdefault("vel", (0, 0, 0))
            aim, fire, wname, _ = self.solve_weapons(me, fe, farm[0], False, danger)
        elif tick - self.last_hit["tick"] < 15 and self.last_hit["bearing"] is not None:
            aim = [self.last_hit["bearing"], 0.0]       # turn to face whoever shot us (rear hits do x1.5)
            wname = "face_hit"
        else:
            for s_ in obs["sounds"]:
                if s_["kind"] in ("gun", "mortar", "laser", "lightning", "laser_charge") and s_["dist"] != "far" and s_["age"] < 0.6:
                    aim = [s_["bearing"], 0.0]
                    wname = "face_snd"
                    break

        # ---- weapon pickups: value each one, take the best if we are standing on it
        free_slot = next((i for i, w in enumerate(me["weapons"]) if w is None), None)
        special_slots = [(i, w["kind"]) for i, w in enumerate(me["weapons"]) if w and w["kind"] != "gun" and w.get("charge", 0) <= 0]
        worst = min(special_slots, key=lambda s: self.weapon_value(s[1], tick)) if special_slots else None
        worst_val = self.weapon_value(worst[1], tick) if worst else -9
        loot_goal, loot_val = None, 0.0
        for pk in obs["pickups"]:
            if pk["kind"] == "gun" or pk["kind"] in held:
                continue
            val = self.weapon_value(pk["kind"], tick)
            if free_slot is None and val < worst_val + 0.3:
                continue
            if free_slot is None and worst is None:
                continue
            d = hdist(pk["pos"], pos)
            if d < self.reach - 0.4 and abs(pk["pos"][2] - z) < 3.5:
                if not (near and near[0][0] < 30):
                    act["take"] = pk["id"]
                    if free_slot is None:
                        act["drop"] = worst[0]
                    break
            elif d < p["loot_range"]:
                score = val - d / 60.0
                if score > 0.5 and (loot_goal is None or score > loot_val):
                    loot_goal, loot_val = pk["pos"], score

        # ---- decide the state and the movement wish
        desired = [0.0, 0.0]

        def add(v, w):
            desired[0] += v[0] * w
            desired[1] += v[1] * w

        state, why, boost, cap, goal_pt = "HOLD", "", False, 1.0, None
        dropinfo = None
        if not flee:
            dropinfo = self.update_drop(obs, pos, my_id, tick, walk, danger_eff, zone_urgent)
        engaged = bool(tgt) and tgt_u > p["engage_min"] and tgt[0] < p["chase"]

        if zone_urgent:
            add(unit(wp_z[0] - x, wp_z[1] - y), 2.2 if zone_urgent == 2 else 1.3)
            goal_pt = zgoal
            if zone_urgent == 2 and en > 5:
                boost = True
            elif zone_urgent == 1 and en > 60 and eta is not None and eta > 12:
                boost = True
        if flee:
            fx = fy = 0.0
            for d, e in near:
                ax, ay = unit(x - e["pos"][0], y - e["pos"][1])
                w = 1.0 / max(d, 20.0)
                fx += ax * w
                fy += ay * w
            add(unit(fx, fy), 1.5)
            add(unit(nc[0] - x, nc[1] - y), 0.5)       # run toward the safe circle, never out of it
            state, why = "FLEE", "danger %.1f hp %.2f" % (danger_eff, hpf)
            if en > 20:
                boost = True
        elif zone_urgent:
            state, why = "ZONE", "margin %s" % (round(zone_margin, 1) if zone_margin is not None else "out")
        elif dropinfo is not None:
            dr = dropinfo["drop"]
            goal_pt = dr["pos"]
            if dropinfo["mode"] == "capture":
                state, why = "DROP_CAPTURE", "prog %.1f/%.0f" % (dr.get("progress", 0), dr.get("capture_s", 4))
                to = (dr["pos"][0] - x, dr["pos"][1] - y)
                dd = math.hypot(to[0], to[1])
                if dd > 2.0:
                    add(unit(to[0], to[1]), min(1.0, dd / 6.0))
            else:
                wp, rem = self.route("main", x, y, dr["pos"], tick)
                add(unit(wp[0] - x, wp[1] - y), 1.0)
                state, why = "DROP_GO", "%dm lands %.0fs" % (rem, dr["lands_in"])
                if rem > 80 and en > 70:
                    boost = True
        elif engaged:
            e = tgt[1]
            d = tgt[0]
            ux, uy = unit(e["pos"][0] - x, e["pos"][1] - y)
            pref = self.pref_dist(held, e)
            radial = clamp((d - pref) / 25.0, -1.0, 1.0)
            ehp = e.get("hp", 1000.0) / max(1.0, e.get("max_hp", 1000.0))
            if radial > 0 and not (hpf > 0.5 and ehp < hpf + 0.3 + (0.2 if p["pref_mul"] < 1 else 0)):
                radial = 0.0                       # do not walk toward someone who is healthier than us
            add((ux, uy), radial)
            state, why = "FIGHT", "tgt %d u%.1f %s" % (e["id"], tgt_u, wname)
            if radial > 0.5 and d > 120:
                wp, _ = self.route("main", x, y, e["pos"], tick)
                add(unit(wp[0] - x, wp[1] - y), 0.6)
        elif loot_goal is not None and danger_eff < 0.6 * thr:
            wp, rem = self.route("main", x, y, loot_goal, tick)
            add(unit(wp[0] - x, wp[1] - y), 1.0)
            goal_pt = loot_goal
            state, why = "LOOT", "%dm" % rem
        else:
            cube = None
            for c in obs["cubes"]:
                dd = hdist(c["pos"], pos)
                if dd < p["cube_range"] and (cube is None or dd < cube[0]):
                    cube = (dd, c["pos"])
            if cube is not None and danger_eff < 0.6 * thr:
                wp, rem = self.route("main", x, y, cube[1], tick)
                add(unit(wp[0] - x, wp[1] - y), 1.0)
                goal_pt = cube[1]
                state, why = "CUBE", "%dm" % cube[0]
            elif farm is not None and farm[0] > 30:
                wp, rem = self.route("main", x, y, farm[1]["pos"], tick)
                add(unit(wp[0] - x, wp[1] - y), 0.9)
                goal_pt = farm[1]["pos"]
                state, why = "FARM", farm[2]
            else:
                # nothing to do: drift toward the middle of the next circle, slowly (slow steps are silent)
                if hdist(pos, nc) > 0.5 * nr:
                    add(unit(nc[0] - x, nc[1] - y), 0.4)
                    cap = 0.4
                state, why = "HOLD", ""

        # strafing while we shoot at someone (not while capturing a drop or running)
        if tgt and state in ("FIGHT", "ZONE", "LOOT", "CUBE", "FARM", "DROP_GO") and tgt[0] < 170:
            ux, uy = unit(tgt[1]["pos"][0] - x, tgt[1]["pos"][1] - y)
            add((-uy, ux), p["strafe"] * self.sign * (0.75 + 0.25 * self.rng.random()) * (0.5 if state == "ZONE" else 1.0))
        if tick >= self.flip_at:
            self.sign = -self.sign
            self.flip_at = tick + self.rng.randint(6, 18)
        # keep away from enemies when not fighting (defensive personality)
        if p["avoid_dist"] > 0 and state in ("HOLD", "LOOT", "CUBE", "FARM", "DROP_GO"):
            for d, e in enemies:
                if d < p["avoid_dist"]:
                    ax, ay = unit(x - e["pos"][0], y - e["pos"][1])
                    add((ax, ay), 1.0 * (1.0 - d / p["avoid_dist"]))
        # dodging
        ev_vec, ev_reasons, ev_jump = self.evade(me, x, y, z, vx, vy, tick, enemies, obs["projectiles"], obs["sounds"])
        if ev_reasons:
            add(ev_vec, p["evade_weight"])
            if en > 20 and ("bullet" in ev_reasons or "shell" in ev_reasons or "aimed" in ev_reasons):
                boost = boost or state != "DROP_CAPTURE"

        # ---- stuck handling (mag is computed AFTER both branches)
        stuck = False
        if tick < self.stuck_until:
            stuck = True
        elif len(self.hist) >= 10:
            h0 = self.hist[-10]
            if hdist((h0[1], h0[2]), pos) < 1.5 and min(hh[3] for hh in self.hist[-10:]) > 0.4 and state != "DROP_CAPTURE":
                ang = math.atan2(self.last_desired[1], self.last_desired[0]) + self.rng.choice((-1, 1)) * self.rng.uniform(1.2, 2.2)
                self.stuck_dir = (math.cos(ang), math.sin(ang))
                self.stuck_until = tick + 8
                self.paths.clear()
                self.hist = []
                self.stuck_events += 1
                stuck = True
        if stuck:
            move = self.stuck_dir
            act["jump"] = True
            if en > 10:
                boost = True
            why = (why + " STUCK").strip()
        else:
            move = self.steer(x, y, desired, zone["center"], zone["radius"] if me["in_zone"] else 0, cap)
        mag = math.hypot(move[0], move[1])
        if mag < 0.12:
            move, mag = (0.0, 0.0), 0.0
        elif not stuck:
            self.last_desired = unit(desired[0], desired[1])

        # while capturing a drop, never step out of the ring
        if dropinfo is not None and state == "DROP_CAPTURE":
            c, r = dropinfo["drop"]["pos"], dropinfo["drop"]["radius"]
            if hdist((x + move[0] * walk * 0.7, y + move[1] * walk * 0.7), c) > r - 1.5:
                tx, ty = unit(c[0] - x, c[1] - y)
                move = (tx * 0.8, ty * 0.8)
                mag = 0.8
            boost = False

        self.hist.append((tick, x, y, mag if state != "DROP_CAPTURE" else 0.0))
        self.hist = self.hist[-12:]

        act["move"] = [move[0], move[1]]
        act["boost"] = bool(boost and mag > 0 and en > 5)
        if aim is None and mag > 0.3:
            aim = [math.atan2(move[1], move[0]), 0.0]
        if aim is not None:
            act["aim"] = aim
        if fire[0] or fire[1]:
            act["fire"] = fire
        if ev_jump and tick - self.last_jump > 8 and me.get("grounded", True):
            act["jump"] = True
            self.last_jump = tick
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        if gun and not (tgt and tgt[0] < 120) and gun.get("reload", 0) <= 0 and gun.get("ammo", 99) <= 3:
            act["reload"] = True

        # ---- logging and debug drawing
        rec = {"k": tick, "t": round(t, 1), "hp": round(me["hp"]), "en": round(en), "p": [round(x), round(y)],
               "al": alive, "s": state, "w": why, "tg": tgt[1]["id"] if tgt else None,
               "td": round(tgt[0]) if tgt else None, "th": round(danger, 2), "ne": len(near),
               "zm": round(zone_margin, 1) if zone_margin is not None else None, "wp": "".join(k[0] for k in held),
               "ms": round((time.perf_counter() - self.T0) * 1000, 1), "iz": 1 if me["in_zone"] else 0}
        if dm:
            rec["dm"] = dm
        if evlog:
            rec["ev"] = evlog
        if ev_reasons:
            rec["evd"] = ",".join(sorted(set(ev_reasons)))
        if stuck:
            rec["stk"] = self.stuck_events
        if self.drop_abandon:
            rec["dab"] = self.drop_abandon
            self.drop_abandon = ""
        if self.log_stderr:
            mf.log(json.dumps(rec, separators=(",", ":")))
        self.draw(tick, rec, tgt, goal_pt, x, y, z, dm or evlog)
        return act

    def draw(self, tick, rec, tgt, goal_pt, x, y, z, notable):
        d = mf.draw
        d.label("%s %s hp%d" % (self.name[:1], rec["s"], rec["hp"]))
        if tgt:
            d.line([x, y, z + 7], [tgt[1]["pos"][0], tgt[1]["pos"][1], tgt[1]["pos"][2] + 4.5], "#ff5040")
        if goal_pt is not None:
            d.point([goal_pt[0], goal_pt[1], z], "#40c0ff", 6)
        if self.draw_path:
            pts = self.draw_path[:10]
            for a, b in zip(pts, pts[1:]):
                d.line([a[0], a[1], z], [b[0], b[1], z], "#40ff80")
            self.draw_path = None
        if self.p["record_debug"] and (tick % 2 == 0 or notable):
            s = json.dumps(rec, separators=(",", ":"))
            for i in range(0, min(len(s), 7 * 66), 66):
                d.text("#%d.%d:%s" % (tick, i // 66, s[i:i + 66]))


def main(cfg):
    mf.run(lambda hello: Brain(hello, cfg))


if __name__ == "__main__":
        main(BASE)
