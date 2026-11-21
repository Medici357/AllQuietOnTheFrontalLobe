"""Survivor bot for mechfight/1.

Ideas (placement counts, not kills, so staying alive comes first):
  1. Never stand outside the zone; move to the next circle BEFORE it shrinks.
  2. Build a coarse "can I walk here?" grid from the map files and path-find around buildings.
  3. Aim with lead (predict where the enemy will be), fire only when the torso really points there.
  4. Keep moving sideways (strafe), dodge bullets, shells and charged-laser lines.
  5. Face enemies (hits on the back do x1.5 damage) while the legs run somewhere else.
  6. Grab a second weapon, cubes, supply drops; choose defensive upgrades.
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


class Bot:
    name = "survivor-py"

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
            if t >= z["shrink_start"] or z["shrink_start"] - t < travel + 15:
                return self.inside_point(nc, nr, x, y, 0.6), 1
        return None, 0

    # ---------------- weapons ----------------
    def shoot(self, me, tpos, tvel, special):
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
            ok = d < st.get("range", 230) * 0.92 and w.get("ammo", 1) > 0 and w.get("reload", 0) <= 0
            cand["gun"] = (yaw, pit, ok, i)
        if special and "laser" in held:
            i, w = held["laser"]
            yaw, pit = angles(o, lead(o, tc, tvel, 1e9, 0.25))
            ok = d < 170 and (w["cooldown"] <= 0.35 or w.get("charge", 0) > 0)
            cand["laser"] = (yaw, pit, ok, i)
        if special and "lightning" in held:
            i, w = held["lightning"]
            yaw, pit = angles(o, tc)
            ok = d < w["stats"].get("range", 45) - 3 and me["energy"] > 12
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
                cand["mortar"] = (yaw, pit, 22 < d < 200 and w["cooldown"] <= 0.5 and slow, i)
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
                tol = tol0 * 1.8  # already charging: keep holding unless badly off
            ye, pe = abs(wrap(cur[0] - cy)), abs(cur[1] - cp)
            if k == "lightning":
                aligned = ye < 0.4
            else:
                aligned = ye < tol and pe < tol * 1.6
            if k == "laser":
                ch = held["laser"][1].get("charge", 0)
                if ch >= 0.7:
                    continue  # fully charged: release the trigger so the beam fires now
                if ch > 0 and (d > 190 or not ok and ch < 0.15):
                    continue
            if ok and aligned:
                fire[i] = True
        return [yaw, pit], fire

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

        for ev in obs["events"]:
            if ev.get("type") == "damaged":
                self.hit_tick = tick
                if ev.get("bearing") is not None:
                    self.hit_bearing = ev["bearing"]
                if self.rng.random() < 0.5:
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
            if d < 70 and (loot_wp is None or d < loot_wp[0]):
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
            aim, fire = self.shoot(me, tgt[1]["pos"], tgt[1]["vel"], True)
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
        return act


mf.run(Bot)