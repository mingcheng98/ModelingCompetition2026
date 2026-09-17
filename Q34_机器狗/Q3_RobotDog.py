# -*- coding: utf-8 -*-
"""
机器狗程序(问题3): 全向干扰源的自动搜索定位与清除
与官方模拟器通信(HTTP+JSON, 四条指令 /enter /measure /clear /exit), 
也可对接自建模拟器(端口不同)。
直接运行时连接由 Simulator.py 准备的本地无端口会话；提供 --robot-id
参数时仍按原方式连接官方 HTTP 模拟器。

策略概要: 
  第一阶段 覆盖扫描: 9个探测点(原点+原点两侧300米两点+半径1254.8米环形6点), 
    任意位置干扰源与某探测点距离≤950米<1000米, 保证必被至少一个探测点检测到。
    每个探测点按频道升序检测(已清除或已有2条良好交会示向线的频道跳过)。
  第二阶段 粗定位: 对每条频道的示向线扇形做半平面交, 取交会多边形质心为位置估计; 
    仅1条示向线或交会角不佳时, 在示向点45°方向707米处补测一点(保证接收), 
    仍失败则按示向线远方点估计。
  第三阶段 精细清除: 按最近邻顺序前往估计点, 先直接尝试清除(20米半径); 
    失败则沿测得示向线步进并逐点尝试清除("near"直接清除), 直至成功。
"""
import json
import os
import time
import math
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ---------- 配置 ----------
BASE_URL = 'http://127.0.0.1:2026'
SPEED = 5.0                       # 移动速度(米/秒)
SCAN_R = 950.0                    # 覆盖裕量(<1000)
RING_R = 1254.8                   # 环形扫描半径
CENTER_OFF = 300.0                # 原点两侧辅助探测点偏移
GOOD_SIN = 0.35                   # 交会角质量阈值
LOCALIZATION_QUALITY_THRESHOLD = 200.0  # 定位质量阈值(交会多边形直径, 米)
PROBE_T = 500.0                   # 补测点沿示向线偏移
PROBE_L = 500.0                   # 补测点垂直偏移

def scan_grid():
    """9个扫描探测点: 原点 + 原点两侧 + 环形6点"""
    pts = [(0.0, 0.0), (CENTER_OFF, 0.0), (-CENTER_OFF, 0.0)]
    for k in range(6):
        a = math.radians(k*60)
        pts.append((RING_R*math.cos(a), RING_R*math.sin(a)))
    return pts

# ---------- HTTP 客户端 ----------
class SimClient:
    """官方协议客户端，也支持注入本地后端进行无网络测试。

    ``backend`` 只需提供 ``post(path, payload)`` 方法并返回协议响应，
    这样策略无需修改即可接入本地模拟器；未提供时仍使用官方 HTTP 接口。
    """
    def __init__(self, base_url=BASE_URL, robot_id=None, backend=None):
        if not robot_id:
            raise ValueError('必须提供参赛队号 robot_id')
        self.base_url = base_url
        self.robot_id = robot_id
        self.backend = backend
        self.req_seq = 0
        self.log = []          # 请求/响应日志
        self.move_t = 0.0      # 移动耗时
        self.switch_t = 0.0    # 频道切换耗时
        self.measure_t = 0.0   # 检测耗时
        self.clear_ok_t = 0.0  # 清除成功耗时
        self.clear_fail_t = 0.0
        self.virtual_time = 0.0
        self.pos = (0.0, 0.0)
        self.channel = 1

    def _post(self, path, payload, retries=6):
        self.req_seq += 1
        if self.backend is not None:
            resp = self.backend.post(path, payload)
            if resp is None:
                raise RuntimeError(f'{path} 本地模拟器未返回响应')
            self.log.append(dict(path=path, payload=str(payload), resp=str(resp),
                                 virtual_time=resp.get('virtual_time_s', 0)))
            if not resp.get('accepted', False):
                raise RuntimeError(f'{path} 未执行: {resp}')
            self.virtual_time = float(resp['virtual_time_s'])
            return resp
        data = json.dumps(payload).encode('utf-8')
        for att in range(retries):
            try:
                req = Request(self.base_url + path, data=data,
                              headers={'Content-Type': 'application/json'}, method='POST')
                with urlopen(req, timeout=10) as r:
                    resp = json.loads(r.read().decode('utf-8'))
                    break
            except (URLError, HTTPError, TimeoutError, OSError) as e:
                if isinstance(e, HTTPError):
                    try:
                        resp = json.loads(e.read().decode('utf-8'))
                        break
                    except Exception:
                        resp = None
                time.sleep(0.2*(att+1))
                resp = None
        else:
            resp = None
        if resp is None:
            raise RuntimeError(f'{path} 请求失败')
        self.log.append(dict(path=path, payload=str(payload), resp=str(resp),
                             virtual_time=resp.get('virtual_time_s', 0)))
        if not resp.get('accepted', False):
            raise RuntimeError(f'{path} 未执行: {resp}')
        self.virtual_time = float(resp['virtual_time_s'])
        return resp

    def enter(self):
        r = self._post('/enter', dict(arena_id='default', robot_id=self.robot_id,
                                      request_id=f'e{self.req_seq}'))
        self.remaining_real = r['remaining_real_duration_s']
        return r

    def measure(self, x, y, ch):
        move_t = math.hypot(x-self.pos[0], y-self.pos[1])/SPEED
        switch_t = 1.0 if ch != self.channel else 0.0
        r = self._post('/measure', dict(arena_id='default', robot_id=self.robot_id,
                                        request_id=f'm{self.req_seq}',
                                        position={'x': x, 'y': y}, channel=ch))
        self.pos = (x, y)
        self.channel = ch
        self.move_t += move_t
        self.switch_t += switch_t
        self.measure_t += 5.0
        return r, move_t, switch_t

    def clear(self, x, y, ch):
        move_t = math.hypot(x-self.pos[0], y-self.pos[1])/SPEED
        r = self._post('/clear', dict(arena_id='default', robot_id=self.robot_id,
                                      request_id=f'c{self.req_seq}',
                                      position={'x': x, 'y': y}, channel=ch))
        self.pos = (x, y)
        self.move_t += move_t
        if r.get('clear_result') == 'success':
            self.clear_ok_t += 5.0
        else:
            self.clear_fail_t += 3.0
        return r, move_t

    def exit(self):
        r = self._post('/exit', dict(arena_id='default', robot_id=self.robot_id,
                                     request_id=f'x{self.req_seq}'))
        return r

# ---------- 几何工具 ----------
def _wedge_hps(S, theta_deg):
    t = math.radians(theta_deg)
    e = math.radians(1.0)
    u1 = (math.cos(t-e), math.sin(t-e))
    u2 = (math.cos(t+e), math.sin(t+e))
    return [(u1[1], -u1[0], u1[1]*S[0]-u1[0]*S[1]),
            (-u2[1], u2[0], -u2[1]*S[0]+u2[0]*S[1])]

def _clip(poly, hp, eps=1e-9):
    a, b, c = hp
    if not poly:
        return []
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i+1) % n]
        fP, fQ = a*P[0]+b*P[1]-c, a*Q[0]+b*Q[1]-c
        if fP <= eps:
            out.append(P)
        if (fP <= eps) != (fQ <= eps):
            t = fP/(fP-fQ)
            out.append((P[0]+t*(Q[0]-P[0]), P[1]+t*(Q[1]-P[1])))
    cl = []
    for p in out:
        if not cl or math.hypot(p[0]-cl[-1][0], p[1]-cl[-1][1]) > 1e-6:
            cl.append(p)
    if len(cl) > 1 and math.hypot(cl[0][0]-cl[-1][0], cl[0][1]-cl[-1][1]) <= 1e-6:
        cl.pop()
    return cl

def localization_polygon(bearings, cap=2000.0):
    """bearings: [(x, y, svd_deg), ...] -> 交会多边形顶点列表(含区域边界裁剪)"""
    poly = [(-cap, -cap), (cap, -cap), (cap, cap), (-cap, cap)]
    for x, y, th in bearings:
        for hp in _wedge_hps((x, y), th):
            poly = _clip(poly, hp)
            if len(poly) < 3:
                return []
    if len(poly) < 3:
        return []
    return poly

def poly_centroid(verts):
    A = 0.0; cx = 0.0; cy = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]; x2, y2 = verts[(i+1) % n]
        f = x1*y2 - x2*y1
        A += f; cx += (x1+x2)*f; cy += (y1+y2)*f
    if abs(A) < 1e-12:
        return (sum(p[0] for p in verts)/n, sum(p[1] for p in verts)/n)
    return (cx/(3*A), cy/(3*A))

def poly_diameter(verts):
    best = 0.0
    n = len(verts)
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(verts[i][0]-verts[j][0], verts[i][1]-verts[j][1])
            best = max(best, d)
    return best

def best_cross_sin(bearings):
    """两两示向线的交会角正弦最大值(>GOOD_SIN认为交会良好)"""
    best = 0.0
    n = len(bearings)
    for i in range(n):
        for j in range(i+1, n):
            P = _line_intersect(bearings[i], bearings[j])
            if P is None:
                continue
            a1 = math.atan2(P[1]-bearings[i][1], P[0]-bearings[i][0])
            a2 = math.atan2(P[1]-bearings[j][1], P[0]-bearings[j][0])
            best = max(best, abs(math.sin(a1-a2)))
    return best

def _line_intersect(b1, b2):
    x1, y1, t1 = b1; x2, y2, t2 = b2
    u1 = (math.cos(math.radians(t1)), math.sin(math.radians(t1)))
    u2 = (math.cos(math.radians(t2)), math.sin(math.radians(t2)))
    det = u1[0]*u2[1] - u1[1]*u2[0]
    if abs(det) < 1e-9:
        return None
    dx, dy = x2-x1, y2-y1
    s = (dx*u2[1] - dy*u2[0])/det
    return (x1 + s*u1[0], y1 + s*u1[1])

def nearest_order(points, start):
    """最近邻排序"""
    remain = list(points)
    order = []
    cur = start
    while remain:
        nxt = min(remain, key=lambda p: math.hypot(p[0]-cur[0], p[1]-cur[1]))
        order.append(nxt)
        remain.remove(nxt)
        cur = nxt
    return order

def optimize_open_route(points, start=(0.0, 0.0)):
    """最近邻初始化+2-opt, 优化不要求回到起点的访问路径。"""
    route = nearest_order(list(points), start)
    if len(route) < 3:
        return route
    def length(seq):
        cur, total = start, 0.0
        for p in seq:
            total += math.hypot(p[0]-cur[0], p[1]-cur[1])
            cur = p
        return total
    best = length(route)
    improved = True
    while improved:
        improved = False
        for i in range(len(route)-1):
            for j in range(i+1, len(route)):
                cand = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                value = length(cand)
                if value + 1e-7 < best:
                    route, best, improved = cand, value, True
    return route

def minimum_enclosing_circle(points):
    """小规模凸多边形的确定性最小包围圆, 返回((x,y), radius)。"""
    pts = list(points)
    if not pts:
        return ((0.0, 0.0), float('inf'))
    def contains(circle, p):
        (cx, cy), r = circle
        return math.hypot(p[0]-cx, p[1]-cy) <= r + 1e-7
    candidates = [((p[0], p[1]), 0.0) for p in pts]
    for i, a in enumerate(pts):
        for b in pts[i+1:]:
            c = ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)
            candidates.append((c, math.hypot(a[0]-b[0], a[1]-b[1])/2.0))
    for i, a in enumerate(pts):
        for j in range(i+1, len(pts)):
            b = pts[j]
            for c in pts[j+1:]:
                d = 2*(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
                if abs(d) < 1e-10:
                    continue
                ux = ((a[0]**2+a[1]**2)*(b[1]-c[1]) +
                      (b[0]**2+b[1]**2)*(c[1]-a[1]) +
                      (c[0]**2+c[1]**2)*(a[1]-b[1]))/d
                uy = ((a[0]**2+a[1]**2)*(c[0]-b[0]) +
                      (b[0]**2+b[1]**2)*(a[0]-c[0]) +
                      (c[0]**2+c[1]**2)*(b[0]-a[0]))/d
                candidates.append(((ux, uy), math.hypot(ux-a[0], uy-a[1])))
    feasible = [z for z in candidates if all(contains(z, p) for p in pts)]
    return min(feasible, key=lambda z: z[1])

# ---------- 策略主体 ----------
class StrategyQ3:
    def __init__(self, client):
        self.c = client
        self.trace = dict(actions=[], sources={})   # sources[ch] = 事件记录

    def run(self):
        c = self.c
        self.t0 = 0.0
        c.enter()
        self.cleared = set()
        self.bearings = {}      # ch -> [(x, y, svd)]
        self.first_heard = {}   # ch -> 虚拟时刻
        self.hear_point = {}    # ch -> (x, y)
        # ---------- 第一阶段: 覆盖扫描 ----------
        # 访问集合不变, 只优化开放路径; 不影响1000米覆盖完备性。
        for idx, (gx, gy) in enumerate(optimize_open_route(scan_grid(), c.pos)):
            act = sorted([ch for ch in range(1, 21) if ch not in self.cleared])
            for ch in act:
                if ch in self.bearings and self._scan_sufficient(self.bearings[ch]):
                    continue
                r, _, _ = c.measure(gx, gy, ch)
                self.trace['actions'].append(dict(kind='scan', x=gx, y=gy, ch=ch,
                                                  res=r.get('measure_result'),
                                                  svd=r.get('svd_deg'), t=c.virtual_time))
                if r['measure_result'] == 'near':
                    cr, _ = c.clear(gx, gy, ch)
                    self.trace['actions'].append(dict(kind='clear_near', x=gx, y=gy, ch=ch,
                                                      res=cr.get('clear_result'), t=c.virtual_time))
                    if cr['clear_result'] == 'success':
                        self.cleared.add(ch)
                        self.trace['sources'][ch] = dict(first_heard=c.virtual_time,
                                                         cleared_t=c.virtual_time, method='near')
                        self.first_heard[ch] = c.virtual_time
                elif r['measure_result'] == 'direction':
                    self.bearings.setdefault(ch, []).append((gx, gy, r['svd_deg']))
                    if ch not in self.first_heard:
                        self.first_heard[ch] = c.virtual_time
                        self.hear_point[ch] = (gx, gy)
        # ---------- 第二阶段: 补测与就地清除(弱交会源, 按空间最近邻) ----------
        weak = []
        good = []
        for ch in sorted(set(range(1, 21)) - self.cleared):
            if ch not in self.bearings:
                continue
            if self._well_localized(self.bearings[ch]):
                good.append(ch)
            else:
                weak.append(ch)
        remain_w = set(weak)
        cur = c.pos
        while remain_w:
            ch = min(remain_w, key=lambda cc: self._weak_dist(cc, cur))
            remain_w.remove(ch)
            if ch in self.cleared:
                continue
            if self._probe_and_kill(ch):
                cur = c.pos
                continue
            est, bound = self._far_estimate(ch)
            cur = est
            self._kill(ch, est, bound)
        # ---------- 第三阶段: 良好交会源最近邻清除 ----------
        estimates = {}
        for ch in good:
            poly = localization_polygon(self.bearings[ch])
            if poly:
                estimates[ch] = minimum_enclosing_circle(poly)
        point_to_ch = {estimates[ch][0]: ch for ch in estimates}
        order = [point_to_ch[p] for p in optimize_open_route(point_to_ch, c.pos)]
        for ch in order:
            if ch in self.cleared:
                continue
            est, bound = estimates[ch]
            self._kill(ch, est, bound)
        # ---------- 第四阶段: 完备性核查与退出 ----------
        for ch in range(1, 21):
            if ch in self.cleared:
                continue
            if ch in self.bearings:
                est, bound = self._far_estimate(ch)
                self._kill(ch, est, bound*2 + 200)
            # 未听到任何信号的频道: 无干扰源(覆盖扫描完备性保证)
        c.exit()
        return self._summary()

    def _well_localized(self, bearings):
        """交会多边形直径不超过定位质量阈值时, 视为定位良好。"""
        if len(bearings) < 2:
            return False
        poly = localization_polygon(bearings)
        if not poly:
            return False
        return poly_diameter(poly) <= LOCALIZATION_QUALITY_THRESHOLD

    def _scan_sufficient(self, bearings):
        """达到稳定交会后停止扫描; 精度不足交给后续近场补测。"""
        return self._well_localized(bearings)

    def _weak_dist(self, ch, cur):
        """弱交会源补测点与当前位置的距离(选靠近当前位置的一侧)"""
        if ch not in self.bearings:
            return 1e12
        x0, y0, th0 = self.bearings[ch][0]
        u = (math.cos(math.radians(th0)), math.sin(math.radians(th0)))
        best = 1e12
        for side in (1, -1):
            n = (-u[1]*side, u[0]*side)
            px, py = x0 + PROBE_T*u[0] + PROBE_L*n[0], y0 + PROBE_T*u[1] + PROBE_L*n[1]
            best = min(best, math.hypot(px-cur[0], py-cur[1]))
        return best

    def _probe_and_kill(self, ch):
        """到补测点补测并就地清除。返回True表示已处理"""
        c = self.c
        x0, y0, th0 = self.bearings[ch][0]
        u = (math.cos(math.radians(th0)), math.sin(math.radians(th0)))
        sides = sorted([1, -1],
                       key=lambda s: math.hypot(x0+PROBE_T*u[0]+PROBE_L*(-u[1]*s)-c.pos[0],
                                                y0+PROBE_T*u[1]+PROBE_L*(u[0]*s)-c.pos[1]))
        est, bound = None, None
        for side in sides:
            n = (-u[1]*side, u[0]*side)
            px, py = x0 + PROBE_T*u[0] + PROBE_L*n[0], y0 + PROBE_T*u[1] + PROBE_L*n[1]
            r, _, _ = c.measure(px, py, ch)
            self.trace['actions'].append(dict(kind='probe', x=px, y=py, ch=ch,
                                              res=r.get('measure_result'),
                                              svd=r.get('svd_deg'), t=c.virtual_time))
            if r['measure_result'] == 'near':
                cr, _ = c.clear(px, py, ch)
                if cr['clear_result'] == 'success':
                    self._mark(ch, c.virtual_time, 'probe_near')
                    return True
                return False
            if r['measure_result'] == 'direction':
                self.bearings.setdefault(ch, []).append((px, py, r['svd_deg']))
                poly = localization_polygon(self.bearings[ch])
                if poly:
                    est = poly_centroid(poly)
                    bound = poly_diameter(poly)/2
                break
        if est is None:
            est, bound = self._far_estimate(ch)
        self._kill(ch, est, bound)
        return True

    def _far_estimate(self, ch):
        """仅依首条示向线的远方点估计"""
        x0, y0, th0 = self.bearings[ch][0]
        u = (math.cos(math.radians(th0)), math.sin(math.radians(th0)))
        return (x0 + 1433*u[0], y0 + 1433*u[1]), 300.0

    def _kill(self, ch, est, bound):
        c = self.c
        t_kill = c.virtual_time
        ex, ey = est
        # 直接尝试清除
        r, _ = c.clear(ex, ey, ch)
        self.trace['actions'].append(dict(kind='clear', x=ex, y=ey, ch=ch,
                                          res=r.get('clear_result'), t=c.virtual_time))
        if r['clear_result'] == 'success':
            self._mark(ch, t_kill, 'direct')
            return
        # 检测当前点示向度
        r, _, _ = c.measure(ex, ey, ch)
        self.trace['actions'].append(dict(kind='fine_measure', x=ex, y=ey, ch=ch,
                                          res=r.get('measure_result'),
                                          svd=r.get('svd_deg'), t=c.virtual_time))
        if r['measure_result'] == 'near':
            cr, _ = c.clear(ex, ey, ch)
            if cr['clear_result'] == 'success':
                self._mark(ch, t_kill, 'near')
                return
        if r['measure_result'] == 'direction':
            if self._fine_fix(ch, ex, ey, r['svd_deg'], t_kill):
                return
            # 细定位失败: 从当前点沿示向线步进
            self._walk_kill(ch, (ex, ey), r['svd_deg'], max(bound, 30.0), t_kill)
            return
        # 无信号: 十字+螺旋兜底
        self._walk_kill(ch, (ex, ey), None, max(bound, 30.0), t_kill)

    def _fine_fix(self, ch, ex, ey, th, t_kill):
        """两点细定位: 垂直基线25米交会 -> 交点处清除。成功返回True"""
        c = self.c
        u = (math.cos(math.radians(th)), math.sin(math.radians(th)))
        n = (-u[1], u[0])
        for side in (1, -1):
            p2 = (ex + 25*n[0]*side, ey + 25*n[1]*side)
            r2, _, _ = c.measure(p2[0], p2[1], ch)
            self.trace['actions'].append(dict(kind='fine_measure', x=p2[0], y=p2[1], ch=ch,
                                              res=r2.get('measure_result'),
                                              svd=r2.get('svd_deg'), t=c.virtual_time))
            if r2['measure_result'] == 'near':
                cr, _ = c.clear(p2[0], p2[1], ch)
                if cr['clear_result'] == 'success':
                    self._mark(ch, t_kill, 'fine_near')
                    return True
            if r2['measure_result'] == 'direction':
                p3 = _line_intersect((ex, ey, th), (p2[0], p2[1], r2['svd_deg']))
                if p3 is not None and math.hypot(p3[0]-ex, p3[1]-ey) <= 120.0:
                    cr, _ = c.clear(p3[0], p3[1], ch)
                    self.trace['actions'].append(dict(kind='clear', x=p3[0], y=p3[1], ch=ch,
                                                      res=cr.get('clear_result'), t=c.virtual_time))
                    if cr['clear_result'] == 'success':
                        self._mark(ch, t_kill, 'fine_fix')
                        return True
                    # 清除失败: 以交点为新起点继续
                    ex, ey = p3[0], p3[1]
                    break
        return False

    def _walk_kill(self, ch, start, th, bound, t_kill):
        """沿示向线步进清除(每步尝试清除, 定期修正方向); th=None时十字+螺旋兜底"""
        c = self.c
        ex, ey = start
        if th is None:
            for dx, dy in [(25, 0), (-25, 0), (0, 25), (0, -25), (18, 18), (-18, 18), (18, -18), (-18, -18)]:
                cr, _ = c.clear(ex+dx, ey+dy, ch)
                if cr['clear_result'] == 'success':
                    self._mark(ch, t_kill, 'cross')
                    return
            for k in range(1, 6):
                rr = 25.0*k
                for ang in range(0, 360, 45):
                    cr, _ = c.clear(ex+rr*math.cos(math.radians(ang)),
                                    ey+rr*math.sin(math.radians(ang)), ch)
                    if cr['clear_result'] == 'success':
                        self._mark(ch, t_kill, 'spiral')
                        return
            self.trace['sources'][ch] = dict(first_heard=self.first_heard.get(ch, c.virtual_time),
                                             cleared_t=None, method='FAILED')
            return
        u = (math.cos(math.radians(th)), math.sin(math.radians(th)))
        step = max(15.0, min(40.0, max(bound/4, 15.0)))
        for k in range(1, 18):
            px, py = ex + k*step*u[0], ey + k*step*u[1]
            cr, _ = c.clear(px, py, ch)
            self.trace['actions'].append(dict(kind='walk_clear', x=px, y=py, ch=ch,
                                              res=cr.get('clear_result'), t=c.virtual_time))
            if cr['clear_result'] == 'success':
                self._mark(ch, t_kill, 'walk')
                return
            if k % 3 == 0:
                mr, _, _ = c.measure(px, py, ch)
                self.trace['actions'].append(dict(kind='walk_measure', x=px, y=py, ch=ch,
                                                  res=mr.get('measure_result'),
                                                  svd=mr.get('svd_deg'), t=c.virtual_time))
                if mr['measure_result'] == 'near':
                    cr2, _ = c.clear(px, py, ch)
                    if cr2['clear_result'] == 'success':
                        self._mark(ch, t_kill, 'walk_near')
                        return
                elif mr['measure_result'] == 'direction':
                    u = (math.cos(math.radians(mr['svd_deg'])), math.sin(math.radians(mr['svd_deg'])))
                    ex, ey = px, py
                    k = 0
        self.trace['sources'][ch] = dict(first_heard=self.first_heard.get(ch, c.virtual_time),
                                         cleared_t=None, method='FAILED')

    def _mark(self, ch, t_kill, method):
        c = self.c
        self.cleared.add(ch)
        self.trace['sources'][ch] = dict(first_heard=self.first_heard.get(ch, c.virtual_time),
                                         cleared_t=c.virtual_time,
                                         kill_start=t_kill, active_t=c.virtual_time-t_kill,
                                         method=method)

    def _summary(self):
        c = self.c
        cleared_n = len([ch for ch, s in self.trace['sources'].items() if s['cleared_t'] is not None])
        return dict(cleared=cleared_n,
                    total_time=c.virtual_time,
                    move_t=c.move_t, switch_t=c.switch_t, measure_t=c.measure_t,
                    clear_ok_t=c.clear_ok_t, clear_fail_t=c.clear_fail_t,
                    n_actions=len(self.trace['actions']),
                    sources=self.trace['sources'])

if __name__ == '__main__':
    import argparse
    program_start = time.perf_counter()
    parser = argparse.ArgumentParser(description='机器狗问题3策略')
    parser.add_argument('--robot-id', default=None,
                        help='参赛队号；提供时连接官方HTTP模拟器，省略时使用本地会话')
    parser.add_argument('--base-url', default=BASE_URL, help=f'模拟器地址(默认: {BASE_URL})')
    parser.add_argument('--log-dir', default=None,
                        help='完整行为日志目录(本地模式默认写入系统临时目录)')
    args = parser.parse_args()
    local_backend = None
    if args.robot_id:
        client = SimClient(base_url=args.base_url, robot_id=args.robot_id)
    else:
        from Simulator import FileLocalSimulator, session_path
        try:
            local_backend = FileLocalSimulator(session_path(3))
        except FileNotFoundError:
            parser.error('未找到Q3本地会话，请先运行 Simulator.py')
        if local_backend.entered or local_backend.finished:
            parser.error('Q3本地会话已经使用，请重新运行 Simulator.py 后再测试')
        args.robot_id = local_backend.robot_id
        client = SimClient(robot_id=args.robot_id, backend=local_backend)
    strat = StrategyQ3(client)
    s = strat.run()
    if local_backend is not None:
        truth = local_backend.truth_summary()
        s['local_simulator'] = dict(source_count=truth['source_count'],
                                    directional_count=truth['directional_count'],
                                    all_cleared=all(item['cleared'] for item in truth['sources']))
    print(json.dumps(s, ensure_ascii=False, indent=2))
    elapsed = time.perf_counter() - program_start
    print(f'整段程序运行时间: {elapsed:.3f} 秒')
    # ---- 自动保存完整日志(含动作轨迹), 供绘图程序使用; 保存失败不影响测试 ----
    try:
        s['actions'] = strat.trace['actions']
        s['program_run_time_s'] = elapsed
        # 本地模拟写到会话的仓库外目录；官方 HTTP 模式保持原 logs_q3 默认值。
        default_dir = (local_backend.log_dir if local_backend is not None else
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_q3'))
        out_dir = args.log_dir or default_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'Q3_{args.robot_id}_{time.strftime("%Y%m%d_%H%M%S")}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        print(f'完整日志已保存: {out_path}')
    except Exception as e:
        print(f'完整日志保存失败(不影响测试): {e}')
