# -*- coding: utf-8 -*-
"""
机器狗程序(问题4): 全向+定向混合干扰源的自动搜索定位与清除
与官方模拟器通信(HTTP+JSON)。

策略概要(在问题3基础上的扩展): 
  定向干扰源仅在定向方向±90°覆盖角内可被检测, 从背面完全无信号。
  第一阶段 双层覆盖扫描: 内圈9点(同问题3)+外圈12点(半径2050米、30°间隔, 
    布置于目标区域外)。经数值验证: 任意位置、任意朝向的干扰源必被至少一个
    探测点从覆盖角内、1000米以内检测到(完备性保证)。
  第二阶段 粗定位: 与问题3相同(示向线扇形半平面交)。补测失败不排除近距离
    定向源(可能背面), 一律按示向线远方点+大范围兜底处理。
  第三阶段 精细清除: 先直接尝试清除(清除与覆盖角无关); 失败后若检测无信号
    (可能从背面接近), 绕估计点半径40米环形6方向探测获得示向线, 再沿示向线
    步进清除; "near"直接清除。
"""
import math
import time
import os
import json
from Q3_RobotDog import (SimClient, StrategyQ3, localization_polygon, poly_centroid,
                          poly_diameter, best_cross_sin, GOOD_SIN, PROBE_T, PROBE_L)

BASE_URL = 'http://127.0.0.1:2026'

SURVEY_RING_RADIUS = 40.0  # 环形探测基础半径(米)

def optimize_open_route(points, start=(0.0, 0.0)):
    """为一组二维点生成较短的开放路径(最近邻 + 2-opt微调)。

    输入可以是点列表或以点为键的字典; 返回点列表, 不闭合回起点。
    """
    pts = list(points)
    if not pts:
        return []
    # 先用最近邻得到稳定初始解
    remain = pts[:]
    route = []
    cur = start
    while remain:
        nxt = min(remain, key=lambda p: math.hypot(p[0]-cur[0], p[1]-cur[1]))
        route.append(nxt)
        remain.remove(nxt)
        cur = nxt
    # 开放路径2-opt: 只接受严格改进, 规模很小时开销可忽略
    improved = True
    while improved:
        improved = False
        for i in range(len(route)-1):
            a = start if i == 0 else route[i-1]
            b = route[i]
            for j in range(i+1, len(route)):
                c = route[j]
                d = route[j+1] if j+1 < len(route) else None
                old = math.hypot(a[0]-b[0], a[1]-b[1])
                if d is not None:
                    old += math.hypot(c[0]-d[0], c[1]-d[1])
                new = math.hypot(a[0]-c[0], a[1]-c[1])
                if d is not None:
                    new += math.hypot(b[0]-d[0], b[1]-d[1])
                if new + 1e-9 < old:
                    route[i:j+1] = reversed(route[i:j+1])
                    improved = True
                    break
            if improved:
                break
    return route


def minimum_enclosing_circle(points):
    """返回覆盖所有点的最小圆 (中心, 半径)。"""
    pts = list(points)
    if not pts:
        return (0.0, 0.0), 0.0
    if len(pts) == 1:
        return pts[0], 0.0

    def circle3(a, b, c):
        ax, ay = a; bx, by = b; cx, cy = c
        d = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
        if abs(d) < 1e-12:
            return None
        aa, bb, cc = ax*ax+ay*ay, bx*bx+by*by, cx*cx+cy*cy
        ux = (aa*(by-cy) + bb*(cy-ay) + cc*(ay-by)) / d
        uy = (aa*(cx-bx) + bb*(ax-cx) + cc*(bx-ax)) / d
        o = (ux, uy)
        return o, math.hypot(ux-ax, uy-ay)

    best = None
    # 单点和两点定义的候选圆
    for p in pts:
        cand = (p, 0.0)
        if all(math.hypot(q[0]-p[0], q[1]-p[1]) <= 1e-9 for q in pts):
            best = cand if best is None or cand[1] < best[1] else best
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            o = ((pts[i][0]+pts[j][0])/2, (pts[i][1]+pts[j][1])/2)
            r = math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])/2
            if all(math.hypot(q[0]-o[0], q[1]-o[1]) <= r+1e-9 for q in pts):
                if best is None or r < best[1]:
                    best = (o, r)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            for k in range(j+1, len(pts)):
                cand = circle3(pts[i], pts[j], pts[k])
                if cand and all(math.hypot(q[0]-cand[0][0], q[1]-cand[0][1]) <= cand[1]+1e-9 for q in pts):
                    if best is None or cand[1] < best[1]:
                        best = cand
    return best if best is not None else (poly_centroid(pts), poly_diameter(pts)/2)

def scan_grid_q4():
    """28点三环覆盖网格, 兼顾覆盖完备性与移动距离。"""
    pts = []
    for n, radius, offset_deg in ((5, 607.17512794, 53.75693290),
                                  (11, 1374.62274481, 31.08579950),
                                  (12, 1984.90077183, 5.58974051)):
        for k in range(n):
            a = math.radians(offset_deg + 360.0*k/n)
            pts.append((radius*math.cos(a), radius*math.sin(a)))
    return pts

class StrategyQ4(StrategyQ3):
    def __init__(self, client):
        super().__init__(client)

    def _scan_sufficient(self, bearings):
        if len(bearings) < 2:
            return False
        poly = localization_polygon(bearings)
        if not poly:
            return False
        _, radius = minimum_enclosing_circle(poly)
        return radius <= 19.0 or len(bearings) >= 4

    def run(self):
        c = self.c
        c.enter()
        self.cleared = set()
        self.bearings = {}
        self.first_heard = {}
        # ---------- 第一阶段: 双层覆盖扫描 ----------
        for idx, (gx, gy) in enumerate(optimize_open_route(scan_grid_q4(), c.pos)):
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
                    if cr['clear_result'] == 'success':
                        self.cleared.add(ch)
                        self.trace['sources'][ch] = dict(first_heard=c.virtual_time,
                                                         cleared_t=c.virtual_time, method='near')
                        self.first_heard[ch] = c.virtual_time
                elif r['measure_result'] == 'direction':
                    self.bearings.setdefault(ch, []).append((gx, gy, r['svd_deg']))
                    if ch not in self.first_heard:
                        self.first_heard[ch] = c.virtual_time
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
            if self._probe_and_kill_q4(ch):
                cur = c.pos
                continue
            est, bound = self._far_estimate(ch)
            cur = est
            self._kill_q4(ch, est, bound)
        # ---------- 第三阶段: 良好交会源最近邻清除 ----------
        estimates = {}
        for ch in good:
            poly = localization_polygon(self.bearings[ch])
            if poly:
                estimates[ch] = minimum_enclosing_circle(poly)
        # 以定位中心规划开放路径; 同坐标的不同频道也要全部保留
        point_to_ch = {}
        for ch, (point, _) in estimates.items():
            point_to_ch.setdefault(point, []).append(ch)
        route = optimize_open_route(list(point_to_ch.keys()), c.pos)
        order = []
        for point in route:
            order.extend(point_to_ch[point])
        for ch in order:
            if ch in self.cleared:
                continue
            est, bound = estimates[ch]
            self._kill_q4(ch, est, bound)
        # ---------- 第四阶段: 完备性核查与退出 ----------
        for ch in range(1, 21):
            if ch in self.cleared:
                continue
            if ch in self.bearings:
                est, bound = estimates.get(ch, self._far_estimate(ch))
                self._kill_q4(ch, est, bound*2 + 200)
        c.exit()
        return self._summary()

    def _probe_and_kill_q4(self, ch):
        """补测并就地清除(无信号不代表远方, 兜底走_kill_q4)"""
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
                    est, bound = minimum_enclosing_circle(poly)
                break
        if est is None:
            return self._fallback_clear_bearing_sector(ch)
        self._kill_q4(ch, est, bound)
        return ch in self.cleared

    def _fallback_clear_bearing_sector(self, ch):
        """单示向线时用两条蛇形轨迹覆盖误差扇区, 保证定向源可清除。"""
        c = self.c
        x0, y0, th0 = self.bearings[ch][0]
        u = (math.cos(math.radians(th0)), math.sin(math.radians(th0)))
        n = (-u[1], u[0])
        t_kill = c.virtual_time
        longitudinal = [30.0*k for k in range(51)]
        for row, lateral in enumerate((-13.2, 13.2)):
            ts = longitudinal if row == 0 else list(reversed(longitudinal))
            for t in ts:
                px, py = x0 + t*u[0] + lateral*n[0], y0 + t*u[1] + lateral*n[1]
                cr, _ = c.clear(px, py, ch)
                self.trace['actions'].append(dict(kind='sector_clear', x=px, y=py, ch=ch,
                                                  res=cr.get('clear_result'), t=c.virtual_time))
                if cr['clear_result'] == 'success':
                    self._mark(ch, t_kill, 'sector_cover')
                    return True
        self.trace['sources'][ch] = dict(first_heard=self.first_heard.get(ch, c.virtual_time),
                                         cleared_t=None, method='FAILED')
        return False

    def _kill_q4(self, ch, est, bound):
        """问题4清除: 直接清除失败后, 若检测无信号则环形探测找示向线"""
        c = self.c
        t_kill = c.virtual_time
        ex, ey = est
        r, _ = c.clear(ex, ey, ch)
        self.trace['actions'].append(dict(kind='clear', x=ex, y=ey, ch=ch,
                                          res=r.get('clear_result'), t=c.virtual_time))
        if r['clear_result'] == 'success':
            self._mark(ch, t_kill, 'direct')
            return
        # 可能从背面接近: 先测; 无信号则环形探测
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
        # 环形探测: 绕估计点按基础半径逐圈、6方向找信号
        for k in range(1, 5):
            rr = SURVEY_RING_RADIUS*k
            for ang in range(0, 360, 60):
                px, py = ex + rr*math.cos(math.radians(ang)), ey + rr*math.sin(math.radians(ang))
                mr, _, _ = c.measure(px, py, ch)
                self.trace['actions'].append(dict(kind='survey', x=px, y=py, ch=ch,
                                                  res=mr.get('measure_result'),
                                                  svd=mr.get('svd_deg'), t=c.virtual_time))
                if mr['measure_result'] == 'near':
                    cr, _ = c.clear(px, py, ch)
                    if cr['clear_result'] == 'success':
                        self._mark(ch, t_kill, 'survey_near')
                        return
                elif mr['measure_result'] == 'direction':
                    if self._fine_fix(ch, px, py, mr['svd_deg'], t_kill):
                        return
                    self._walk_kill(ch, (px, py), mr['svd_deg'], max(bound, 100.0), t_kill)
                    if ch in self.cleared:
                        return
        # 十字+螺旋清除兜底
        self._walk_kill(ch, (ex, ey), None, bound, t_kill)

if __name__ == '__main__':
    import json
    import argparse
    program_start = time.perf_counter()
    parser = argparse.ArgumentParser(description='机器狗问题4策略')
    parser.add_argument('--robot-id', required=True, help='参赛队号')
    parser.add_argument('--base-url', default=BASE_URL, help=f'模拟器地址(默认: {BASE_URL})')
    args = parser.parse_args()
    client = SimClient(base_url=args.base_url, robot_id=args.robot_id)
    strat = StrategyQ4(client)
    s = strat.run()
    print(json.dumps(s, ensure_ascii=False, indent=2))
    elapsed = time.perf_counter() - program_start
    print(f'整段程序运行时间: {elapsed:.3f} 秒')
    # ---- 自动保存完整日志(含动作轨迹), 供绘图程序使用; 保存失败不影响测试 ----
    try:
        import os
        s['actions'] = strat.trace['actions']
        s['program_run_time_s'] = elapsed
        # 问题四日志独立保存到图表程序默认读取的 logs_q4 目录。
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_q4')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'Q4_{args.robot_id}_{time.strftime("%Y%m%d_%H%M%S")}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        print(f'完整日志已保存: {out_path}')
    except Exception as e:
        print(f'完整日志保存失败(不影响测试): {e}')
