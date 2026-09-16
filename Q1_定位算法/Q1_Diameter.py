# -*- coding: utf-8 -*-
"""
问题一: 交会定位法求定位区域直径
  1) 扇形(±1°)交会 -> 半平面交求凸多边形区域
  2) 旋转卡壳求直径(顶点穷举校验)
  3) 最小包围圆(Welzl)
  4) 直径圆覆盖判定 + Jung定理分析
  5) 随机实验统计 r*/D 分布
  6) 自动随机生成三组合法探测场景（不依赖外部场景文件）
先算后画。
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings, json, itertools
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['STHeiti', 'SimHei', 'Heiti TC',
    'Arial Unicode MS', 'Hiragino Sans GB', 'PingFang SC',
    'Microsoft YaHei', 'Songti SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FIG_DIR = os.path.join(BASE_DIR, '图片')
OUT_DIR = os.path.join(BASE_DIR, '结果')
TARGET_RADIUS = 1800.0  # 题设目标区域半径（米）
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def save_fig(fig, name_cn):
    # 图片存到脚本同目录的"图片"文件夹
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name_cn)
    fig.savefig(path)
    plt.close(fig)
    print(f'已生成图片: {path}')

def save_csv(df, name_cn):
    df.to_csv(os.path.join(OUT_DIR, name_cn), index=False, encoding='utf-8-sig')

def uvec(deg):
    """方位角(度, 0=东, 逆时针) -> 单位向量"""
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])

# ---------- 计算 ----------
def wedge_halfplanes(S, theta_deg):
    """检测点S处示向度theta(度)构成的±1°扇形 -> 两个半平面 [a,b,c] (a x + b y <= c)"""
    t = np.deg2rad(theta_deg)
    d1, d2 = t - np.deg2rad(1.0), t + np.deg2rad(1.0)
    u1 = np.array([np.cos(d1), np.sin(d1)])
    u2 = np.array([np.cos(d2), np.sin(d2)])
    S = np.asarray(S, float)
    # 扇形 = {P : cross(u1, P-S)<=0 且 cross(u2, P-S)>=0}
    # cross(u,P-S) = uy*Px - ux*Py - (uy*Sx - ux*Sy)
    hp = []
    a1, b1, c1 = u1[1], -u1[0], u1[1]*S[0] - u1[0]*S[1]          # a1 x + b1 y <= c1
    a2, b2, c2 = -u2[1], u2[0], -u2[1]*S[0] + u2[0]*S[1]          # a2 x + b2 y <= c2
    hp.append((a1, b1, c1))
    hp.append((a2, b2, c2))
    return hp

def clip_polygon(poly, hp, eps=1e-9):
    """Sutherland-Hodgman: 用半平面 a x + b y <= c 裁剪多边形(CCW顶点列表)"""
    a, b, c = hp
    if len(poly) == 0:
        return []
    out = []
    n = len(poly)
    for i in range(n):
        P = poly[i]
        Q = poly[(i+1) % n]
        fP = a*P[0] + b*P[1] - c
        fQ = a*Q[0] + b*Q[1] - c
        inP = fP <= eps
        inQ = fQ <= eps
        if inP:
            out.append(P)
        if inP != inQ:
            # 交点
            t = fP / (fP - fQ)
            out.append([P[0] + t*(Q[0]-P[0]), P[1] + t*(Q[1]-P[1])])
    # 去重(相邻重复)
    cleaned = []
    for p in out:
        if not cleaned or np.hypot(p[0]-cleaned[-1][0], p[1]-cleaned[-1][1]) > 1e-6:
            cleaned.append(p)
    if len(cleaned) > 1 and np.hypot(cleaned[0][0]-cleaned[-1][0], cleaned[0][1]-cleaned[-1][1]) <= 1e-6:
        cleaned.pop()
    return cleaned

def localization_polygon(points, bearings, cap=6000.0):
    """k个检测点+示向度 -> 交会定位凸多边形(顶点CCW列表)
    返回 (vertices, unbounded_flag)"""
    poly = [[-cap, -cap], [cap, -cap], [cap, cap], [-cap, cap]]  # 初始大方形
    unbounded = False
    for S, th in zip(points, bearings):
        for hp in wedge_halfplanes(S, th):
            poly = clip_polygon(poly, hp)
            if len(poly) < 3:
                break
        if len(poly) < 3:
            break
    # 检查是否触碰初始边界(区域无界)
    if poly:
        for p in poly:
            if abs(abs(p[0]) - cap) < 1e-6 or abs(abs(p[1]) - cap) < 1e-6:
                unbounded = True
    return poly, unbounded

def polygon_area(verts):
    A = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]; x2, y2 = verts[(i+1) % n]
        A += x1*y2 - x2*y1
    return abs(A)/2.0

def polygon_centroid(verts):
    A = 0.0; cx = 0.0; cy = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]; x2, y2 = verts[(i+1) % n]
        f = x1*y2 - x2*y1
        A += f; cx += (x1+x2)*f; cy += (y1+y2)*f
    A /= 2.0
    return [cx/(6*A), cy/(6*A)] if A > 1e-12 else [np.mean([p[0] for p in verts]),
                                                    np.mean([p[1] for p in verts])]

def brute_diameter(verts):
    """顶点对穷举直径 O(n^2)"""
    n = len(verts); best = 0.0; pair = None
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(verts[i][0]-verts[j][0], verts[i][1]-verts[j][1])
            if d > best:
                best = d; pair = (i, j)
    return best, pair

def rotating_calipers_diameter(verts):
    """旋转卡壳求凸多边形直径 O(n), verts为CCW凸多边形"""
    n = len(verts)
    if n < 2:
        return 0.0, (0, 0)
    if n == 2:
        return np.hypot(verts[0][0]-verts[1][0], verts[0][1]-verts[1][1]), (0, 1)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    j = 1
    while cross(verts[n-1], verts[0], verts[(j+1) % n]) > cross(verts[n-1], verts[0], verts[j]):
        j = (j+1) % n
    best = 0.0; pair = (0, 0)
    for i in range(n):
        # 顺时针推进对跖点
        while cross(verts[i], verts[(i+1) % n], verts[(j+1) % n]) > cross(verts[i], verts[(i+1) % n], verts[j]):
            j = (j+1) % n
        for (a, b) in [(i, j), ((i+1) % n, j), (i, (j+1) % n), ((i+1) % n, (j+1) % n)]:
            d = np.hypot(verts[a][0]-verts[b][0], verts[a][1]-verts[b][1])
            if d > best:
                best = d; pair = (a, b)
    return best, pair

def circle_from_2(p, q):
    """过两点最小圆(以中点为圆心)"""
    c = [(p[0]+q[0])/2, (p[1]+q[1])/2]
    r = np.hypot(p[0]-q[0], p[1]-q[1])/2
    return c, r

def circle_from_3(p, q, s):
    """过三点最小圆: 外接圆; 钝角时取最长边直径圆"""
    # 外接圆
    d = 2*(p[0]*(q[1]-s[1]) + q[0]*(s[1]-p[1]) + s[0]*(p[1]-q[1]))
    if abs(d) < 1e-12:
        return circle_from_2(p, q)
    px, py = p; qx, qy = q; sx, sy = s
    ux = ((px*px+py*py)*(qy-sy) + (qx*qx+qy*qy)*(sy-py) + (sx*sx+sy*sy)*(py-qy))/d
    uy = ((px*px+py*py)*(sx-qx) + (qx*qx+qy*qy)*(px-sx) + (sx*sx+sy*sy)*(qx-px))/d
    c = [ux, uy]
    r = np.hypot(px-ux, py-uy)
    # 钝角情形: 某边2点圆能包住第三点且更小, 就换它
    for (a, b), other in [((p, q), s), ((q, s), p), ((s, p), q)]:
        cc, rr = circle_from_2(a, b)
        if (other[0]-cc[0])**2 + (other[1]-cc[1])**2 <= rr*rr + 1e-9 and rr < r:
            c, r = cc, rr
    return c, r

def smallest_enclosing_circle(verts):
    """最小包围圆: 随机增量(Welzl)实现, 另以穷举校验"""
    pts = [list(p) for p in verts]
    if len(pts) == 0:
        return [0, 0], 0
    if len(pts) == 1:
        return pts[0], 0
    if len(pts) == 2:
        return circle_from_2(pts[0], pts[1])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pts))
    pts = [pts[i] for i in perm]

    def welzl(P, R):
        if len(P) == 0 or len(R) == 3:
            if len(R) == 0:
                return [0, 0], 0
            if len(R) == 1:
                return R[0], 0
            if len(R) == 2:
                return circle_from_2(R[0], R[1])
            return circle_from_3(R[0], R[1], R[2])
        p = P.pop()
        c, r = welzl(P, R)
        if np.hypot(p[0]-c[0], p[1]-c[1]) <= r + 1e-9:
            P.append(p)
            return c, r
        R.append(p)
        c, r = welzl(P, R)
        R.pop()
        P.append(p)
        return c, r

    c, r = welzl(pts, [])
    # 穷举校验: 所有2点圆与3点圆取最小
    c2, r2 = None, np.inf
    n = len(verts)
    for (a, b) in itertools.combinations(range(n), 2):
        cc, rr = circle_from_2(verts[a], verts[b])
        if all(np.hypot(v[0]-cc[0], v[1]-cc[1]) <= rr + 1e-6 for v in verts) and rr < r2:
            c2, r2 = cc, rr
    for (a, b, d) in itertools.combinations(range(n), 3):
        cc, rr = circle_from_3(verts[a], verts[b], verts[d])
        if all(np.hypot(v[0]-cc[0], v[1]-cc[1]) <= rr + 1e-6 for v in verts) and rr < r2:
            c2, r2 = cc, rr
    assert abs(r - r2) < 1e-6, f"Welzl与穷举不一致: {r} vs {r2}"
    return c2, r2

def coverage_by_diameter_circle(verts, D, pair):
    """直径圆(以直径对中点为圆心、D/2为半径)是否覆盖全部顶点"""
    p, q = verts[pair[0]], verts[pair[1]]
    m = [(p[0]+q[0])/2, (p[1]+q[1])/2]
    r = D/2
    return all(np.hypot(v[0]-m[0], v[1]-m[1]) <= r + 1e-9 for v in verts), m, r

def stats_print(name, arr):
    arr = np.asarray(arr, float)
    print(f"[{name}] n={len(arr)} min={arr.min():.4f} max={arr.max():.4f} "
          f"mean={arr.mean():.4f} std={arr.std(ddof=1):.4f} "
          f"CV={arr.std(ddof=1)/arr.mean():.4f} amplitude={arr.max()-arr.min():.4f}")

# ---------- 随机场景(含真值, 验证用) ----------
# 默认使用系统熵，因此每次运行都会得到不同的探测点；设置环境变量
# Q1_RANDOM_SEED 可以复现实验，例如 PowerShell 中：$env:Q1_RANDOM_SEED=2026。
_seed_text = os.environ.get('Q1_RANDOM_SEED', '').strip()
try:
    RANDOM_SEED = int(_seed_text) if _seed_text else None
except ValueError:
    raise ValueError('环境变量 Q1_RANDOM_SEED 必须是整数')
rng = np.random.default_rng(RANDOM_SEED)


def make_scenario(G, S_list, seed=None, rng=None):
    """给定真值 G 与检测点，生成带 ±1° 误差的示向度。

    ``rng`` 用于主程序的连续随机生成；保留 ``seed`` 参数以兼容旧代码。
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    bearings = []
    for S in S_list:
        true_deg = np.rad2deg(np.arctan2(G[1]-S[1], G[0]-S[0])) % 360
        err = rng.uniform(-1.0, 1.0)
        bearings.append((true_deg + err) % 360)
    return bearings


def _circular_separations(angles_deg):
    """返回一组角度的相邻圆周间隔（度）。"""
    angles = np.sort(np.asarray(angles_deg, dtype=float) % 360.0)
    if len(angles) < 2:
        return np.array([], dtype=float)
    return np.diff(np.r_[angles, angles[0] + 360.0])


def _random_detector_angles(n_points, rng):
    """生成分布在真值周围的探测方向，避免近似平行导致区域无界/病态。"""
    # 两条示向线取 55°--125° 的交会角，避开近似平行（0°或180°）；
    # 3、4 个点则采用等角基准加小扰动，让探测点分布在真值四周。
    base = rng.uniform(0.0, 360.0)
    if n_points == 2:
        return np.array([base, base + rng.uniform(55.0, 125.0)]) % 360.0
    jitter = rng.uniform(-18.0, 18.0, n_points)
    angles = (base + np.arange(n_points) * 360.0 / n_points + jitter) % 360.0
    # 极小概率扰动使两个方向过近时重新采样。
    while np.min(_circular_separations(angles)) < 35.0:
        base = rng.uniform(0.0, 360.0)
        jitter = rng.uniform(-18.0, 18.0, n_points)
        angles = (base + np.arange(n_points) * 360.0 / n_points + jitter) % 360.0
    return angles


def generate_random_scenarios(rng=None, max_attempts=2000):
    """随机生成三个满足定位条件的场景 A/B/C。

    场景 A、B、C 分别含 2、3、4 个探测点。真值和探测点位于题设半径
    1800 m 的目标圆域内，探测点到真值距离在 600--1200 m 之间，方位
    分布充分分散；每个示向度加入 ±1° 测量误差。
    函数只接受通过几何检查的结果：交会区域至少有 3 个顶点、有界、面积
    为正，且真值位于区域内。这样下游直径、包围圆和绘图不会因随机退化而失败。
    """
    if rng is None:
        rng = np.random.default_rng()

    def point_in_polygon(verts, point):
        inside = False
        x, y = point
        j = len(verts) - 1
        for i in range(len(verts)):
            xi, yi = verts[i]
            xj, yj = verts[j]
            if ((yi > y) != (yj > y)) and (
                    x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside

    scenarios = []
    for tag, n_points in [('A', 2), ('B', 3), ('C', 4)]:
        accepted = None
        for _ in range(max_attempts):
            # 在半径 900 m 的圆盘内均匀取真值，确保真值远离边界，
            # 再将探测点限制在题设目标圆域内。
            source_r = 900.0 * np.sqrt(rng.uniform())
            source_theta = rng.uniform(0.0, 2.0 * np.pi)
            G = source_r * np.array([np.cos(source_theta), np.sin(source_theta)])
            directions = _random_detector_angles(n_points, rng)
            distances = rng.uniform(600.0, 1200.0, size=n_points)
            S_list = [G - distance * uvec(direction)
                      for direction, distance in zip(directions, distances)]
            if any(np.hypot(S[0], S[1]) > TARGET_RADIUS for S in S_list):
                continue
            B_list = make_scenario(G, S_list, rng=rng)
            poly, unbounded = localization_polygon(S_list, B_list)
            if unbounded or len(poly) < 3 or polygon_area(poly) <= 1e-6:
                continue
            if not point_in_polygon(poly, G):
                continue
            # 两点交会应为四边形；若随机误差使某条边冗余，重新采样。
            if tag == 'A' and len(poly) != 4:
                continue
            accepted = dict(tag=tag, G=np.asarray(G, dtype=float),
                            S=[np.asarray(S, dtype=float) for S in S_list],
                            B=[float(b) for b in B_list])
            break
        if accepted is None:
            raise RuntimeError(f'无法在 {max_attempts} 次尝试内生成场景{tag}，请调整随机范围')
        scenarios.append(accepted)
    return scenarios


# 场景 A/B/C 每次由程序自动随机生成，不再依赖任何外部场景文件。
_scenarios = generate_random_scenarios(rng)
print(f'已生成随机场景 A/B/C（seed={RANDOM_SEED if RANDOM_SEED is not None else "系统熵"}）')
G_A, S_A, B_A = _scenarios[0]['G'], _scenarios[0]['S'], _scenarios[0]['B']
G_B, S_B, B_B = _scenarios[1]['G'], _scenarios[1]['S'], _scenarios[1]['B']
G_C, S_C, B_C = _scenarios[2]['G'], _scenarios[2]['S'], _scenarios[2]['B']

results = []
scenario_rows = []
for tag, G, Ss, Bs in [('A', G_A, S_A, B_A), ('B', G_B, S_B, B_B), ('C', G_C, S_C, B_C)]:
    poly, unb = localization_polygon(Ss, Bs)
    D_rc, pair_rc = rotating_calipers_diameter(poly)
    D_bf, pair_bf = brute_diameter(poly)
    assert abs(D_rc - D_bf) < 1e-6, f"{tag} 旋转卡壳与穷举不一致"
    c, rstar = smallest_enclosing_circle(poly)
    cov, m, rD = coverage_by_diameter_circle(poly, D_rc, pair_rc)
    # 用射线法验证G在多边形内
    def pip(verts, P):
        n = len(verts); inside = False
        x, y = P
        j = n-1
        for i in range(n):
            xi, yi = verts[i]; xj, yj = verts[j]
            if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-30) + xi):
                inside = not inside
            j = i
        return inside
    G_in = pip(poly, G)
    results.append(dict(tag=tag, n_pts=len(Ss), n_vert=len(poly), area=polygon_area(poly),
                        D=D_rc, rstar=rstar, ratio=rstar/D_rc, covered=cov, G_in_poly=G_in,
                        unbounded=unb, cx=c[0], cy=c[1], Dcx=m[0], Dcy=m[1]))
    print(f"场景{tag}: 检测点{len(Ss)} 顶点{len(poly)} 面积{polygon_area(poly):.1f} "
          f"直径D={D_rc:.2f} 最小包围圆r*={rstar:.2f} r*/D={rstar/D_rc:.4f} "
          f"直径圆覆盖={cov} 真值在多边形内={G_in}")
    for i, (S, th) in enumerate(zip(Ss, Bs)):
        scenario_rows.append(dict(场景=tag, 检测点编号=i+1,
                                 x=round(float(S[0]), 2), y=round(float(S[1]), 2),
                                 示向度=round(float(th), 2)))
    vert_str = ';'.join(f'{v[0]:.2f},{v[1]:.2f}' for v in poly)
    scenario_rows.append(dict(场景=tag, 检测点编号='区域', x='', y='', 示向度='',
                              顶点=vert_str, 直径=round(D_rc, 2), 最小包围圆半径=round(rstar, 2),
                              直径圆覆盖=('是' if cov else '否'), 真值在多边形内=('是' if G_in else '否')))

# 等边三角形反例(直径圆不能覆盖的典型形状)
tri = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3)/2])]
D_tri, pair_tri = brute_diameter(tri)
c_tri, r_tri = smallest_enclosing_circle(tri)
cov_tri, m_tri, _ = coverage_by_diameter_circle(tri, D_tri, pair_tri)
print(f"等边三角形: D={D_tri:.6f} r*={r_tri:.6f} r*/D={r_tri/D_tri:.6f} (理论1/√3={1/np.sqrt(3):.6f}) 直径圆覆盖={cov_tri}")

# 场景D: 直径圆恰好覆盖的例子(矩形2x1, 直径对为对角, 其余顶点在圆内)
rect = [np.array([1.0, 0.5]), np.array([-1.0, 0.5]), np.array([-1.0, -0.5]), np.array([1.0, -0.5])]
D_rect, pair_rect = brute_diameter(rect)
cov_rect, m_rect, _ = coverage_by_diameter_circle(rect, D_rect, pair_rect)
c_rect, r_rect = smallest_enclosing_circle(rect)
print(f"矩形2x1: D={D_rect:.4f} r*={r_rect:.4f} 直径圆覆盖={cov_rect}")

# 随机实验: k个检测点随机布设, 统计r*/D与覆盖比例。
# 与上面的三组示例场景使用独立随机流，保证示例坐标随机时箱线图仍可复现。
experiment_rng = np.random.default_rng(2026)
exp_rows = []
for k in range(2, 7):
    rs = []
    for trial in range(1000):
        G = np.array([experiment_rng.uniform(-1000, 1000), experiment_rng.uniform(-1000, 1000)])
        Ss = [np.array([experiment_rng.uniform(-1500, 1500),
                        experiment_rng.uniform(-1500, 1500)]) for _ in range(k)]
        # 剔除与G重合或过近的点
        Ss = [S for S in Ss if np.hypot(*(S-G)) > 50]
        if len(Ss) < 2:
            continue
        Bs = make_scenario(G, Ss, seed=trial+k*1000)
        poly, unb = localization_polygon(Ss, Bs)
        if unb or len(poly) < 3:
            continue
        D, pair = brute_diameter(poly)
        c, rstar = smallest_enclosing_circle(poly)
        rs.append(rstar/D)
    rs = np.array(rs)
    stats_print(f"随机实验k={k} r*/D", rs)
    for v in rs:
        exp_rows.append(dict(k=k, ratio=v))
exp_df = pd.DataFrame(exp_rows)
save_csv(exp_df, '问题一_随机实验比值.csv')

# 汇总输出
res_df = pd.DataFrame(results)
save_csv(res_df, '问题一_结果汇总.csv')
# 预设场景.csv 由本次随机场景实时生成（程序不读取任何外部场景文件）。
sc_df = pd.DataFrame(scenario_rows)
save_csv(sc_df, '问题一_场景明细.csv')
prep_rows = []
for tag, G, Ss, Bs in [('A', G_A, S_A, B_A), ('B', G_B, S_B, B_B), ('C', G_C, S_C, B_C)]:
    poly, _ = localization_polygon(Ss, Bs)
    D, pair = brute_diameter(poly)
    c, rstar = smallest_enclosing_circle(poly)
    for i, (S, th) in enumerate(zip(Ss, Bs)):
        prep_rows.append(dict(场景=tag, 检测点序号=i+1, 检测点x=round(S[0],2), 检测点y=round(S[1],2),
                              示向度=round(th,4), 真值x=round(G[0],2), 真值y=round(G[1],2),
                              多边形顶点数=len(poly), 定位区域直径=round(D,3), 最小包围圆半径=round(rstar,3)))
prep_df = pd.DataFrame(prep_rows)
prep_df.to_csv(os.path.join(BASE_DIR, '预设场景.csv'), index=False, encoding='utf-8-sig')
print(f'预设场景.csv 已根据本次随机场景生成: {os.path.join(BASE_DIR, "预设场景.csv")}')

# 场景统计
stats_print('场景直径', [r['D'] for r in results])
stats_print('场景r*/D', [r['ratio'] for r in results])

# ---------- 画图 ----------
def draw_wedge(ax, S, theta, length, color, lw=1.2, ls='-', label=None):
    for off in [-1, 1]:
        v = uvec(theta + off)
        ax.plot([S[0], S[0]+v[0]*length], [S[1], S[1]+v[1]*length],
                color=color, lw=lw, ls=ls, label=(label if off == -1 else None))

def fill_polygon(ax, poly, color='tab:orange', alpha=0.35, label='定位区域'):
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.fill(xs, ys, color=color, alpha=alpha, label=label)

# ---- 图1-1 两检测点交会定位四边形 ----
fig, ax = plt.subplots(figsize=(7, 6))
for S, th in zip(S_A, B_A):
    draw_wedge(ax, S, th, 1600, color='tab:blue', lw=1.0, ls='--')
ax.plot([S_A[0][0], S_A[1][0]], [S_A[0][1], S_A[1][1]], color='0.7', lw=0.8, ls=':')
for S in S_A:
    ax.plot(S[0], S[1], 'ks', ms=8)
    ax.annotate('检测点S1' if (S == S_A[0]).all() else '检测点S2',
                (S[0], S[1]), textcoords='offset points', xytext=(8, -16), fontsize=11)
polyA, _ = localization_polygon(S_A, B_A)
fill_polygon(ax, polyA)
ax.plot(G_A[0], G_A[1], 'r*', ms=16, zorder=5)
ax.annotate('干扰源G', (G_A[0], G_A[1]), textcoords='offset points', xytext=(8, 8), fontsize=11)
D_A, pair_A = brute_diameter(polyA)
pA, qA = polyA[pair_A[0]], polyA[pair_A[1]]
ax.plot([pA[0], qA[0]], [pA[1], qA[1]], color='tab:red', lw=2.0, ls='-')
ax.annotate(f'直径D={D_A:.1f}米', ((pA[0]+qA[0])/2, (pA[1]+qA[1])/2),
            textcoords='offset points', xytext=(10, -20), fontsize=11, color='tab:red')
ax.set_xlabel('x(米)'); ax.set_ylabel('y(米)')
ax.legend(loc='lower left', fontsize=10)
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
ax.set_aspect('equal')
save_fig(fig, '图1-1_两检测点交会定位四边形.png')

# ---- 图1-2 三检测点多边形交会 ----
fig, ax = plt.subplots(figsize=(7, 6))
for S, th in zip(S_B, B_B):
    draw_wedge(ax, S, th, 1600, color='tab:blue', lw=1.0, ls='--')
for i, S in enumerate(S_B):
    ax.plot(S[0], S[1], 'ks', ms=8)
    ax.annotate(f'检测点S{i+1}', (S[0], S[1]), textcoords='offset points',
                xytext=(8, -16), fontsize=11)
polyB, _ = localization_polygon(S_B, B_B)
fill_polygon(ax, polyB)
ax.plot(G_B[0], G_B[1], 'r*', ms=16, zorder=5)
ax.annotate('干扰源G', (G_B[0], G_B[1]), textcoords='offset points', xytext=(8, 8), fontsize=11)
D_B, pair_B = brute_diameter(polyB)
pB, qB = polyB[pair_B[0]], polyB[pair_B[1]]
ax.plot([pB[0], qB[0]], [pB[1], qB[1]], color='tab:red', lw=2.0)
ax.annotate(f'直径D={D_B:.1f}米', ((pB[0]+qB[0])/2, (pB[1]+qB[1])/2),
            textcoords='offset points', xytext=(10, -20), fontsize=11, color='tab:red')
ax.set_xlabel('x(米)'); ax.set_ylabel('y(米)')
ax.legend(loc='lower left', fontsize=10)
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
ax.set_aspect('equal')
save_fig(fig, '图1-2_三检测点交会定位多边形.png')

# ---- 图1-3 直径圆覆盖检验(场景C, 展示未覆盖顶点)----
fig, ax = plt.subplots(figsize=(7, 6))
polyC, _ = localization_polygon(S_C, B_C)
D_C, pair_C = brute_diameter(polyC)
pC, qC = polyC[pair_C[0]], polyC[pair_C[1]]
mC = [(pC[0]+qC[0])/2, (pC[1]+qC[1])/2]
circle = plt.Circle(mC, D_C/2, fill=False, color='tab:red', lw=2.0, ls='-')
ax.add_patch(circle)
ax.plot([pC[0], qC[0]], [pC[1], qC[1]], color='tab:red', lw=2.0)
fill_polygon(ax, polyC)
covC, _, _ = coverage_by_diameter_circle(polyC, D_C, pair_C)
for idx, v in enumerate(polyC):
    dist = np.hypot(v[0]-mC[0], v[1]-mC[1])
    if dist > D_C/2 + 1e-9:
        ax.plot(v[0], v[1], 'o', color='tab:purple', ms=10, zorder=5)
        ax.annotate('圆外顶点', (v[0], v[1]), textcoords='offset points', xytext=(8, 8), fontsize=11)
for S in S_C:
    ax.plot(S[0], S[1], 'ks', ms=7)
ax.plot(mC[0], mC[1], 'r+', ms=12, zorder=5)
ax.annotate('直径圆圆心', (mC[0], mC[1]), textcoords='offset points', xytext=(8, -16), fontsize=11, color='tab:red')
ax.plot(G_C[0], G_C[1], 'r*', ms=16, zorder=5)
ax.annotate('干扰源G', (G_C[0], G_C[1]), textcoords='offset points', xytext=(8, 8), fontsize=11)
ax.set_xlabel('x(米)'); ax.set_ylabel('y(米)')
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
ax.set_aspect('equal')
save_fig(fig, '图1-3_直径圆覆盖检验.png')

# ---- 图1-4 等边三角形反例 ----
fig, ax = plt.subplots(figsize=(6, 5.5))
fill_polygon(ax, tri, color='tab:orange', alpha=0.35, label='定位区域(正三角形)')
ax.plot([0, 1], [0, 0], color='tab:red', lw=2.0, label='直径D=1')
circle = plt.Circle(m_tri, D_tri/2, fill=False, color='tab:red', lw=2.0, ls='--', label='以D为直径的圆')
ax.add_patch(circle)
circle2 = plt.Circle(c_tri, r_tri, fill=False, color='tab:green', lw=2.0, ls='-.', label=f'最小包围圆(r*=1/√3≈{r_tri:.3f})')
ax.add_patch(circle2)
ax.plot(0.5, np.sqrt(3)/2, 'o', color='tab:purple', ms=10)
ax.annotate('未覆盖顶点\n到圆心距离≈0.866>0.5', (0.5, np.sqrt(3)/2), textcoords='offset points',
            xytext=(30, 0), fontsize=10, color='tab:purple')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
ax.set_aspect('equal')
save_fig(fig, '图1-4_等边三角形反例.png')

# ---- 图1-5 最小包围圆与直径圆对比(场景C)----
fig, ax = plt.subplots(figsize=(7, 6))
fill_polygon(ax, polyC, color='tab:orange', alpha=0.35, label='定位区域')
cC, rC = smallest_enclosing_circle(polyC)
circleD = plt.Circle(mC, D_C/2, fill=False, color='tab:red', lw=2.0, ls='--', label=f'直径圆(半径D/2={D_C/2:.1f})')
circleS = plt.Circle(cC, rC, fill=False, color='tab:green', lw=2.0, ls='-.', label=f'最小包围圆(r*={rC:.1f})')
ax.add_patch(circleD); ax.add_patch(circleS)
ax.plot([pC[0], qC[0]], [pC[1], qC[1]], color='tab:red', lw=2.0)
ax.plot(cC[0], cC[1], 'g+', ms=12)
ax.plot(G_C[0], G_C[1], 'r*', ms=16, zorder=5)
ax.set_xlabel('x(米)'); ax.set_ylabel('y(米)')
ax.legend(loc='lower left', fontsize=10)
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
ax.set_aspect('equal')
save_fig(fig, '图1-5_最小包围圆与直径圆对比.png')

# ---- 图1-6 随机实验 r*/D 箱线图 ----
fig, ax = plt.subplots(figsize=(7, 5))
data = [exp_df[exp_df.k == k]['ratio'].values for k in range(2, 7)]
bp = ax.boxplot(data, patch_artist=True,
                widths=0.6, medianprops=dict(color='tab:red', lw=1.5))
ax.set_xticklabels([f'k={k}' for k in range(2, 7)])
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.axhline(1/np.sqrt(3), color='tab:green', lw=1.5, ls='--')
ax.annotate(r'Jung上界 $1/\sqrt{3}\approx 0.577$', (4.4, 1/np.sqrt(3)+0.004), fontsize=10, color='tab:green')
ax.axhline(0.5, color='tab:red', lw=1.5, ls=':')
ax.annotate('直径圆半径下界 D/2', (0.1, 0.505), fontsize=10, color='tab:red')
ax.set_xlabel('检测点个数k'); ax.set_ylabel('最小包围圆半径与直径之比 r*/D')
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
save_fig(fig, '图1-6_随机实验半径直径比箱线图.png')

# ---- 图1-7 两检测点四边形直径与解析近似对比 ----
# 验证解析式 D ≈ 2ε(d1+d2)/sinφ (两扇形交会四边形)
fig, ax = plt.subplots(figsize=(7, 5))
eps = np.deg2rad(1.0)
d1_list, ratio_list = [], []
for _ in range(400):
    G = np.array([experiment_rng.uniform(-1000, 1000), experiment_rng.uniform(-1000, 1000)])
    d1 = experiment_rng.uniform(300, 1200)
    phi = experiment_rng.uniform(np.deg2rad(25), np.deg2rad(150))
    S1 = G - d1*uvec(experiment_rng.uniform(0, 360))
    # 构造S2使交会角为phi、距离d2
    d2 = d1
    # S2方向: 从G出发与S1夹角phi
    ang1 = np.arctan2(G[1]-S1[1], G[0]-S1[0])
    ang2 = ang1 + phi
    S2 = G - d2*np.array([np.cos(ang2), np.sin(ang2)])
    Bs = make_scenario(G, [S1, S2], seed=None)
    poly, unb = localization_polygon([S1, S2], Bs)
    if unb or len(poly) < 3:
        continue
    D, _ = brute_diameter(poly)
    D_approx = 2*eps*(d1+d2)/np.sin(phi)
    d1_list.append(D_approx); ratio_list.append(D/D_approx)
ratio_list = np.array(ratio_list)
stats_print('数值直径/解析近似', ratio_list)
ax.hist(ratio_list, bins=40, color='tab:blue', alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(1.0, color='tab:red', lw=1.5, ls='--')
ax.set_xlabel('数值直径 / 解析近似 2ε(d1+d2)/sinφ'); ax.set_ylabel('频数')
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
save_fig(fig, '图1-7_直径解析近似验证.png')

print('问题一 全部完成: 6+1张图与结果CSV已输出')
