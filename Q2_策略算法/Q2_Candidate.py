# -*- coding: utf-8 -*-
"""
问题二: 已知某全向干扰源在一个检测点处测得的示向度, 
给出第二个检测点的选择策略与候选区域。
  1) 交会定位四边形面积解析式 A = 4*eps^2*d1*d2/sin(phi)(数值验证)
  2) 候选区域: 保证接收的胶囊形区域(扇形四角点的1000米圆交)
  3) 极小极大最优第二检测点 (t*, L*): 最坏情形面积最小
  4) 两阶段自适应策略(先判段后优化)与蒙特卡洛对比
先算后画。
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
from scipy.optimize import minimize
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
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])

EPS_DEG = 1.0                     # 示向度误差界(度)
EPS = np.deg2rad(EPS_DEG)         # 弧度
R_MIN = 1000.0                    # 有效接收半径下限(保证接收)
D1_RANGE = (5.0, 1500.0)          # d1的可能范围(S1已收到信号)

# ---------- 定位四边形面积(数值, 与问题一同算法) ----------
def wedge_halfplanes(S, theta_deg):
    t = np.deg2rad(theta_deg)
    u1 = np.array([np.cos(t-EPS), np.sin(t-EPS)])
    u2 = np.array([np.cos(t+EPS), np.sin(t+EPS)])
    S = np.asarray(S, float)
    return [(u1[1], -u1[0], u1[1]*S[0]-u1[0]*S[1]),
            (-u2[1], u2[0], -u2[1]*S[0]+u2[0]*S[1])]

def clip_polygon(poly, hp, eps=1e-9):
    a, b, c = hp
    if len(poly) == 0:
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
            out.append([P[0]+t*(Q[0]-P[0]), P[1]+t*(Q[1]-P[1])])
    cleaned = []
    for p in out:
        if not cleaned or np.hypot(p[0]-cleaned[-1][0], p[1]-cleaned[-1][1]) > 1e-6:
            cleaned.append(p)
    if len(cleaned) > 1 and np.hypot(cleaned[0][0]-cleaned[-1][0], cleaned[0][1]-cleaned[-1][1]) <= 1e-6:
        cleaned.pop()
    return cleaned

def localization_area(points, bearings, cap=6000.0):
    poly = [[-cap, -cap], [cap, -cap], [cap, cap], [-cap, cap]]
    unbounded = False
    for S, th in zip(points, bearings):
        for hp in wedge_halfplanes(S, th):
            poly = clip_polygon(poly, hp)
    if not poly:
        return 0.0, True
    for p in poly:
        if abs(abs(p[0])-cap) < 1e-6 or abs(abs(p[1])-cap) < 1e-6:
            unbounded = True
    A = 0.0
    n = len(poly)
    for i in range(n):
        A += poly[i][0]*poly[(i+1) % n][1] - poly[i][1]*poly[(i+1) % n][0]
    return abs(A)/2.0, unbounded

def area_analytic(d1, d2, phi):
    """A = 4 eps^2 d1 d2 / sin(phi), phi为交会角(弧度)"""
    return 4*EPS**2*d1*d2/np.sin(phi)

# ---------- 计算 ----------
# 1) 面积解析式数值验证
rng = np.random.default_rng(7)
ratios = []
for _ in range(800):
    d1 = rng.uniform(200, 1400)
    d2 = rng.uniform(200, 1400)
    phi = rng.uniform(np.deg2rad(20), np.deg2rad(150))
    S1 = np.array([0.0, 0.0])
    S2 = np.array([rng.uniform(-200, 200), rng.uniform(-200, 200)])
    G = S1 + d1*uvec(rng.uniform(0, 360))
    # 调整S2使G-S2距离=d2且交会角=phi
    ang1 = np.arctan2(G[1]-S1[1], G[0]-S1[0])
    S2 = G - d2*np.array([np.cos(ang1+phi), np.sin(ang1+phi)])
    th1 = np.rad2deg(np.arctan2(G[1]-S1[1], G[0]-S1[0])) % 360
    th2 = np.rad2deg(np.arctan2(G[1]-S2[1], G[0]-S2[0])) % 360
    A_num, unb = localization_area([S1, S2], [th1, th2])
    if unb or A_num <= 0:
        continue
    ratios.append(A_num/area_analytic(d1, d2, phi))
ratios = np.array(ratios)
print(f"[面积解析式验证] n={len(ratios)} min={ratios.min():.4f} max={ratios.max():.4f} "
      f"mean={ratios.mean():.4f} std={ratios.std(ddof=1):.4f}")

def stats_print(name, arr):
    arr = np.asarray(arr, float)
    print(f"[{name}] n={len(arr)} min={arr.min():.4f} max={arr.max():.4f} "
          f"mean={arr.mean():.4f} std={arr.std(ddof=1):.4f} "
          f"CV={arr.std(ddof=1)/arr.mean():.4f} amplitude={arr.max()-arr.min():.4f}")

# 2) 候选区域与极小极大最优
# 扇形可能位置W的四个角点(凸集, 最远点必在角点)
corner_dist = [D1_RANGE[0], D1_RANGE[1]]
W_corners = []
for d in corner_dist:
    for s in [-1, 1]:
        W_corners.append(np.array([d*np.cos(s*EPS), d*np.sin(s*EPS)]))

def feasible_S2(t, L, R=R_MIN):
    """S2=(t沿示向线, L垂直) 是否对W中一切G满足 |S2-G|<=R"""
    for c in W_corners:
        if (t-c[0])**2 + (L-c[1])**2 > R**2 + 1e-6:
            return False
    return True

def corner_area(G, S1, S2):
    """给定真实干扰源位置G与两检测点, 计算交会定位区域面积(示向度=真实方位)"""
    d1 = np.hypot(*(G-S1)); d2 = np.hypot(*(G-S2))
    cross = abs((G[0]-S1[0])*(G[1]-S2[1]) - (G[1]-S1[1])*(G[0]-S2[0]))
    if cross < 1e-12 or d1 < 1e-12 or d2 < 1e-12:
        return np.inf
    sin_phi = cross/(d1*d2)
    return 4*EPS**2*d1*d2/sin_phi

def worst_area(t, L, dmax=D1_RANGE[1], dmin=D1_RANGE[0]):
    """最坏情形面积: G遍历扇形W的四个角点(凸集最远/最偏点)"""
    S1 = np.array([0.0, 0.0])
    S2 = np.array([t, L])
    worst = 0.0
    for d in [dmin, dmax]:
        for s in [-1, 1]:
            G = np.array([d*np.cos(s*EPS), d*np.sin(s*EPS)])
            worst = max(worst, corner_area(G, S1, S2))
    return worst

# 数值最优化(极小极大: 四角点最坏面积, 最近距离≥5.1保证非"near")
def neg_objective(x):
    t, L = x
    if not feasible_S2(t, L):
        return 1e9
    # 距离约束: 与一切可能G的距离≥5.1米(避免near)
    for c in W_corners:
        if (t-c[0])**2 + (L-c[1])**2 < 5.1**2:
            return 1e9
    return worst_area(t, L)

best = None
for t0 in np.linspace(300, 1000, 15):
    for L0 in np.linspace(100, 900, 17):
        res = minimize(neg_objective, [t0, L0], method='Nelder-Mead',
                       options=dict(xatol=1e-6, fatol=1e-6, maxiter=2000))
        if best is None or res.fun < best.fun:
            best = res
t_star, L_star = best.x
A_worst_star = best.fun
print(f"极小极大最优: t*={t_star:.2f}米 L*={L_star:.2f}米 最坏面积={A_worst_star:.2f}平方米")
print(f"  S2位置 = S1 + {t_star:.1f}u ± {L_star:.1f}n")

# 3) 两阶段自适应: S2' = S1 + 1000u 判段
#    判段依据: |theta2 - theta1| > 90° -> G在两检测点之间(d1<1000)
#    段1: d1 in [5,1000]; 段2: d1 in [1005,1500]
def segment_opt(dlo, dhi):
    def corners(d):
        return [np.array([d*np.cos(s*EPS), d*np.sin(s*EPS)]) for s in [-1, 1]]
    Cs = corners(dlo) + corners(dhi)
    def feas(x):
        t, L = x
        for c in Cs:
            d2 = (t-c[0])**2 + (L-c[1])**2
            if d2 > R_MIN**2 + 1e-6 or d2 < 5.1**2:
                return False
        return True
    def wA(x):
        t, L = x
        if not feas(x):
            return 1e9
        S1 = np.array([0.0, 0.0]); S2 = np.array([t, L])
        worst = 0.0
        for G in Cs:
            worst = max(worst, corner_area(G, S1, S2))
        return worst
    b = None
    for t0 in np.linspace(dlo, dhi, 12):
        for L0 in np.linspace(50, 950, 12):
            r = minimize(wA, [t0, L0], method='Nelder-Mead',
                         options=dict(xatol=1e-6, fatol=1e-6, maxiter=2000))
            if b is None or r.fun < b.fun:
                b = r
    return b.x, b.fun

seg1 = segment_opt(5.0, 1000.0)
seg2 = segment_opt(1005.0, 1500.0)
print(f"自适应策略段1(d1∈[5,1000]): S3 = S1 + {seg1[0][0]:.1f}u ± {seg1[0][1]:.1f}n, 最坏面积={seg1[1]:.2f}")
print(f"自适应策略段2(d1∈[1005,1500]): S3 = S1 + {seg2[0][0]:.1f}u ± {seg2[0][1]:.1f}n, 最坏面积={seg2[1]:.2f}")

# 4) 蒙特卡洛策略对比
def strategy_area(G, S2, S1=np.array([0.0, 0.0])):
    th1 = np.rad2deg(np.arctan2(G[1]-S1[1], G[0]-S1[0])) % 360
    th2 = np.rad2deg(np.arctan2(G[1]-S2[1], G[0]-S2[0])) % 360
    A, unb = localization_area([S1, S2], [th1, th2])
    return A if not unb else np.inf

strategy_names = ['沿示向线1000米', '垂直偏移600米', '极小极大最优', '45度折中700米', '已知距离最优(5.1米)']
mc = {n: [] for n in strategy_names}
worst = {n: 0 for n in strategy_names}
for trial in range(3000):
    d1 = rng.uniform(50, 1500)
    alpha = rng.uniform(-EPS, EPS)
    G = np.array([d1*np.cos(alpha), d1*np.sin(alpha)])
    candidates = {
        '沿示向线1000米': np.array([1000.0, 0.0]),
        '垂直偏移600米': np.array([0.0, 600.0]),
        '极小极大最优': np.array([t_star, L_star]),
        '45度折中700米': np.array([700*np.cos(np.pi/4), 700*np.sin(np.pi/4)]),
        '已知距离最优(5.1米)': np.array([d1, 5.1]),
    }
    for name, S2 in candidates.items():
        A = strategy_area(G, S2)
        # 接收半径检查(有效半径1000-1500, 保守用1000)
        if np.hypot(*(G-S2)) > 1000:
            A = np.inf
        mc[name].append(A)
        if np.isfinite(A):
            worst[name] = max(worst[name], A)
for name in strategy_names:
    arr = np.array([a for a in mc[name] if np.isfinite(a)])
    fail = sum(1 for a in mc[name] if not np.isfinite(a))
    stats_print(f"策略[{name}] 面积", arr)
    print(f"    最坏={worst[name]:.1f} 接收失败次数={fail}/3000")

# 5) 最优偏移随距离不确定范围的变化(t*、L*、最坏面积随dmax)
dmax_grid = np.linspace(300, 1500, 13)
opt_rows = []
for dmax in dmax_grid:
    def wA2(x):
        t, L = x
        Cs = []
        for d in [5.0, dmax]:
            Cs += [np.array([d*np.cos(s*EPS), d*np.sin(s*EPS)]) for s in [-1, 1]]
        for c in Cs:
            dd2 = (t-c[0])**2 + (L-c[1])**2
            if dd2 > R_MIN**2 + 1e-6 or dd2 < 5.1**2:
                return 1e9
        S1 = np.array([0.0, 0.0]); S2 = np.array([t, L])
        worst = 0.0
        for G in Cs:
            worst = max(worst, corner_area(G, S1, S2))
        return worst
    b = None
    for t0 in np.linspace(5, dmax, 10):
        for L0 in np.linspace(50, 950, 10):
            r = minimize(wA2, [t0, L0], method='Nelder-Mead',
                         options=dict(xatol=1e-6, fatol=1e-6, maxiter=1500))
            if b is None or r.fun < b.fun:
                b = r
    opt_rows.append(dict(dmax=dmax, t_star=b.x[0], L_star=b.x[1], A_worst=b.fun))
    print(f"dmax={dmax:5.0f}: t*={b.x[0]:6.1f} L*={b.x[1]:6.1f} 最坏面积={b.fun:7.1f}")
opt_df = pd.DataFrame(opt_rows)
save_csv(opt_df, '问题二_最优偏移随不确定范围.csv')

# 策略对比汇总CSV
summary_rows = []
for name in strategy_names:
    arr = np.array([a for a in mc[name] if np.isfinite(a)])
    summary_rows.append(dict(策略=name, 样本数=len(arr), 平均面积=round(arr.mean(),2),
                             中位面积=round(float(np.median(arr)),2), 标准差=round(arr.std(ddof=1),2),
                             最坏面积=round(worst[name],2), 接收失败数=3000-len(arr)))
sum_df = pd.DataFrame(summary_rows)
save_csv(sum_df, '问题二_策略对比汇总.csv')
print(sum_df.to_string(index=False))

# 候选区域网格(用于热力图)
xgrid = np.linspace(-1200, 2600, 190)
ygrid = np.linspace(-900, 900, 90)
XX, YY = np.meshgrid(xgrid, ygrid)
Feas = np.zeros_like(XX, dtype=bool)
WorstArea = np.full_like(XX, np.nan)
for i in range(XX.shape[0]):
    for j in range(XX.shape[1]):
        t, L = XX[i, j], YY[i, j]
        if feasible_S2(t, L) and min((t-c[0])**2 + (L-c[1])**2 for c in W_corners) >= 5.1**2:
            Feas[i, j] = True
            WorstArea[i, j] = worst_area(t, L)

# ---------- 画图 ----------
# 图2-1 示向度扇形与候选区域
fig, ax = plt.subplots(figsize=(7.5, 5.5))
# 扇形W
for s in [-1, 1]:
    v = np.array([np.cos(s*EPS), np.sin(s*EPS)])
    ax.plot([0, 1500*v[0]], [0, 1500*v[1]], color='tab:blue', lw=1.2)
ax.plot([0, 1500], [0, 0], color='tab:blue', lw=1.2, ls='--', alpha=0.6)
ax.fill_between([0, 1500], [-1500*np.sin(EPS)], [1500*np.sin(EPS)], alpha=0.10, color='tab:blue')
ax.plot([0, 0], [-1500*np.sin(EPS), 1500*np.sin(EPS)], color='tab:blue', lw=1.2)
ax.plot(0, 0, 'ks', ms=8)
ax.annotate('检测点S1', (0, 0), textcoords='offset points', xytext=(-10, -18), fontsize=11)
# 候选区域
cont = ax.contour(XX, YY, Feas.astype(float), levels=[0.5], colors='tab:red', linewidths=2)
ax.annotate('候选区域\n(保证接收)', (400, 420), fontsize=12, color='tab:red')
ax.plot(t_star, L_star, 'r*', ms=15, zorder=5)
ax.annotate(f'S2*=(t*,L*)=({t_star:.0f},{L_star:.0f})', (t_star, L_star),
            textcoords='offset points', xytext=(14, 6), fontsize=11, color='tab:red')
# 角点
for c in W_corners:
    ax.plot(c[0], c[1], 'o', color='tab:blue', ms=5)
ax.set_xlabel('沿示向线方向(米)'); ax.set_ylabel('垂直示向线方向(米)')
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
ax.set_aspect('equal')
save_fig(fig, '图2-1_示向度扇形与候选区域.png')

# 图2-2 定位区域面积随垂直偏移L变化(多d1)
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
for k, d1 in enumerate([300, 600, 900, 1200, 1500]):
    Ls = np.linspace(50, 900, 200)
    As = []
    for L in Ls:
        t = 0.0
        G = np.array([d1, 0.0])
        d2 = np.hypot(t-G[0], L-G[1])
        phi = np.arcsin(np.clip(abs(d1*L)/(d1*d2), -1, 1))
        As.append(area_analytic(d1, d2, phi))
    ax.plot(Ls, As, color=colors[k], lw=2, marker='o', ms=2, label=f'd1={d1}米')
for k, d1 in enumerate([300, 600, 900, 1200, 1500]):
    ax.axvline(d1, color=colors[k], ls=':', lw=1.2, alpha=0.7)
ax.set_xlabel('垂直偏移L(米)'); ax.set_ylabel('定位区域面积A(平方米)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
save_fig(fig, '图2-2_面积随垂直偏移变化.png')

# 图2-3 交会角与面积关系
fig, ax = plt.subplots(figsize=(7, 5))
phis = np.linspace(np.deg2rad(5), np.deg2rad(175), 300)
for k, (d1, d2) in enumerate([(500, 500), (1000, 1000), (1500, 1500), (1000, 500)]):
    A = area_analytic(d1, d2, phis)
    ax.plot(np.rad2deg(phis), A, lw=2, label=f'd1={d1}, d2={d2}')
ax.set_xlabel('交会角φ(度)'); ax.set_ylabel('定位区域面积A(平方米)')
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
save_fig(fig, '图2-3_交会角与面积关系.png')

# 图2-4 两阶段自适应策略流程图
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.axis('off')
box = dict(boxstyle='round,pad=0.45', fc='white', ec='tab:blue', lw=1.5)
nodes = [
    ('S1处测示向度θ1', (1, 4.5)),
    ('S2\'=S1+1000u处\n测得示向度θ2', (1, 3.0)),
    ('|θ2-θ1|>90°?', (1, 1.6)),
    ('d1∈[5,1000]米\n(源在两检测点之间)', (4.2, 2.6)),
    ('d1∈[1005,1500]米\n(源在S2\'外侧)', (4.2, 0.6)),
    ('S3=S1+t1*u±L1*n\n(段1极小极大点)', (7.6, 2.6)),
    ('S3=S1+t2*u±L2*n\n(段2极小极大点)', (7.6, 0.6)),
]
for txt, (x, y) in nodes:
    ax.text(x, y, txt, ha='center', va='center', fontsize=10.5, bbox=box)
arr = dict(arrowstyle='-|>', color='tab:blue', lw=1.6)
ax.annotate('', (1, 3.55), (1, 3.95), arrowprops=arr)
ax.annotate('', (1, 2.15), (1, 2.45), arrowprops=arr)
ax.annotate('', (3.65, 2.6), (1.55, 1.75), arrowprops=arr)
ax.annotate('', (3.65, 0.6), (1.55, 1.5), arrowprops=arr)
ax.annotate('', (7.0, 2.6), (4.75, 2.6), arrowprops=arr)
ax.annotate('', (7.0, 0.6), (4.75, 0.6), arrowprops=arr)
ax.text(1, 1.55, '是', fontsize=11, color='tab:red', ha='center')
ax.text(1, 1.15, '否', fontsize=11, color='tab:green', ha='center')
ax.text(2.6, 2.35, '是', fontsize=11, color='tab:red')
ax.text(2.6, 0.35, '否', fontsize=11, color='tab:green')
ax.set_xlim(0, 9); ax.set_ylim(0, 5.4)
save_fig(fig, '图2-4_两阶段自适应策略流程图.png')

# 图2-5 策略对比箱线图(蒙特卡洛)
fig, ax = plt.subplots(figsize=(8, 5))
data = [np.array([a for a in mc[n] if np.isfinite(a)]) for n in strategy_names]
bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                medianprops=dict(color='tab:red', lw=1.5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(strategy_names)))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticklabels(['沿示向线\n1000米', '垂直偏移\n600米', '极小极大\n最优', '45°折中\n700米', '已知距离\n最优(5.1米)'], fontsize=9.5)
ax.set_ylabel('定位区域面积A(平方米)')
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
save_fig(fig, '图2-5_策略对比箱线图.png')

# 图2-6 候选区域内部定位质量热力图
fig, ax = plt.subplots(figsize=(8, 4.5))
W = WorstArea.copy()
W[~Feas] = np.nan
im = ax.imshow(W, extent=[xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]], origin='lower',
               cmap='viridis_r', aspect='auto', vmin=200, vmax=6000)
ax.contour(XX, YY, Feas.astype(float), levels=[0.5], colors='red', linewidths=1.8)
cb = plt.colorbar(im, ax=ax)
cb.set_label('最坏情形定位面积(平方米)')
ax.plot(t_star, L_star, 'r*', ms=14)
ax.annotate(f'最优S2*=({t_star:.0f},{L_star:.0f})', (t_star, L_star),
            textcoords='offset points', xytext=(12, 6), fontsize=10, color='white')
ax.plot(0, 0, 'ks', ms=7)
ax.annotate('S1', (0, 0), textcoords='offset points', xytext=(-14, -16), fontsize=11)
ax.set_xlabel('沿示向线方向(米)'); ax.set_ylabel('垂直示向线方向(米)')
despine(ax)
save_fig(fig, '图2-6_候选区域定位质量热力图.png')

# 图2-7 最优偏移随dmax变化
fig, ax = plt.subplots(figsize=(7, 5))
ax2 = ax.twinx()
ax.plot(opt_df.dmax, opt_df.t_star, 'o-', color='tab:blue', lw=2, label='t*(沿示向线偏移)')
ax.plot(opt_df.dmax, opt_df.L_star, 's-', color='tab:green', lw=2, label='L*(垂直偏移)')
ax2.plot(opt_df.dmax, opt_df.A_worst, '^-', color='tab:red', lw=2, label='最坏情形面积A')
ax.set_xlabel('距离不确定范围上界dmax(米)')
ax.set_ylabel('最优偏移(米)')
ax2.set_ylabel('最坏情形面积(平方米)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc='upper left', fontsize=9)
ax.grid(alpha=0.3, linestyle='--'); despine(ax)
save_fig(fig, '图2-7_最优偏移随不确定范围.png')

print('问题二 全部完成')
