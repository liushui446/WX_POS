import math
import numpy as np
from scipy.linalg import inv
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体与负号正常显示
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

# -------------------------- 1. 仿真基础参数配置 --------------------------
# 卫星与星座参数
C_N0_nom = 45  # 标称载噪比 (dB-Hz)
C_N0_th = 28   # 接收机跟踪阈值载噪比 (dB-Hz)
c = 3e8        # 光速 (m/s)

signal_type = "CA_code"  # CA_code/P_code/M_code
Tc = 0.9775e-6  # C/A码码元宽度(s)，P码：0.09775e-6，M码：0.1e-6
fs = 1.023e6  # M码副载频(Hz)，仅M码使用

# 接收机参数
Bn = 2.046e6   # 噪声带宽 (Hz, GPS L1 频段)
sigma_rho_nom = 0.2  # 标称伪距基线误差 (m)
G_ant = 10     # 抗干扰波束成形增益 (dB)

jam_type = "continuous_wave"  # continuous_wave/bandlimited_gaussian/pulse
fj = 1575.42e6  # 干扰中心频率(Hz，GPS L1频段)
beta = 20e6  # 干扰带宽(Hz)，带限高斯干扰用
tau = 10e-6  # 脉冲宽度(s)，脉冲干扰用
T = 1e-3  # 脉冲周期(s)，脉冲干扰用

# 干扰参数
P_j = -100     # 干扰功率 (dBm)
P_s = -130     # 卫星信号功率 (dBm)
B_j = 2e6      # 干扰带宽 (Hz, 窄带干扰)

# 积分项参数
d = 1  # 码跟踪误差系数(1或1/8)
integral_range = (-beta/2, beta/2)  # 积分范围（-β/2 ~ β/2）

# 卫星几何参数 (示例: 4颗卫星的方位角/俯仰角，实际需从星历获取)
sat_az = [30, 120, 210, 300]    # 卫星方位角 (°)
sat_el = [45, 45, 45, 45]       # 卫星俯仰角 (°)
user_pos = [0, 0, 0]            # 用户接收机位置 (x,y,z 笛卡尔坐标, m)

# -------------------------- 额外：接收机预定航线与固定基站干扰配置 --------------------------
# 预定航线（地面笛卡尔坐标，单位：米），航线以折线方式连接各航点
waypoints = [(0, 0, 10), (1500, 500, 10)]  # 示例航点 (x, y, z)，z 为高度（m）
traj_total_time = 200  # 总仿真时间 (s)
traj_dt = 1            # 时间步长 (s)

# 固定基站（干扰源）配置：位置、发射功率、干扰类型、干扰带宽、占空比等
# type: 'continuous'|'pulsed'|'narrowband'
jammers = [
    {'pos': (200, 200), 'P_tx': -10, 'type': 'continuous', 'B_j': 2e6, 'duty': 1.0},
    {'pos': (800, 100), 'P_tx': -30, 'type': 'pulsed',     'B_j': 1e6, 'duty': 0.2},
    {'pos': (900, 350), 'P_tx': -20, 'type': 'narrowband', 'B_j': 5e5, 'duty': 1.0},
    {'pos': (600, 300), 'P_tx': -20, 'type': 'continuous', 'B_j': 2e7, 'duty': 1.0},
    {'pos': (1000, 250), 'P_tx': -20, 'type': 'continuous', 'B_j': 3e9, 'duty': 1.0}
]


# ========================= 卫星信号功率谱密度G_S(f)计算 =========================
def calc_GS(f, params):
    """计算卫星信号功率谱密度G_S(f)（文档公式5）"""
    if params.signal_type in ["CA_code", "P_code"]:
        # C/A码/P码：G_S(f) = Tc·sinc²(πfTc)
        return params.Tc * (np.sinc(np.pi * f * params.Tc)) ** 2
    elif params.signal_type == "M_code":
        # M码：G_S(f) = Tm·sinc²(πfTm)·tan²(πf/(2fs))
        tan_term = np.tan(np.pi * f / (2 * params.fs)) if (2 * params.fs) != 0 else 0
        return params.Tc * (np.sinc(np.pi * f * params.Tc)) ** 2 * (tan_term ** 2)
    else:
        return 0

# ========================= 干扰信号功率谱密度G_J(f)计算 =========================
def calc_GJ(f, params):
    """计算干扰信号功率谱密度G_J(f)（文档公式4）"""
    if params.jam_type == "continuous_wave":
        # 连续波干扰：G_J(f) = δ(f - fj)（数值仿真用窄脉冲近似）
        delta = 1e9 if abs(f - params.fj) < 1e3 else 0  # 1e3Hz带宽近似δ函数
        return delta
    elif params.jam_type == "bandlimited_gaussian":
        # 带限高斯干扰：G_J(f) = 1/β
        return 1 / params.beta if abs(f) <= params.beta/2 else 0
    elif params.jam_type == "pulse":
        # 脉冲干扰：G_J(f) = |(τ/T)·ΣSa[(f-fj)πτ]·δ(f-fj + n/T)|²
        n = round((f - params.fj) * params.T)
        sa_term = np.sinc((f - params.fj) * params.tau)
        return (params.tau / params.T) * (sa_term ** 2) if abs(f - params.fj) <= params.beta/2 else 0
    else:
        return 0

# ========================= 4个核心积分项求解函数 =========================
def integral_term1(params):
    """积分项1：∫G_J(f)G_S(f)·sin²(πf d Tc) df"""
    def integrand(f):
        GJ = calc_GJ(f, params)
        GS = calc_GS(f, params)
        sin_term = np.sin(np.pi * f * params.d * params.Tc) ** 2
        return GJ * GS * sin_term
    result, _ = integrate.quad(integrand, params.integral_range[0], params.integral_range[1])
    return result

def integral_term2(params):
    """积分项2：∫f·G_S(f)·sin(πf d Tc) df"""
    def integrand(f):
        GS = calc_GS(f, params)
        sin_term = np.sin(np.pi * f * params.d * params.Tc)
        return f * GS * sin_term
    result, _ = integrate.quad(integrand, params.integral_range[0], params.integral_range[1])
    return result

def integral_term3(params):
    """积分项3：∫G_J(f)G_S(f)·cos²(πf d Tc) df"""
    def integrand(f):
        GJ = calc_GJ(f, params)
        GS = calc_GS(f, params)
        cos_term = np.cos(np.pi * f * params.d * params.Tc) ** 2
        return GJ * GS * cos_term
    result, _ = integrate.quad(integrand, params.integral_range[0], params.integral_range[1])
    return result

def integral_term4(params):
    """积分项4：∫G_S(f)·cos(πf d Tc) df"""
    def integrand(f):
        GS = calc_GS(f, params)
        cos_term = np.cos(np.pi * f * params.d * params.Tc)
        return GS * cos_term
    result, _ = integrate.quad(integrand, params.integral_range[0], params.integral_range[1])
    return result

# ========================= 集成调用函数（一键求解所有积分项） =========================
def solve_all_integrals(params):
    """求解4个核心积分项，返回结果字典"""
    integrals = {
        "term1": integral_term1(params),
        "term2": integral_term2(params),
        "term3": integral_term3(params),
        "term4": integral_term4(params)
    }
    # 输出验证信息（交叉验证：term1 + term3 ≈ ∫GJ·GS df）
    def verify_integral():
        def integrand(f):
            return calc_GJ(f, params) * calc_GS(f, params)
        total, _ = integrate.quad(integrand, params.integral_range[0], params.integral_range[1])
        return abs(integrals["term1"] + integrals["term3"] - total) < 1e-6
    
    print(f"积分项求解完成，验证结果：{'通过' if verify_integral() else '失败'}")
    print(f"积分项1（sin²项）：{integrals['term1']:.6e}")
    print(f"积分项2（f·sin项）：{integrals['term2']:.6e}")
    print(f"积分项3（cos²项）：{integrals['term3']:.6e}")
    print(f"积分项4（cos项）：{integrals['term4']:.6e}")
    return integrals

def interpolate_trajectory(waypoints, total_time, dt):
    """按时间等分插值航点，返回 list of positions (x,y,z) 和 times"""
    # 计算每段地面距离（仅用 x,y）
    dist_list = []
    for i in range(len(waypoints)-1):
        x0, y0 = waypoints[i][0], waypoints[i][1]
        x1, y1 = waypoints[i+1][0], waypoints[i+1][1]
        dist_list.append(math.hypot(x1-x0, y1-y0))
    total_dist = sum(dist_list)
    if total_dist == 0:
        wp = waypoints[0]
        z = wp[2] if len(wp) > 2 else 0
        return [(wp[0], wp[1], z)], [0]
    # 按距离比例分配时间
    num_steps = int(total_time / dt)
    positions = []
    times = []
    # 累积长度
    seg_cum = [0]
    for l in dist_list:
        seg_cum.append(seg_cum[-1] + l)
    for step in range(num_steps+1):
        frac = (step/num_steps) * total_dist
        # find segment
        for k in range(len(dist_list)):
            if seg_cum[k] <= frac <= seg_cum[k+1]:
                seg_len = dist_list[k]
                seg_frac = (frac - seg_cum[k]) / seg_len if seg_len>0 else 0
                x0, y0 = waypoints[k][0], waypoints[k][1]
                x1, y1 = waypoints[k+1][0], waypoints[k+1][1]
                x = x0 + seg_frac * (x1 - x0)
                y = y0 + seg_frac * (y1 - y0)
                # 插值高度 z（若提供）
                z0 = waypoints[k][2] if len(waypoints[k]) > 2 else 0
                z1 = waypoints[k+1][2] if len(waypoints[k+1]) > 2 else 0
                z = z0 + seg_frac * (z1 - z0)
                positions.append((x, y, z))
                times.append(step*dt)
                break
    return positions, times


def calc_jammer_rx_power(jammer, rx_pos, time):
    """估算接收位置从单个干扰源接收到的功率（dBm），使用三维距离"""
    tx = jammer['P_tx']
    jx, jy = jammer['pos'][0], jammer['pos'][1]
    jz = jammer.get('alt', 0.0)
    rx_x, rx_y = rx_pos[0], rx_pos[1]
    rx_z = rx_pos[2] if len(rx_pos) > 2 else 0.0
    d = math.sqrt((rx_x - jx)**2 + (rx_y - jy)**2 + (rx_z - jz)**2)
    d = max(d, 1.0)  # 最小距离1m避免无限大
    # 简化的自由空间衰减模型（仅参考）
    path_loss_db = 20 * math.log10(d)
    P_rx = tx - path_loss_db
    return P_rx


def calc_cn0_with_jammers(C_N0_nom, P_s, jammers, rx_pos, B_n, G_ant, time):
    """根据多个干扰源在接收位置的贡献计算 C/N0_j"""
    P_s_lin = 10 ** (P_s / 10)
    sum_term = 0.0
    for jammer in jammers:
        P_rx = calc_jammer_rx_power(jammer, rx_pos, time)
        # 扣除接收端天线抗干扰增益
        P_jr = P_rx - G_ant
        P_jr_lin = 10 ** (P_jr / 10)
        # 考虑占空比（脉冲干扰）
        if jammer.get('type') == 'pulsed':
            duty = jammer.get('duty', 1.0)
            P_jr_lin *= duty
        B_j_eff = jammer.get('B_j', B_j)
        sum_term += (P_jr_lin * B_n) / (P_s_lin * B_j_eff)
    delta_CN0 = 10 * math.log10(1 + sum_term) if sum_term > 0 else 0.0
    C_N0_j = C_N0_nom - delta_CN0
    return C_N0_j, delta_CN0


def simulate_trajectory_and_errors():
    """按预定航线模拟逐时刻定位误差并绘图（将每个插值点赋给 user_pos，用于计算）"""
    positions, times = interpolate_trajectory(waypoints, traj_total_time, traj_dt)
    sigma_pos_ts = []
    c_n0_list = []
    global user_pos
    for t, pos in zip(times, positions):
        # 将当前插值点赋给 user_pos（x,y,z）用于后续计算
        user_pos = [pos[0], pos[1], pos[2]]
        C_N0_j, _ = calc_cn0_with_jammers(C_N0_nom, P_s, jammers, pos, Bn, G_ant, t)
        c_n0_list.append(C_N0_j)
        sigma_rho = calc_sigma_rho_jamming(C_N0_j, Bn, c, sigma_rho_nom)
        if sigma_rho == np.inf:
            sigma_pos_ts.append(np.nan)
        else:
            pdop = calc_pdop(sat_az, sat_el, user_pos)
            sigma_pos_ts.append(pdop * sigma_rho)
    # 绘制航线与干扰站分布和时间序列
    fig2 = plt.figure(figsize=(12,5))
    axA = plt.subplot(1,2,1)
    xs = [p[0] for p in positions]; ys = [p[1] for p in positions]
    axA.plot(xs, ys, '-o', label='航线轨迹')
    for j in jammers:
        axA.scatter(j['pos'][0], j['pos'][1], s=200, marker='X', label=f"Jammer ({j['type']})")
    axA.set_xlabel('x (m)'); axA.set_ylabel('y (m)'); axA.set_title('航线与固定基站分布')
    axA.legend(); axA.grid(True)
    axB = plt.subplot(1,2,2)
    axB.plot(times, sigma_pos_ts, '-r', marker='o', label='定位误差 σ_pos (m)')
    axB.set_xlabel('时间 (s)'); axB.set_ylabel('三维定位误差 σ_pos (m)')
    axB.set_title('定位误差随时间变化')
    axB.grid(True)
    # 右轴显示 C/N0
    axBc = axB.twinx()
    axBc.plot(times, c_n0_list, '--b', label='C/N0 (dB-Hz)')
    axBc.axhline(y=C_N0_th, color='gray', linestyle='--', alpha=0.7)
    axBc.set_ylabel('C/N0 (dB-Hz)')
    # 合并图例
    lines, labels = axB.get_legend_handles_labels()
    lines2, labels2 = axBc.get_legend_handles_labels()
    axB.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.tight_layout()
    plt.show()

# -------------------------- 2. 载噪比衰减与干扰后载噪比计算 --------------------------
def calc_cn0_jamming(C_N0_nom, P_s, P_j, B_n, B_j, G_ant):
    """计算干扰后的载噪比 C/N0_j"""
    # 干扰功率扣除天线增益
    P_jr = P_j - G_ant  # 接收机输入端干扰功率 (dBm)
    # 线性功率换算
    P_s_lin = 10 ** (P_s / 10)
    P_jr_lin = 10 ** (P_jr / 10)
    # 载噪比衰减量 ΔC/N0
    delta_CN0 = 10 * math.log10(1 + (P_jr_lin * B_n) / (P_s_lin * B_j))
    # 干扰后载噪比
    C_N0_j = C_N0_nom - delta_CN0
    return C_N0_j, delta_CN0

# -------------------------- 3. 干扰下伪距误差计算 --------------------------
def calc_sigma_rho_jamming(C_N0_j, Bn, c, sigma_rho_nom):
    """计算干扰下的总伪距误差"""
    if C_N0_j < C_N0_th:
        return np.inf  # 载噪比低于阈值，卫星失锁
    # 干扰导致的伪距误差
    sigma_rho_jam = c / Bn * (1 / math.sqrt(10 ** (C_N0_j / 10)))
    # 总伪距误差 (基线误差+干扰误差 均方和)
    sigma_rho_total = math.sqrt(sigma_rho_jam ** 2 + sigma_rho_nom ** 2)
    return sigma_rho_total

# -------------------------- 4. PDOP 计算 --------------------------
def calc_pdop(sat_az, sat_el, user_pos):
    """通过卫星方位角/俯仰角计算 PDOP"""
    n_sat = len(sat_az)
    H = np.zeros((n_sat, 4))  # 几何矩阵 (卫星数×4)
    for i in range(n_sat):
        az_rad = math.radians(sat_az[i])
        el_rad = math.radians(sat_el[i])
        # 卫星单位矢量
        x = math.cos(el_rad) * math.cos(az_rad)
        y = math.cos(el_rad) * math.sin(az_rad)
        z = math.sin(el_rad)
        # 几何矩阵行向量 [x,y,z,1]
        H[i] = [x, y, z, 1]
    # 计算 PDOP
    HtH = H.T @ H
    HtH_inv = inv(HtH)
    pdop = math.sqrt(HtH_inv[0,0] + HtH_inv[1,1] + HtH_inv[2,2])
    return pdop

# -------------------------- 5. 定位误差计算主函数 --------------------------
def calc_position_error():
    """按预定航线逐时刻计算定位误差，并保存时间序列供可视化使用"""
    global traj_positions, traj_times, traj_sigma_pos, traj_c_n0
    traj_positions, traj_times = interpolate_trajectory(waypoints, traj_total_time, traj_dt)
    traj_sigma_pos = []
    traj_c_n0 = []

    lost_count = 0
    for t, pos in zip(traj_times, traj_positions):
        # 将当前插值点赋给 user_pos
        global user_pos
        user_pos = [pos[0], pos[1], pos[2]]
        # 计算当前时刻的 C/N0
        C_N0_j, delta = calc_cn0_with_jammers(C_N0_nom, P_s, jammers, user_pos, Bn, G_ant, t)
        traj_c_n0.append(C_N0_j)
        # 计算伪距误差
        sigma_rho = calc_sigma_rho_jamming(C_N0_j, Bn, c, sigma_rho_nom)
        if sigma_rho == np.inf:
            traj_sigma_pos.append(np.nan)
            lost_count += 1
        else:
            pdop = calc_pdop(sat_az, sat_el, user_pos)
            traj_sigma_pos.append(pdop * sigma_rho)

    # 打印摘要信息
    print(f"逐时刻仿真完成，总时刻数: {len(traj_times)}，失锁时刻数: {lost_count}")
    if np.nansum(traj_sigma_pos) == 0:
        print("注意：所有时刻均失锁或定位误差无效。")
    else:
        valid = np.array([v for v in traj_sigma_pos if not np.isnan(v)])
        print(f"最大定位误差: {np.nanmax(traj_sigma_pos):.3f} m，平均有效定位误差: {np.nanmean(valid):.3f} m")

    return traj_times, traj_positions, traj_sigma_pos, traj_c_n0

# -------------------------- 6. 可视化分析函数 --------------------------
def visualize_jamming_analysis():
    """仅绘制：航线地图（含干扰站）与三维定位误差随时间曲线（含 C/N0）"""
    # 需要先运行 calc_position_error() 来填充 traj_* 全局变量
    try:
        times = traj_times
        positions = traj_positions
        sigma_pos_ts = traj_sigma_pos
        c_n0_list = traj_c_n0
    except NameError:
        positions, times = interpolate_trajectory(waypoints, traj_total_time, traj_dt)
        # 若未运行 calc_position_error，则快速计算简单版本
        sigma_pos_ts = []
        c_n0_list = []
        for t, pos in zip(times, positions):
            C_N0_j, _ = calc_cn0_with_jammers(C_N0_nom, P_s, jammers, pos, Bn, G_ant, t)
            c_n0_list.append(C_N0_j)
            sigma_rho = calc_sigma_rho_jamming(C_N0_j, Bn, c, sigma_rho_nom)
            sigma_pos_ts.append(np.nan if sigma_rho == np.inf else calc_pdop(sat_az, sat_el, pos) * sigma_rho)

    # 绘制图形
    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    true_traj = np.array(positions)
    ax1.plot(true_traj[:, 0], true_traj[:, 1], 'y-', label='预定轨迹', linewidth=2)
    ax1.scatter([w[0] for w in waypoints], [w[1] for w in waypoints], c='blue', s=30, marker='.', label='航点')
    shown = set()
    for j in jammers:
        key = j['type']
        label = f"Jammer ({j['type']})" if key not in shown else None
        ax1.scatter(j['pos'][0], j['pos'][1], s=150, marker='X', label=label)
        shown.add(key)
    ax1.scatter(user_pos[0], user_pos[1], c='k', s=50, marker='*', label='接收机当前位置')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)'); ax1.set_title('航线地图与干扰站分布')
    ax1.legend(); ax1.grid(True)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(times, sigma_pos_ts, '-r', marker='o', label='定位误差 σ_pos (m)')
    ax2.set_xlabel('时间 (s)'); ax2.set_ylabel('定位误差 σ_pos (m)')
    ax2.set_title('定位误差随时间变化')
    ax2.grid(True)
    ax2b = ax2.twinx()
    ax2b.plot(times, c_n0_list, '--b', label='C/N0 (dB-Hz)')
    ax2b.axhline(y=C_N0_th, color='gray', linestyle='--', alpha=0.7)
    ax2b.set_ylabel('C/N0 (dB-Hz)')
    # 合并图例
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()

# -------------------------- 7. 数据导出函数 --------------------------
def export_analysis_report():
    """生成分析报告"""
    print("\n" + "="*60)
    print("GNSS 干扰定位误差仿真分析报告")
    print("="*60)
    
    C_N0_j, delta_CN0 = calc_cn0_jamming(C_N0_nom, P_s, P_j, Bn, B_j, G_ant)
    sigma_rho = calc_sigma_rho_jamming(C_N0_j, Bn, c, sigma_rho_nom)
    pdop = calc_pdop(sat_az, sat_el, user_pos)
    sigma_pos = pdop * sigma_rho
    
    print(f"\n【干扰参数】")
    print(f"  干扰功率: {P_j} dBm")
    print(f"  信号功率: {P_s} dBm")
    print(f"  干扰带宽: {B_j/1e6:.1f} MHz")
    print(f"  抗干扰增益: {G_ant} dB")
    
    print(f"\n【接收机参数】")
    print(f"  标称载噪比: {C_N0_nom} dB-Hz")
    print(f"  失锁阈值: {C_N0_th} dB-Hz")
    print(f"  噪声带宽: {Bn/1e6:.2f} MHz")
    print(f"  基线误差: {sigma_rho_nom} m")
    
    print(f"\n【仿真结果】")
    print(f"  干扰后载噪比: {C_N0_j:.2f} dB-Hz")
    print(f"  载噪比衰减: {delta_CN0:.2f} dB")
    print(f"  总伪距误差: {sigma_rho:.4f} m")
    print(f"  PDOP值: {pdop:.2f}")
    print(f"  三维定位误差: {sigma_pos:.4f} m")
    
    if sigma_pos > 100:
        print(f"\n⚠️  警告：定位误差超过100m，定位精度严重下降!")
    elif sigma_pos > 10:
        print(f"\n⚠️  警告：定位误差超过10m，定位精度受到影响!")
    else:
        print(f"\n✓ 定位误差在可接受范围内")
    
    print("\n" + "="*60 + "\n")

# 运行仿真 --------------------------
if __name__ == "__main__":
    print("正在执行GNSS干扰定位误差仿真...")
    # 在主流程中使用按时序的航线仿真计算
    times, positions, sigma_pos_ts, c_n0 = calc_position_error()
    # export_analysis_report()
    print("生成可视化分析图表...")
    visualize_jamming_analysis()
