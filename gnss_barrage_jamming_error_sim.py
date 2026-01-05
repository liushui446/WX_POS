import math
import numpy as np
from scipy.linalg import inv
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体与负号正常显示
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

# ========================= 1. 核心参数配置（参考仿真系统取值）=========================
class SimParams:
    def __init__(self):
        # 卫星与星座参数
        self.C_N0_nom = 45  # 标称载噪比 (dB-Hz)
        self.C_N0_th = 28   # 接收机跟踪阈值载噪比 (dB-Hz)
        self.c = 3e8        # 光速 (m/s)

        self.signal_type = "CA_code"  # CA_code/P_code/M_code
        self.Tc = 0.9775e-6  # C/A码码元宽度(s)，P码：0.09775e-6，M码：0.1e-6
        self.fs = 1.023e6  # M码副载频(Hz)，仅M码使用
        self.C = 1e-16  # 卫星信号载波功率(W)

        # 接收机参数
        self.Bn = 2.046e6   # 噪声带宽 (Hz, GPS L1 频段)
        self.sigma_rho_nom = 0.2  # 标称伪距基线误差 (m)
        self.G_ant = 10     # 抗干扰波束成形增益 (dB)

        self.jam_type = "continuous_wave"  # continuous_wave/bandlimited_gaussian/pulse
        self.fj = 1575.42e6  # 干扰中心频率(Hz，GPS L1频段)
        self.beta = 20e6  # 干扰带宽(Hz)，带限高斯干扰用
        self.tau = 10e-6  # 脉冲宽度(s)，脉冲干扰用
        self.T = 1e-3  # 脉冲周期(s)，脉冲干扰用

        # 干扰参数
        self.P_j = -100     # 干扰功率 (dBm)
        self.P_s = -130     # 卫星信号功率 (dBm)
        self.B_j = 2e6      # 干扰带宽 (Hz, 窄带干扰)

        # 积分项参数
        self.d = 1  # 码跟踪误差系数(1或1/8)
        self.integral_range = (-self.beta/2, self.beta/2)  # 积分范围（-β/2 ~ β/2）

        # 5. 失锁阈值
        self.pll_unlock_thresh = 15  # 载波环失锁阈值(°)
        self.dll_unlock_thresh = self.d / 6  # 码环失锁阈值

        # 卫星几何参数 (示例: 4颗卫星的方位角/俯仰角，实际需从星历获取)
        self.sat_az = [30, 120, 210, 300]    # 卫星方位角 (°)
        self.sat_el = [45, 45, 45, 45]       # 卫星俯仰角 (°)
        self.user_pos = [0, 0, 0]            # 用户接收机位置 (x,y,z 笛卡尔坐标, m)

        # 定位误差结果
        self.sigma_rho_ts = []  # 伪距误差时间序列

        # -------------------------- 额外：接收机预定航线与固定基站干扰配置 --------------------------
        # 预定航线（地面笛卡尔坐标，单位：米），航线以折线方式连接各航点
        self.waypoints = [(0, 0, 10), (1500, 500, 10)]  # 示例航点 (x, y, z)，z 为高度（m）
        self.traj_total_time = 200  # 总仿真时间 (s)
        self.traj_dt = 1            # 时间步长 (s)

        # 固定基站（干扰源）配置：位置、发射功率、干扰类型、干扰带宽、占空比等
        # type: 'continuous'|'pulsed'|'narrowband'
        self.jammers = [
            {'pos': (200, 200), 'P_tx': -10, 'type': 'continuous', 'B_j': 2e6, 'duty': 1.0},
            {'pos': (800, 100), 'P_tx': -30, 'type': 'pulsed',     'B_j': 1e6, 'duty': 0.2},
            {'pos': (900, 350), 'P_tx': -20, 'type': 'narrowband', 'B_j': 5e5, 'duty': 1.0},
            {'pos': (600, 300), 'P_tx': -20, 'type': 'continuous', 'B_j': 2e7, 'duty': 1.0},
            {'pos': (1000, 250), 'P_tx': -20, 'type': 'continuous', 'B_j': 3e9, 'duty': 1.0}
        ]


# ========================= 2. APM电磁传播模型（计算干扰信号功率）=========================
def calc_jam_power_apm(jammer, target_pos, params):
    """
    基于APM模型计算来自单个干扰源到达接收机的干扰信号功率P_J
    :param jammer: 干扰源 dict，包含 'pos' (经纬高), 'power' (W), 可选其他字段
    :param target_pos: 目标位置(经纬高)
    :param params: 仿真参数类
    :return: 干扰信号接收功率P_J(W)
    """
    jam_pos = jammer['pos']
    jam_power = jammer.get('power', params.jam_power)
    # 转换为距离（经纬度转地表距离，简化处理，实际需用WGS84坐标系）
    lon_diff = (jam_pos[0] - target_pos[0]) * np.pi / 180
    lat_diff = (jam_pos[1] - target_pos[1]) * np.pi / 180
    earth_radius = 6371  # 地球半径(km)
    dist_horizontal = earth_radius * np.sqrt(lon_diff**2 + lat_diff**2)  # 水平距离(km)
    dist_vertical = abs(jam_pos[2] - target_pos[2])  # 垂直距离(km)
    dist = np.sqrt(dist_horizontal**2 + dist_vertical**2) * 1000  # 总距离(米)

    # APM模型选择（根据距离和仰角）
    antenna_elevation = np.arctan2(dist_vertical * 1000, dist_horizontal * 1000) * 180 / np.pi  # 天线仰角(°)

    if antenna_elevation > 5 or dist < 5000:
        # FE模型（平面地球，忽略折射和曲率）
        loss = (4 * np.pi * dist * params.fc / params.c) ** 2  # 自由空间损耗
    elif dist < 20000:
        # RO模型（射线光学，考虑折射和曲率）
        refraction_factor = 1.0003  # 大气折射系数
        loss = (4 * np.pi * dist * params.fc * refraction_factor / params.c) ** 2
    elif target_pos[2] * 1000 < 10000:
        # PE模型（抛物方程，适用于中距离低空）
        loss = (4 * np.pi * dist * params.fc / params.c) ** 2 * 1.2  # 额外损耗系数
    else:
        # XO模型（扩展光学，适用于高空）
        loss = (4 * np.pi * dist * params.fc / params.c) ** 2 * 0.8  # 损耗修正

    # 干扰接收功率 = 发射功率 * 天线增益（简化为1） / 路径损耗
    Pj = jam_power / loss
    return Pj

# ========================= 4. 载噪比计算（C/NJ）=========================
def calc_cnr(Pj, params):
    """
    计算接收机前端载噪比C/NJ
    :param Pj: 干扰信号接收功率(W)
    :param params: 仿真参数类
    :return: C/NJ (dB-Hz)
    """
    # 计算积分项：∫GJ(f)GS(f)df（积分范围：-β/2 到 β/2）
    integrand = lambda f: calc_GJ(f, params) * calc_GS(f, params)
    integral_result, _ = integrate.quad(integrand, params.fc - params.beta/2, params.fc + params.beta/2)
    
    # 载噪比计算（线性值）
    C_NJ_linear = params.sat_carrier_power / (Pj * integral_result)
    # 转换为dB-Hz
    C_NJ_dB = 10 * np.log10(C_NJ_linear) if C_NJ_linear > 0 else 0
    return C_NJ_dB

# ========================= 功率谱密度计算 =========================
def calc_GS(f, params):
    """计算卫星信号功率谱密度G_S(f) """
    if params.signal_type in ["CA_code", "P_code"]:
        return params.Tc * (np.sinc(np.pi * f * params.Tc)) ** 2
    elif params.signal_type == "M_code":
        tan_term = np.tan(np.pi * f / (2 * params.fs)) if (2 * params.fs) != 0 else 0
        return params.Tc * (np.sinc(np.pi * f * params.Tc)) ** 2 * (tan_term ** 2)
    return 0

def calc_GJ(f, params):
    """计算干扰信号功率谱密度G_J(f) """
    if params.jam_type == "continuous_wave":
        return 1e9 if abs(f - params.fj) < 1e3 else 0  # 窄脉冲近似δ函数
    elif params.jam_type == "bandlimited_gaussian":
        return 1 / params.beta if abs(f) <= params.beta/2 else 0
    elif params.jam_type == "pulse":
        n = round((f - params.fj) * params.T)
        sa_term = np.sinc((f - params.fj) * params.tau)
        return (params.tau / params.T) * (sa_term ** 2) if abs(f - params.fj) <= params.beta/2 else 0
    return 0

# ========================= 4个核心积分项求解 =========================
def integral_term1(params):
    """积分项1：∫G_J(f)G_S(f)·sin²(πf d Tc) df"""
    def integrand(f):
        return calc_GJ(f, params) * calc_GS(f, params) * np.sin(np.pi * f * params.d * params.Tc) ** 2
    result, _ = integrate.quad(integrand, params.integral_range[0], params.integral_range[1], epsabs=params.epsabs)
    return result

def integral_term2(params):
    """积分项2：∫f·G_S(f)·sin(πf d Tc) df"""
    def integrand(f):
        return f * calc_GS(f, params) * np.sin(np.pi * f * params.d * params.Tc)
    result, _ = integrate.quad(integrand, params.integral_range[0], params.integral_range[1], epsabs=params.epsabs)
    return result

def integral_term3(params):
    """积分项3：∫G_J(f)G_S(f)·cos²(πf d Tc) df"""
    def integrand(f):
        return calc_GJ(f, params) * calc_GS(f, params) * np.cos(np.pi * f * params.d * params.Tc) ** 2
    result, _ = integrate.quad(integrand, params.integral_range[0], params.integral_range[1], epsabs=params.epsabs)
    return result

def integral_term4(params):
    """积分项4：∫G_S(f)·cos(πf d Tc) df"""
    def integrand(f):
        return calc_GS(f, params) * np.cos(np.pi * f * params.d * params.Tc)
    result, _ = integrate.quad(integrand, params.integral_range[0], params.integral_range[1], epsabs=params.epsabs)
    return result

# ========================= 载噪比C/N_J计算 =========================
def calc_CNJ(params):
    """计算载噪比C/N_J (dB-Hz)"""
    # 计算∫G_J(f)G_S(f)df = term1 + term3
    integral_GJGS = integral_term1(params) + integral_term3(params)
    if integral_GJGS <= 0 or params.Pj <= 0:
        return 0  # 避免除零
    C_NJ_linear = params.C / (params.Pj * integral_GJGS)
    return 10 * np.log10(C_NJ_linear) if C_NJ_linear > 0 else 0

# ========================= σ_J PLL（振荡器颤动）计算 =========================
def calc_sigma_PLL(params):
    """计算载波锁相环振荡器颤动σ_J PLL"""
    C_NJ_dB = calc_CNJ(params)
    if C_NJ_dB <= 0:
        return np.inf  # 载噪比无效，返回无穷大
    C_NJ_linear = 10 ** (C_NJ_dB / 10)
    
    # 文档公式：σ_J PLL = (360/(2π)) * sqrt( Bp/(C/NJ) * (1 + 1/(2Td·C/NJ)) )
    term = (params.Bp / C_NJ_linear) * (1 + 1 / (2 * params.Td * C_NJ_linear))
    sigma_pll = (360 / (2 * np.pi)) * np.sqrt(term)
    
    # 失锁判定
    unlock_flag = sigma_pll > params.pll_unlock_thresh
    return sigma_pll, unlock_flag, C_NJ_dB

# ========================= σ_J DLL(NELP)（码跟踪误差）计算 =========================
def calc_sigma_DLL_NELP(params):
    """计算NELP码跟踪误差σ_J DLL(NELP)"""
    # 求解4个积分项
    term1 = integral_term1(params)
    term2 = integral_term2(params)
    term3 = integral_term3(params)
    term4 = integral_term4(params)
    
    # 避免除零
    if term2 == 0 or term4 == 0:
        return np.inf, False
    
    # 文档公式核心计算
    numerator = np.sqrt(params.Bd) * term1
    denominator = 2 * np.pi * term2
    term = (params.Pj * term3) / (params.Td * (term4 ** 2))
    sigma_dll = (numerator / denominator) * np.sqrt(1 + term)
    
    # 失锁判定
    unlock_flag = sigma_dll > params.dll_unlock_thresh
    return sigma_dll, unlock_flag

# ========================= 集成计算函数（一键输出所有结果）=========================
def calc_PLL_DLL_errors(params):
    """集成计算：σ_J PLL、σ_J DLL(NELP)及失锁状态"""
    # 计算σ_J PLL
    sigma_pll, pll_unlock, cnj_dB = calc_sigma_PLL(params)
    # 计算σ_J DLL(NELP)
    sigma_dll, dll_unlock = calc_sigma_DLL_NELP(params)
    
    # 输出结果汇总
    result = {
        "载噪比C/NJ": f"{cnj_dB:.2f} dB-Hz",
        "振荡器颤动σ_J PLL": f"{sigma_pll:.4f} °",
        "载波环失锁状态": "失锁" if pll_unlock else "正常",
        "码跟踪误差σ_J DLL(NELP)": f"{sigma_dll:.6e} s",
        "码环失锁状态": "失锁" if dll_unlock else "正常",
        "伪距测量误差": f"{sigma_dll * params.C:.6e} m"  # 码跟踪误差→伪距误差(Δρ = σ_dll * c)
    }

    return result


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


def calc_cn0_with_jammers(params, rx_pos, time):
    """根据多个干扰源在接收位置的贡献计算 C/N0_j"""
    P_s_lin = 10 ** (params.P_s / 10)
    sum_term = 0.0
    for jammer in params.jammers:
        P_rx = calc_jammer_rx_power(jammer, rx_pos, time)
        # 扣除接收端天线抗干扰增益
        P_jr = P_rx - params.G_ant
        P_jr_lin = 10 ** (P_jr / 10)
        # 考虑占空比（脉冲干扰）
        if jammer.get('type') == 'pulsed':
            duty = jammer.get('duty', 1.0)
            P_jr_lin *= duty
        B_j_eff = jammer.get('B_j', params.B_j)
        sum_term += (P_jr_lin * params.B_n) / (P_s_lin * B_j_eff)
    delta_CN0 = 10 * math.log10(1 + sum_term) if sum_term > 0 else 0.0
    C_N0_j = params.C_N0_nom - delta_CN0
    return C_N0_j, delta_CN0

# -------------------------- 5. 定位误差计算主函数 --------------------------
def calc_position_error():
    # 初始化参数
    params = SimParams()

    """按预定航线逐时刻计算定位误差，并保存时间序列供可视化使用"""
    global traj_positions, traj_times, traj_sigma_pos, traj_c_n0
    traj_positions, traj_times = interpolate_trajectory(params.waypoints, params.traj_total_time, params.traj_dt)
    traj_sigma_pos = []
    traj_c_n0 = []

    lost_count = 0
    for t, pos in zip(traj_times, traj_positions):
        # 1.将当前插值点赋给 user_pos
        global user_pos
        user_pos = [pos[0], pos[1], pos[2]]

        # 2. 计算干扰信号接收功率（来自多个干扰源的贡献）
        Pj_total = 0.0
        for jammer in params.jammers:
            Pj_total += calc_jam_power_apm(jammer, user_pos, params)

        # 3. 计算载噪比
        C_NJ_dB = calc_cnr(Pj_total, params)
        if C_NJ_dB < 30:  # 干扰有效（载噪比低于正常阈值）
            traj_c_n0.append(C_NJ_dB)

            # 3.计算当前时刻的 PLL 与 DLL 误差
            results = calc_PLL_DLL_errors(params)

            # 4.判断当前时刻的 载波相位跟踪环路 PLL 或 码跟踪环路 DLL 是否失锁误
            if results["载波环失锁状态"] == "失锁" or results["码环失锁状态"] == "失锁":
                params.sigma_rho_ts.append(np.nan)  # 失锁时刻定位误差无效
                lost_count += 1
                continue

            params.sigma_rho_ts.append(results["伪距测量误差"])

    # 打印摘要信息
    print(f"逐时刻仿真完成，总时刻数: {len(traj_times)}，失锁时刻数: {lost_count}")
    if np.nansum(params.sigma_rho_ts) == 0:
        print("注意：所有时刻均失锁或定位误差无效。")
    else:
        valid = np.array([v for v in params.sigma_rho_ts if not np.isnan(v)])
        print(f"最大定位误差: {np.nanmax(params.sigma_rho_ts):.3f} m，平均有效定位误差: {np.nanmean(valid):.3f} m")
    # export_analysis_report()
    print("生成可视化分析图表...")
    visualize_jamming_analysis(params)


# -------------------------- 6. 可视化分析函数 --------------------------
def visualize_jamming_analysis(params):
    """仅绘制：航线地图（含干扰站）与三维定位误差随时间曲线（含 C/N0）"""
    # 需要先运行 calc_position_error() 来填充 traj_* 全局变量
    try:
        times = traj_times
        positions = traj_positions
        sigma_pos_ts = traj_sigma_pos
        c_n0_list = traj_c_n0
    except NameError:
        print("请先运行 calc_position_error() 来生成数据！")
        return

    # 绘制图形
    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    true_traj = np.array(positions)
    ax1.plot(true_traj[:, 0], true_traj[:, 1], 'y-', label='预定轨迹', linewidth=2)
    ax1.scatter([w[0] for w in params.waypoints], [w[1] for w in params.waypoints], c='blue', s=30, marker='.', label='航点')
    shown = set()
    for j in params.jammers:
        key = j['type']
        label = f"Jammer ({j['type']})" if key not in shown else None
        ax1.scatter(j['pos'][0], j['pos'][1], s=150, marker='X', label=label)
        shown.add(key)
    ax1.scatter(user_pos[0], user_pos[1], c='k', s=50, marker='*', label='接收机当前位置')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)'); ax1.set_title('航线地图与干扰站分布')
    ax1.legend(); ax1.grid(True)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(times, params.sigma_pos_ts, '-r', marker='o', label='定位误差 σ_pos (m)')
    ax2.set_xlabel('时间 (s)'); ax2.set_ylabel('定位误差 σ_pos (m)')
    ax2.set_title('定位误差随时间变化')
    ax2.grid(True)
    ax2b = ax2.twinx()
    ax2b.plot(times, c_n0_list, '--b', label='C/N0 (dB-Hz)')
    ax2b.axhline(y=params.C_N0_th, color='gray', linestyle='--', alpha=0.7)
    ax2b.set_ylabel('C/N0 (dB-Hz)')
    # 合并图例
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()

# 运行仿真 --------------------------
if __name__ == "__main__":
    print("正在执行GNSS干扰定位误差仿真...")
    # 在主流程中使用按时序的航线仿真计算
    calc_position_error()
