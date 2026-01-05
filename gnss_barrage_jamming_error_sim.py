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
        self.c_light = 3e8        # 光速 (m/s)

        self.signal_type = "CA_code"  # CA_code/P_code/M_code
        self.Tc = 0.9775e-6  # C/A码码元宽度(s)，P码：0.09775e-6，M码：0.1e-6
        self.fs = 1.023e6  # M码副载频(Hz)，仅M码使用
        self.C = 1e-16  # 卫星信号载波功率(W)
        self.sat_carrier_power = self.C  # 兼容 calc_cnr 函数命名

        # 接收机参数
        self.Bn = 2.046e6   # 噪声带宽 (Hz, GPS L1 频段)
        self.sigma_rho_nom = 0.2  # 标称伪距基线误差 (m)
        self.G_ant = 10     # 抗干扰波束成形增益 (dB)

        self.jam_type = "continuous_wave"  # continuous_wave/bandlimited_gaussian/pulse
        self.fj = 1575.42e6  # 干扰中心频率(Hz，GPS L1频段)
        self.beta_r = 20e6  # 接收机前端等效预相关带宽  [0.5e6, 1e6, 2e6, 5e6, 10e6]
        self.fc = self.fj  # 兼容 sim.py 的命名（中心频率）
        self.tau = 10e-6  # 脉冲宽度(s)，脉冲干扰用
        self.Bp = 18  # PLL 带宽(Hz)
        self.Bd = self.Bp  # DLL 带宽(Hz)
        self.Td = 0.02  # 相关积分时间(s)

        # 连续波窄带近似带宽（Hz），用于将 δ 函数近似为窄带矩形谱
        self.continuous_bw = 1e4  # 10 kHz 默认值
        self.T = 1e-3  # 脉冲周期(s)，脉冲干扰用

        # 干扰参数
        self.P_j = -100     # 干扰功率 (dBm)
        self.P_s = -130     # 卫星信号功率 (dBm)
        self.B_j = 2e6      # 干扰带宽 (Hz, 窄带干扰)

        # 积分项参数
        self.d = 1  # 码跟踪误差系数(1或1/8)
        self.integral_range = (-self.beta_r/2, self.beta_r/2)  # 积分范围（-β/2 ~ β/2）
        self.epsabs = 1e-6  # 积分绝对精度控制

        # 5. 失锁阈值
        self.pll_unlock_thresh = 15  # 载波环失锁阈值(°)
        self.dll_unlock_thresh = self.d / 6  # 码环失锁阈值        # 跟踪环带宽与相关积分时间（用于 PLL/DLL 计算）
        
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
    # 假定 pos 单位与轨迹一致（米），并且 P_tx 以 dBm 表示
    P_tx_dbm = jammer.get('P_tx', params.P_j if hasattr(params, 'P_j') else params.P_j)
    jam_power_w = 10 ** ((P_tx_dbm - 30) / 10)  # dBm -> W

    jx, jy = jam_pos[0], jam_pos[1]
    jz = jam_pos[2] if len(jam_pos) > 2 else jammer.get('alt', 0.0)
    rx_x, rx_y, rx_z = target_pos[0], target_pos[1], target_pos[2]

    # 距离 (m)
    dist = math.sqrt((rx_x - jx)**2 + (rx_y - jy)**2 + (rx_z - jz)**2)
    dist = max(dist, 1.0)  # 避免零

    # 天线仰角
    horiz = math.hypot(rx_x - jx, rx_y - jy)
    antenna_elevation = math.degrees(math.atan2(abs(rx_z - jz), horiz))

    if antenna_elevation > 5 or dist < 5000:
        loss = (4 * math.pi * dist * params.fc / params.c_light) ** 2
    elif dist < 20000:
        refraction_factor = 1.0003
        loss = (4 * math.pi * dist * params.fc * refraction_factor / params.c_light) ** 2
    elif rx_z * 1000 < 10000:
        loss = (4 * math.pi * dist * params.fc / params.c_light) ** 2 * 1.2
    else:
        loss = (4 * math.pi * dist * params.fc / params.c_light) ** 2 * 0.8

    # 干扰接收功率 (W)
    Pj = jam_power_w / loss
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
    # 避免除零
    if integral_result <= 0 or Pj <= 0:
        return 0

    # 载噪比计算（线性值）
    C_NJ_linear = params.sat_carrier_power / (Pj * integral_result)
    # 转换为dB-Hz
    C_NJ_dB = 10 * np.log10(C_NJ_linear) if C_NJ_linear > 0 else 0
    return C_NJ_dB

# ========================= 功率谱密度计算 =========================
def calc_GS(f, params):
    """计算卫星信号功率谱密度G_S(f)。
    输入 f 为绝对频率(Hz)，这里将其转换为相对载频的偏移 f_rel = f - fc 后计算基带谱。
    """
    f_rel = f - params.fc
    if params.signal_type in ["CA_code", "P_code"]:
        return params.Tc * (np.sinc(np.pi * f_rel * params.Tc)) ** 2
    elif params.signal_type == "M_code":
        tan_term = np.tan(np.pi * f_rel / (2 * params.fs)) if (2 * params.fs) != 0 else 0
        return params.Tc * (np.sinc(np.pi * f_rel * params.Tc)) ** 2 * (tan_term ** 2)
    return 0

def calc_GJ(f, params):
    """计算干扰信号功率谱密度G_J(f) """
    if params.jam_type == "continuous_wave":
        # 将 δ 近似为宽度为 continuous_bw 的矩形功率谱，幅值归一化为 1/带宽
        bw = getattr(params, 'continuous_bw', 1e3)
        return (1.0 / bw) if abs(f - params.fj) <= bw/2 else 0
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
        # 载噪比无效，返回失锁指示
        return np.inf, True, C_NJ_dB
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

# ========================= PDOP计算 =========================
def calc_pdop(sat_az, sat_el, rx_pos):
    """计算位置稀释因子PDOP (Position Dilution of Precision)
    :param sat_az: 卫星方位角列表 (°)
    :param sat_el: 卫星仰角列表 (°)
    :param rx_pos: 接收机位置 (x,y,z)
    :return: PDOP值
    """
    # 构建几何矩阵 H
    H = []
    for az, el in zip(sat_az, sat_el):
        az_rad = np.radians(az)
        el_rad = np.radians(el)
        # 单位向量：sin(el)*cos(az), sin(el)*sin(az), cos(el), 1
        h_row = [
            np.sin(el_rad) * np.cos(az_rad),
            np.sin(el_rad) * np.sin(az_rad),
            np.cos(el_rad),
            1.0
        ]
        H.append(h_row)
    
    H = np.array(H)
    
    # 计算 (H^T H)^(-1)
    try:
        HTH = H.T @ H
        HTH_inv = inv(HTH)
        # PDOP = sqrt(trace of position components)
        pdop = np.sqrt(HTH_inv[0, 0] + HTH_inv[1, 1] + HTH_inv[2, 2])
    except:
        pdop = np.inf
    
    return pdop

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
# -------------------------- 5. 仿真主流程（sim.py 风格） --------------------------

def main_simulation():
    """逐时刻按预定航线仿真，使用 SimParams、APM 多源干扰与跟踪环误差判定"""
    params = SimParams()

    # 若需要可覆盖默认参数，例如使用当前文件中定义的 waypoints
    positions, times = interpolate_trajectory(params.waypoints, params.traj_total_time, params.traj_dt)

    traj_true = []
    traj_err = []
    cnr_ts = []
    unlock_ts = []
    sigma_jpll_ts = []

    for t_idx, (t, pos) in enumerate(zip(times, positions)):
        rx_pos = np.array([pos[0], pos[1], pos[2]])
        traj_true.append(rx_pos)

        # 多源干扰功率累加（APM）
        Pj_total = 0.0
        for jammer in params.jammers:
            Pj_total += calc_jam_power_apm(jammer, rx_pos, params)

        # 将多源合成的干扰功率注入到 params 中，以供后续的跟踪环误差计算使用
        params.Pj = Pj_total
        # 合成带宽采用最大干扰带宽的近似
        params.B_j = max([j.get('B_j', params.B_j) for j in params.jammers])

        # 计算 C/N0 (dB-Hz)
        cnr = calc_cnr(Pj_total, params)
        # 若积分项导致 calc_cnr 返回 0（无法积分或模型退化），采用能量比近似并考虑带宽作为回退
        if (cnr == 0 or cnr <= 0) and Pj_total > 0:
            B_j_eff = max(getattr(params, 'B_j', 1e6), 1e3)
            delta_CN0 = 10 * np.log10(1 + (Pj_total * params.Bn) / (params.sat_carrier_power * B_j_eff))
            cnr = params.C_N0_nom - delta_CN0
        cnr_ts.append(cnr)

        if t_idx < 20:
            # 增加更多诊断信息：积分项、C/NJ、σ_PLL、σ_DLL 和判定
            term1 = integral_term1(params)
            term3 = integral_term3(params)
            cnj = calc_CNJ(params)
            sigma_pll, pll_unlock_calc, cnj_dB2 = calc_sigma_PLL(params)
            sigma_dll, dll_unlock_calc = calc_sigma_DLL_NELP(params)
            print(f"诊断 t={t} s: Pj_total={Pj_total:.3e} W, C/N0={cnr:.2f} dB-Hz, integral1={term1:.3e}, integral3={term3:.3e}, C/NJ={cnj:.3e}, σ_PLL={sigma_pll:.3f}, σ_DLL={sigma_dll:.3e}, unlock_calc={pll_unlock_calc or dll_unlock_calc}")

        # 计算跟踪环误差并判定失锁（使用现有函数）
        sigma_pll, pll_unlock, cnj_dB = calc_sigma_PLL(params)
        sigma_dll, dll_unlock = calc_sigma_DLL_NELP(params)

        unlock = pll_unlock or dll_unlock
        unlock_ts.append(unlock)

        # 如果未失锁，估算定位误差（简化: 用伪距误差近似）
        if not unlock:
            # 伪距测量误差（m）
            sigma_rho = sigma_dll * params.c_light if not np.isinf(sigma_dll) else np.inf
            # 估算 PDOP
            pdop = calc_pdop(params.sat_az, params.sat_el, rx_pos)
            pos_err = pdop * sigma_rho
        else:
            pos_err = np.nan

        traj_err.append([pos_err, pdop if not unlock else np.nan])
        sigma_jpll_ts.append(sigma_pll)

    # 转换为 numpy 数组
    traj_true = np.array(traj_true)
    traj_err = np.array(traj_err)
    cnr_ts = np.array(cnr_ts)
    unlock_ts = np.array(unlock_ts)
    sigma_jpll_ts = np.array(sigma_jpll_ts)

    print(f"仿真结束：总时刻={len(times)}, 失锁时刻={np.sum(unlock_ts)}")

    visualize_results(traj_true, traj_err, cnr_ts, unlock_ts, times, sigma_jpll_ts, params)


# 可视化函数（简洁明了）
def visualize_results(true_traj, err_traj, cnr_ts, unlock_ts, times, sigma_jpll_ts, params):
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(true_traj[:, 0], true_traj[:, 1], 'y-', linewidth=2, label='航线')
    ax1.scatter([w[0] for w in params.waypoints], [w[1] for w in params.waypoints], c='blue', s=30, marker='.', label='航点')
    for j in params.jammers:
        ax1.scatter(j['pos'][0], j['pos'][1], c='red', marker='x', s=80, label=f"干扰源:{j.get('type','')}")
    # 去重图例
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax1.set_title('航线与干扰站分布')
    ax1.grid(True)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(times, err_traj[:, 0], '-r', marker='o', label='估算定位误差 (m)')
    ax2.set_xlabel('时间 (s)'); ax2.set_ylabel('定位误差 (m)')
    ax2.grid(True)

    ax2b = ax2.twinx()
    ax2b.plot(times, cnr_ts, '--b', label='C/N0 (dB-Hz)')
    ax2b.plot(times, sigma_jpll_ts, ':g', label='σ_JPLL (°)')
    ax2b.axhline(y=params.C_N0_th, color='gray', linestyle='--', alpha=0.7)

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()


# 运行仿真 --------------------------
if __name__ == "__main__":
    print("正在执行GNSS干扰定位误差仿真（sim.py 风格）...")
    main_simulation()
