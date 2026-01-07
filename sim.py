import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# 设置中文字体与负号正常显示（Windows 常用字体：Microsoft YaHei / SimHei）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 1. 核心参数配置（贴合仿真系统取值）=========================
class SimParams:
    def __init__(self):
        # 基础物理参数
        self.c = 3e8  # 光速(m/s)
        self.Td = 0.02  # 相关积分时间(s)
        self.fc = 1575.42e6  # GNSS中心频率(GPS L1频段, Hz)

        # 目标平台参数（巡航导弹示例）
        self.target_init_pos = np.array([119.045, 27.2233, 1.3])  # 初始经纬高(°/km)
        self.target_velocity = np.array([0.0005, 0.0003, 0])  # 速度(°/s, °/s, km/s)
        self.sim_time = 200  # 仿真时长(s)
        self.sampling_freq = 1  # 采样频率(Hz)
        
        # 干扰装备参数（保留单源参数作为默认，同时支持多个干扰源列表）
        self.jam_type = "multi-tone"  # 干扰样式：continuous_wave/multi-tone/bandlimited_gaussian/pulse
        self.jam_power = 50  # 单个干扰源发射功率(W)（默认）
        self.jam_bandwidth = 20e6  # 干扰带宽(Hz)
        self.pulse_width = 10e-6  # 脉冲宽度(s)，仅脉冲干扰用
        self.pulse_period = 1e-3  # 脉冲周期(s)，仅脉冲干扰用
        self.multi_tone_freqs = [1575.42e6 + 1e6, 1575.42e6 - 1e6]  # 多音干扰频点(Hz)
        self.jam_gaint = 1e9# 天线增益

        # 支持多个干扰源：在预定轨迹中心附近布置（范围较小，默认半径约0.01°≈1.1km）
        # 生成策略：以轨迹中点为中心，等角度分布 5 个干扰源，参数可在 self.jammers 中查看/修改
        center = self.target_init_pos + self.target_velocity * (self.sim_time / 2)
        center_lon, center_lat, center_alt = center
        spread_deg = 0.01  # 半径（度），约1.11 km（范围不要太大）
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # 5 等分角
        powers = [50, 30, 20, 10, 5]
        types = ['multi-tone', 'continuous_wave', 'pulsed', 'bandlimited_gaussian', 'multi-tone']
        bandwidths = [20e6, 1e6, 1e6, 10e6, 5e6]
        self.jammers = []
        for i, ang in enumerate(angles):
            lon = center_lon + spread_deg * math.cos(ang)
            lat = center_lat + spread_deg * math.sin(ang)
            alt = center_alt  # 使用与轨迹中心相近高度
            jammer = {'pos': np.array([lon, lat, alt]), 'power': powers[i], 'type': types[i], 'bandwidth': bandwidths[i]}
            if types[i] == 'pulsed':
                jammer.update({'pulse_width': 5e-6, 'pulse_period': 1e-3, 'duty': 0.005})
            self.jammers.append(jammer)
        
        # 导航装备参数
        self.combined_nav = "loose"  # 组合导航方式：loose/tight/deep（松/紧/深耦合）
        self.anti_jam_filter = "frequency"  # 抗干扰滤波：time/frequency/spatial（时/频/空域）
        self.beta = 20e6  # 接收机等效预相关带宽(Hz)
        self.Tc = 0.9775e-6  # 伪码码元宽度(s)，C/A码典型值
        self.d = 1  # 码跟踪误差系数(1或1/8)
        self.ins_drift = 0.01  # 惯导漂移率(km/s)，失锁时用
        
        # 卫星参数（模拟4颗GDOP最优卫星）
        self.satellite_pos = np.array([[125.0, 30.0, 20000],  # 卫星经纬高(°/km)
                                       [115.0, 35.0, 20000],
                                       [130.0, 25.0, 20000],
                                       [110.0, 28.0, 20000]])
        self.sat_carrier_power = 1e-16  # 卫星信号载波功率(C, W)
        
        # 跟踪环参数（与组合导航方式绑定）
        self.Bp = 18 if self.combined_nav in ["loose", "tight"] else 2  # PLL带宽(Hz)
        self.Bd = self.Bp  # DLL带宽(Hz)，与PLL一致
        
        # 失锁判定阈值
        self.pll_unlock_thresh = 15  # 载波环失锁阈值(°)
        self.dll_unlock_thresh = self.d / 6  # 码环失锁阈值

# ========================= . 经纬度转换ECEF(单位:米)=========================
def lla_to_ecef(lon_deg, lat_deg, h_km):
    """
    将经纬度高度（lat_deg, lon_deg, h_km，单位：度/度/千米）转换为 ECEF（米）。
    返回 [X, Y, Z]（单位：米）。
    """
    a = 6378137.0
    f = 1/298.257223563
    e2 = 2*f - f*f
    h_m = h_km * 1000.0  # km -> m

    φ = np.deg2rad(lat_deg); λ = np.deg2rad(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(φ)**2)
    X = (N + h_m) * np.cos(φ) * np.cos(λ)
    Y = (N + h_m) * np.cos(φ) * np.sin(λ)
    Z = ((1 - e2) * N + h_m) * np.sin(φ)
    return np.array([X, Y, Z])        


# ========================= ECEF 转回 经纬度高度（单位: 度/度/米）=========================
def ecef_to_lla(X, Y, Z):
    """
    将 ECEF（米）转换为经纬度高度：返回 (lon_deg, lat_deg, h_m)。
    使用 Bowring / 非迭代近似方法，适用于定位精度需求的仿真场景。
    """
    a = 6378137.0
    f = 1/298.257223563
    b = a * (1 - f)
    e2 = 2*f - f*f
    ep2 = (a**2 - b**2) / (b**2)

    p = np.sqrt(X*X + Y*Y)
    # 处理极区（p near 0）情况
    if p < 1e-12:
        lon = 0.0
        lat = np.sign(Z) * (np.pi/2)
        h = abs(Z) - b
        return lon, np.rad2deg(lat), h

    theta = np.arctan2(Z * a, p * b)
    lon = np.arctan2(Y, X)
    lat = np.arctan2(Z + ep2 * b * np.sin(theta)**3,
                     p - e2 * a * np.cos(theta)**3)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = (p / np.cos(lat) - N) / 1000.0  # m -> km
    return np.rad2deg(lon), np.rad2deg(lat), h

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
    # 转换为距离（经纬度转ecef坐标）
    jam_ecef = lla_to_ecef(jam_pos[0], jam_pos[1], jam_pos[2])
    target_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])
    dist = np.linalg.norm(jam_ecef - target_ecef)  # 总距离(米)

    dist_vertical = abs((jam_pos[2] - target_pos[2]) * 1000)  # 垂直距离(米)
    dist_horizontal = np.sqrt(dist**2 - dist_vertical**2)  # 水平距离(米)

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
    Pj = jam_power * params.jam_gaint / loss
    return Pj

# ========================= 3. 功率谱密度计算（随干扰样式变化）=========================
def calc_power_spectral_density(f, params):
    """
    计算干扰信号功率谱密度GJ(f)和卫星信号功率谱密度GS(f)
    :param f: 频率点(Hz)
    :param params: 仿真参数类
    :return: GJ(f), GS(f)
    """
    # 卫星信号功率谱密度GS(f)（C/A码）
    GS = params.Tc * (np.sinc(np.pi * f * params.Tc)) ** 2
    
    # 干扰信号功率谱密度GJ(f)（随干扰样式变化）
    if params.jam_type == "continuous_wave":
        # 单频干扰：δ(f - fj)
        GJ = 1 if abs(f - params.fc) < 1e3 else 0
    elif params.jam_type == "multi-tone":
        # 多音干扰：Σδ(f - fn)
        GJ = 1 if any(abs(f - freq) < 1e3 for freq in params.multi_tone_freqs) else 0
    elif params.jam_type == "bandlimited_gaussian":
        # 带限高斯干扰：1/β
        GJ = 1 / params.jam_bandwidth if abs(f - params.fc) < params.jam_bandwidth/2 else 0
    elif params.jam_type == "pulse":
        # 脉冲干扰：|(τ/T)ΣSa[(f-fj)πτ]δ(f-fj + n/T)|²
        tau = params.pulse_width
        T = params.pulse_period
        fj = params.fc
        n = round((f - fj) * T)
        sa_term = np.sinc((f - fj) * tau)
        GJ = (tau / T) * (sa_term ** 2) if abs(f - fj) < params.jam_bandwidth/2 else 0
    else:
        GJ = 0
    
    return GJ, GS

# ========================= 4. 载噪比计算（C/NJ）=========================
def calc_cnr(Pj, params):
    """
    计算接收机前端载噪比C/NJ
    :param Pj: 干扰信号接收功率(W)
    :param params: 仿真参数类
    :return: C/NJ (dB-Hz)
    """
    # 计算积分项：∫GJ(f)GS(f)df（积分范围：-β/2 到 β/2）
    integrand = lambda f: calc_power_spectral_density(f, params)[0] * calc_power_spectral_density(f, params)[1]
    integral_result, _ = integrate.quad(integrand, params.fc - params.beta/2, params.fc + params.beta/2)
    
    # 载噪比计算（线性值）
    C_NJ_linear = params.sat_carrier_power / (Pj * integral_result)
    # 转换为dB-Hz
    C_NJ_dB = 10 * np.log10(C_NJ_linear) if C_NJ_linear > 0 else 0
    return C_NJ_dB

# ========================= 5. 跟踪环误差计算（PLL+DLL）=========================
def calc_tracking_errors(C_NJ_dB, params):
    """
    计算载波环振荡器颤动σ_JPLL和码环跟踪误差σ_JDLL(NELP)
    :param C_NJ_dB: 载噪比(dB-Hz)
    :param params: 仿真参数类
    :return: σ_JPLL(°), σ_JDLL
    """
    # 转换载噪比为线性值
    C_NJ_linear = 10 ** (C_NJ_dB / 10)
    
    # 1. 载波环振荡器颤动σ_JPLL
    term = (params.Bp / C_NJ_linear) * (1 + 1 / (2 * params.Td * C_NJ_linear))
    sigma_jpll = (360 / (2 * np.pi)) * np.sqrt(term)
    
    # 2. 码环跟踪误差σ_JDLL(NELP)
    # 计算积分项
    """积分项1：∫G_J(f)G_S(f)·sin²(πf d Tc) df"""
    def integrand_sin2(f):
        GJ, GS = calc_power_spectral_density(f, params)
        return GJ * GS * (np.sin(np.pi * f * params.d * params.Tc)) ** 2
    
    """积分项2：∫f·G_S(f)·sin(πf d Tc) df"""
    def integrand_fsin(f):
        GJ, GS = calc_power_spectral_density(f, params)
        return f * GS * np.sin(np.pi * f * params.d * params.Tc)
    
    """积分项3：∫G_J(f)G_S(f)·cos²(πf d Tc) df"""
    def integrand_cos2(f):
        GJ, GS = calc_power_spectral_density(f, params)
        return GJ * GS * (np.cos(np.pi * f * params.d * params.Tc)) ** 2
    
    """积分项4：∫G_S(f)·cos(πf d Tc) df"""
    def integrand_cos(f):
        GJ, GS = calc_power_spectral_density(f, params)
        return GS * np.cos(np.pi * f * params.d * params.Tc)
    
    integral_sin2, _ = integrate.quad(integrand_sin2, params.fc - params.beta/2, params.fc + params.beta/2)
    integral_fsin, _ = integrate.quad(integrand_fsin, params.fc - params.beta/2, params.fc + params.beta/2)
    integral_cos2, _ = integrate.quad(integrand_cos2, params.fc - params.beta/2, params.fc + params.beta/2)
    integral_cos, _ = integrate.quad(integrand_cos, params.fc - params.beta/2, params.fc + params.beta/2)
    
    # 码环误差公式（简化参数，Pj已包含在C/NJ中）
    numerator = np.sqrt(params.Bd) * integral_sin2
    denominator = 2 * np.pi * integral_fsin
    term2 = (integral_cos2) / (params.Td * (integral_cos ** 2))
    sigma_jdll = (numerator / denominator) * np.sqrt(1 + term2) if denominator != 0 else 0
    
    return sigma_jpll, sigma_jdll

# ========================= 6. GDOP计算（几何精度因子）=========================
def calc_gdop(target_pos, satellite_pos):
    """
    计算几何精度因子GDOP
    :param target_pos: 目标位置(经纬高)
    :param satellite_pos: 卫星位置数组
    :return: GDOP值
    """
    # 构建几何矩阵G（简化版，实际需转换为ECEF坐标系）
    G = []
    for sat in satellite_pos:
        delta = sat - target_pos
        dist = np.linalg.norm(delta)
        if dist == 0:
            dist = 1e-6
        unit_vec = delta / dist
        G.append([unit_vec[0], unit_vec[1], unit_vec[2], 1])
    G = np.array(G)
    
    # GDOP = sqrt(trace((G^T G)^-1))
    try:
        GtG_inv = np.linalg.inv(G.T @ G)
        gdop = np.sqrt(np.trace(GtG_inv))
    except np.linalg.LinAlgError:
        gdop = 10  # 求解失败时设为大值（最差情况）
    return gdop

# ========================= 7. 定位误差求解（分失锁/未失锁场景）=========================
def solve_position_error(target_pos, satellite_pos, Pj, params):
    """
    分场景计算定位误差
    :return: 三维定位误差(Δx, Δy, Δz)，失锁标志(unlock_flag)
    """
    # 1. 计算载噪比
    C_NJ_dB = calc_cnr(Pj, params)
    if C_NJ_dB < 30:  # 干扰有效（载噪比低于正常阈值）
        # 2. 计算跟踪环误差
        sigma_jpll, sigma_jdll = calc_tracking_errors(C_NJ_dB, params)
        
        # 3. 判定失锁状态
        pll_unlock = sigma_jpll > params.pll_unlock_thresh
        dll_unlock = sigma_jdll > params.dll_unlock_thresh
        unlock_flag = pll_unlock or dll_unlock
        
        if unlock_flag:
            # 场景1：失锁，使用惯导定位，误差随时间累积
            pos_error = np.array([params.ins_drift, params.ins_drift, 0.01])  # 简化累积偏差
        else:
            # 场景2：未失锁，伪距误差×GDOP
            pseudo_range_error = sigma_jdll * params.c  # 码跟踪误差转换为伪距误差(米)
            gdop = calc_gdop(target_pos, satellite_pos)
            # 转换为经纬高误差（简化：米→度，1度≈111km）
            pos_error = (pseudo_range_error / 1000 / 111) * gdop  # 经度/纬度误差(°)
            pos_error = np.array([pos_error, pos_error, pseudo_range_error / 1000 * gdop])  # 高度误差(km)
    else:
        # 干扰无效，定位误差为正常GNSS误差
        pos_error = np.array([0.0001, 0.0001, 0.001])
        unlock_flag = False
    
    return pos_error, unlock_flag, C_NJ_dB

# ========================= 8. 主仿真流程=========================
def main_simulation():
    # 初始化参数
    params = SimParams()
    time_steps = int(params.sim_time * params.sampling_freq)
    target_positions = []  # 目标真实轨迹
    error_positions = []   # 受干扰后轨迹
    errors = []            # 定位误差序列
    cnr_list = []          # 载噪比序列
    unlock_flags = []      # 失锁标志序列
    
    # 仿真主循环
    for t in range(time_steps):
        # 1. 计算目标实时位置
        target_pos = params.target_init_pos + params.target_velocity * t
        target_positions.append(target_pos)
        
        # 2. 计算干扰信号接收功率（来自多个干扰源的贡献）
        Pj_total = 0.0
        for jammer in params.jammers:
            Pj_total += calc_jam_power_apm(jammer, target_pos, params)

        # 3. 求解定位误差（使用所有干扰源的总接收功率）
        pos_error, unlock_flag, cnr = solve_position_error(target_pos, params.satellite_pos, Pj_total, params)
        errors.append(pos_error)
        unlock_flags.append(unlock_flag)
        cnr_list.append(cnr)

        
        # 4. 计算受干扰后目标位置
        error_pos = target_pos + pos_error
        error_positions.append(error_pos)
    
    # 转换为numpy数组
    target_positions = np.array(target_positions)
    error_positions = np.array(error_positions)
    errors = np.array(errors)
    cnr_list = np.array(cnr_list)
    unlock_flags = np.array(unlock_flags)
    
    # 5. 结果可视化
    visualize_results(target_positions, error_positions, errors, cnr_list, unlock_flags, params)

# ========================= 9. 结果可视化（轨迹+误差+载噪比）=========================
def visualize_results(true_traj, error_traj, errors, cnr_list, unlock_flags, params):
    time = np.arange(len(errors)) / params.sampling_freq
    plt.figure(figsize=(15, 10))
    
    # 1. 2D轨迹对比（经纬度平面）
    plt.subplot(2, 2, 1)
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'g-', label='预定轨迹', linewidth=2)
    plt.plot(error_traj[:, 0], error_traj[:, 1], 'r-', label='受干扰轨迹', linewidth=2)
    # 绘制所有干扰源
    for j in params.jammers:
        plt.scatter(j['pos'][0], j['pos'][1], c='blue', marker='x', s=100, label=f"干扰源 ({j.get('type','')})")
    # 确保图例项不重复
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('经度(°)')
    plt.ylabel('纬度(°)')
    plt.legend()
    plt.title(f'压制干扰轨迹对比（{params.jam_type}）')
    plt.grid(True)
    
    # 2. 定位误差随时间变化
    plt.subplot(2, 2, 2)
    plt.plot(time, errors[:, 0], 'r-', label='经度误差(°)')
    plt.plot(time, errors[:, 1], 'g-', label='纬度误差(°)')
    plt.plot(time, errors[:, 2], 'b-', label='高度误差(km)')
    # 标记失锁时段
    unlock_times = time[unlock_flags]
    if len(unlock_times) > 0:
        plt.axvspan(unlock_times[0], unlock_times[-1], alpha=0.3, color='gray', label='失锁时段')
    plt.xlabel('时间(s)')
    plt.ylabel('误差值')
    plt.legend()
    plt.title('定位误差变化曲线')
    plt.grid(True)
    
    # 3. 载噪比变化曲线
    plt.subplot(2, 2, 3)
    plt.plot(time, cnr_list, 'orange', linewidth=2)
    plt.axhline(y=30, color='red', linestyle='--', label='干扰有效阈值(30 dB-Hz)')
    plt.xlabel('时间(s)')
    plt.ylabel('载噪比(dB-Hz)')
    plt.legend()
    plt.title('载噪比变化曲线')
    plt.grid(True)
    
    # 4. 误差统计直方图
    plt.subplot(2, 2, 4)
    plt.plot(time, np.sqrt(errors[:, 0]**2 + errors[:, 1]**2 + errors[:, 2]**2), 'r-')
    plt.xlabel('时间(s)')
    plt.ylabel('总定位误差(°)')
    plt.title('总定位误差变化曲线')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 运行仿真
if __name__ == "__main__":
    main_simulation()