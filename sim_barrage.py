import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# 设置中文字体与负号正常显示（Windows 常用字体：Microsoft YaHei / SimHei）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


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


# ========================= ECEF 转回 经纬度高度（单位: 度/度/千米）=========================
def ecef_to_lla(X, Y, Z):
    """
    将 ECEF（米）转换为经纬度高度：返回 (lon_deg, lat_deg, h_km)。
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


# ========================= 1. 核心参数配置（贴合仿真系统取值）=========================
class SimParams:
    def __init__(self):
        # 基础物理参数
        self.c = 3e8  # 光速(m/s)
        self.Td = 0.02  # 相关积分时间(s)
        self.fc = 1575.42e6  # GNSS中心频率(GPS L1频段, Hz)

        # 目标平台参数（巡航导弹示例）
        self.target_init_pos = np.array([119.045, 27.2233, 1.3])  # 初始经纬高(°/km)
        self.target_velocity = np.array([0.005, 0.003, 0])  # 速度(°/s, °/s, km/s)[0.0005, 0.0003, 0]
        self.sim_time = 400  # 仿真时长(s)200
        self.sampling_freq = 1  # 采样频率(Hz)
        
        # 干扰装备参数（保留单源参数作为默认，同时支持多个干扰源列表）
        self.jam_type = "multi-tone"  # 干扰样式：continuous_wave/multi-tone/bandlimited_gaussian/pseudocode/pulse
        self.jam_power = 50  # 单个干扰源发射功率(W)（默认）
        self.jam_bandwidth = 100e6  # 干扰带宽(Hz) 20e6
        self.pulse_width = 10e-6  # 脉冲宽度(s)，仅脉冲干扰用
        self.pulse_period = 1e-3  # 脉冲周期(s)，仅脉冲干扰用
        self.multi_tone_freqs = [1575.42e6 + 1e6, 1575.42e6 - 1e6]  # 多音干扰频点(Hz)
        self.jam_gaint = 1 # 天线增益

        # 支持多个干扰源：在预定轨迹中心附近布置（范围较小，默认半径约0.01°≈1.1km）
        # 生成策略：以轨迹中点为中心，等角度分布 5 个干扰源，参数可在 self.jammers 中查看/修改
        center = self.target_init_pos + self.target_velocity * (self.sim_time / 2)
        center_lon, center_lat, center_alt = center
        spread_deg = 0.07 # 半径（度），约1.11 km（范围不要太大）
        angles = np.linspace(0, 2 * np.pi, 4)[:-1]  # 3 等分角
        # powers = [50, 30, 20, 10, 5]
        powers = [10, 10, 10]
        # types = ['multi-tone', 'continuous_wave', 'pulsed', 'bandlimited_gaussian', 'pseudocode'] 噪声调频干扰
        types = ['continuous_wave', 'continuous_wave', 'continuous_wave']
        bandwidths = [20e6, 1e6, 1e6, 10e6, 5e6]
        self.jammers = []
        jammer = {'pos': np.array([120.0, 27.63, 8.3]), 'power': powers[0], 'type': types[0], 'bandwidth': bandwidths[0], 'freq': self.fc }
        self.jammers.append(jammer)

        # for i, ang in enumerate(angles):
        #     lon = center_lon + spread_deg * math.cos(ang)
        #     lat = center_lat + spread_deg * math.sin(ang)
        #     alt = center_alt + 7 # 使用与轨迹中心相近高度
        #     jammer = {'pos': np.array([lon, lat, alt]), 'power': powers[i], 'type': types[i], 'bandwidth': bandwidths[i], 'freq': self.fc }
        #     if types[i] == 'pulsed':
        #         jammer.update({'pulse_width': 5e-6, 'pulse_period': 1e-3, 'duty': 0.005})
        #     self.jammers.append(jammer)

        # # self.jammers[0]['pos'] = np.array([119.75, 27.64, 8.3])  # 微调第一个干扰源位置
        # self.jammers[0]['pos'] = np.array([120.25, 28.0, 8.3])
        # self.jammers[1]['pos'] = np.array([119.75, 27.64, 8.3])
        # self.jammers[2]['pos'] = np.array([120.0, 27.63, 8.3])
        
        # 导航装备参数
        self.combined_nav = "loose"  # 组合导航方式：loose/tight/deep（松/紧/深耦合）
        self.anti_jam_filter = "frequency"  # 抗干扰滤波：time/frequency/spatial（时/频/空域）
        self.pseudocode = "C/A"  # 伪码类型："C/A" 、"P(Y)"、"M"（GPS常用C/A码）
        
        self.Tc = 9.77e-7  # 伪码码元宽度(s)，C/A码典型值, P(Y)码 9.7752e-5 (s)，M码为1e-4 (s)
        self.fs = 10.23e6  # M码副载频(Hz)
        self.d = 1/8  # 码跟踪误差系数(1或1/8)
        self.beta = 2e6  # 接收机等效预相关带宽(Hz)，C/A 码典型带宽
        self.ins_drift = 0.01  # 惯导漂移率(km/s)，失锁时用
        
        # 卫星参数（模拟4颗GDOP最优卫星）
        self.satellite_pos = np.array([[125.0, 30.0, 5000],  # 卫星经纬高(°/km)
                                       [115.0, 35.0, 5000],
                                       [130.0, 25.0, 5000],
                                       [110.0, 28.0, 5000]])
        self.sat_carrier_power = 1e-16  # 卫星信号载波功率(C, W)
        
        # 跟踪环参数（与组合导航方式绑定）
        self.Bp = 18 if self.combined_nav in ["loose", "tight"] else 2  # PLL带宽(Hz)
        self.Bd = self.Bp  # DLL带宽(Hz)，与PLL一致
        
        # 失锁判定阈值
        self.pll_unlock_thresh = 15  # 载波环失锁阈值(°)
        self.dll_unlock_thresh = self.d / 6  # 码环失锁阈值

        # 标称载噪比
        self.C_N0_nom = 45  # dB-Hz
        
        self.Bn = 2.046e6   # 噪声带宽 (Hz, GPS L1 频段)
        # self.sigma_rho_nom = 0.2  # 标称伪距基线误差 (m)
        self.G_ant = 10     # 抗干扰波束成形增益 (dB)

        # 干扰参数
        #self.P_j = -100     # 干扰功率 (dBm)
        self.P_s = -130     # 卫星信号功率 (dBm)
        self.B_j = 20e6      # 干扰带宽 (Hz, 窄带干扰)


# -------------------------- 2.1 载噪比衰减与干扰后载噪比计算 --------------------------
def calc_cn0_jamming(params, Pj_list):
    """计算干扰后的载噪比 C/N0_j"""
    # 干扰功率扣除天线增益
    P_jr = Pj_list  # 接收机输入端干扰功率 (线性)
    # 线性功率换算
    P_s_lin = P_jr
    # P_s_lin = 10 ** (params.P_s / 10)
    P_jr_lin = 10 ** (P_jr / 10)
    # 载噪比衰减量 ΔC/N0
    delta_CN0 = 10 * math.log10(1 + (P_jr_lin * params.Bn) / (P_s_lin * params.B_j))
    # 干扰后载噪比
    C_NJ_dB = params.C_N0_nom - delta_CN0
    return C_NJ_dB

# ========================= 2.2 载噪比计算（C/NJ）=========================
def calc_cnr(P_j, params, cnt=0):
    """
    计算接收机前端载噪比C/NJ
    :param Pj_list: 干扰信号接收功率列表(W)
    :param params: 仿真参数类
    :return: C/NJ (dB-Hz)
    """
    integral_total = 0.0

    # 计算积分项：∫GJ(f)GS(f)df（积分范围：-β/2 到 β/2）
    def integrand(f):
        # 干扰频率偏移到基带：f_j = jam_freq - fc
        jam_freq = params.jammers[cnt].get('freq', params.fc)
        f_base = f + (jam_freq - params.fc)  # 干扰基带频率 = 积分变量 + 干扰频偏
        GJ, GS = calc_power_spectral_density(f_base, params, cnt)
        return GJ * GS
    #integrand = lambda f: calc_power_spectral_density(f, params, cnt)[0] * calc_power_spectral_density(f, params, cnt)[1]
    integral_result, _ = integrate.quad(integrand, params.fc - params.beta/2, params.fc + params.beta/2)
    # integral_result, _ = integrate.quad(integrand, - params.beta/2, params.beta/2)
    # 新增打印：验证积分结果是否为 0
    print(f"干扰源{cnt}：integral_result = {integral_result}, Pj = {P_j}")
    integral_total = P_j * integral_result

    # 打印总积分结果
    # print(f"integral_total = {integral_total}")
    # 载噪比计算（线性值）
    C_NJ_linear = params.sat_carrier_power / integral_total
    # 转换为dB-Hz
    C_NJ_dB = 10 * np.log10(C_NJ_linear) if C_NJ_linear > 0 else -100
    return C_NJ_dB

# ========================= 3. 功率谱密度计算（随干扰样式变化）=========================
def calc_power_spectral_density(f, params, cnt):
    """
    计算干扰信号功率谱密度GJ(f)和卫星信号功率谱密度GS(f)
    :param f: 频率点(Hz)
    :param params: 仿真参数类
    :return: GJ(f), GS(f)
    """
    # 卫星信号功率谱密度GS(f)（C/A码）
    if params.pseudocode == "C/A" or params.pseudocode == "P(Y)":
        GS = params.Tc * (np.sinc(np.pi * f * params.Tc)) ** 2
    elif params.pseudocode == "M":
        GS = params.Tc * (np.sinc(np.pi * f * params.Tc / 2)) ** 2 * (np.tan(np.pi * f / 2 * params.fs)) ** 2

    jam_type_ = params.jammers[cnt].get('type')
    jam_freq = params.jammers[cnt].get('freq')  # 优先使用干扰源自身的频率
    # 干扰频偏（基带）：f_j = jam_freq - fc
    # f_j = f - (jam_freq - params.fc)
    # 干扰信号功率谱密度GJ(f)（随干扰样式变化）
    if jam_type_ == "continuous_wave":
        # 单频干扰：δ(f - fj)
        # GJ = 1 if abs(f - jam_freq) < 1e3 else 0
        # 替换硬阈值为窄带高斯窗（带宽1kHz，模拟CW干扰的实际频谱）
        # sigma_cw = 1e3  # CW干扰的等效带宽
        # GJ = np.exp(-(f_j **2) / (2 * sigma_cw**2)) / (np.sqrt(2*np.pi) * sigma_cw)
        GJ = 1
    elif jam_type_ == "multi-tone":
        # 多音干扰：Σδ(f - fn)
        GJ = 1 if any(abs(f - freq) < 1e3 for freq in params.multi_tone_freqs) else 0
    elif jam_type_ == "bandlimited_gaussian":
        # 带限高斯干扰：1/β
        GJ = 1 / params.jam_bandwidth if abs(f - jam_freq) < params.jam_bandwidth/2 else 0
    elif jam_type_ == "pseudocode":
        # 伪码干扰：使用伪随机码
        GJ = 1 if abs(f - jam_freq) < params.jam_bandwidth/2 else 0
    elif jam_type_ == "pulse":
        # 脉冲干扰：|(τ/T)ΣSa[(f-fj)πτ]δ(f-fj + n/T)|²
        tau = params.pulse_width
        T = params.pulse_period
        fj = jam_freq
        n = round((f - fj) * T)
        sa_term = np.sinc((f - fj) * tau)
        GJ = (tau / T) * (sa_term ** 2) if abs(f - fj) < params.jam_bandwidth/2 else 0
    else:
        GJ = 0
    
    return GJ, GS

# ========================= 4. APM电磁传播模型（计算干扰信号功率）=========================
def calc_jam_power_apm(jammer, target_pos, params):
    """
    基于APM模型计算来自单个干扰源到达接收机的干扰信号功率P_J
    :param jammer: 干扰源 dict，包含 'pos' (经纬高), 'power' (W), 可选其他字段
    :param target_pos: 目标位置(经纬高)
    :param params: 仿真参数类
    :return: 干扰信号接收功率P_J(W)
    """
    jam_pos = jammer['pos']
    jam_power = jammer.get('power')

    jam_power = 10**(jam_power/10) if jam_power < 100 else jam_power  # 如果输入为dBm则转换为W
    # 转换为距离（经纬度转ecef坐标）
    jam_ecef = lla_to_ecef(jam_pos[0], jam_pos[1], jam_pos[2])
    target_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])
    dist = np.linalg.norm(jam_ecef - target_ecef)  # 总距离(米)

    dist_vertical = abs((jam_pos[2] - target_pos[2]) * 1000)  # 垂直距离(米)
    dist_horizontal = np.sqrt(dist**2 - dist_vertical**2)  # 水平距离(米)

    # APM模型选择（根据距离和仰角）
    antenna_elevation = np.arctan2(dist_vertical, dist_horizontal) * 180 / np.pi  # 天线仰角(°)

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
    print(f"干扰源距离：{dist/1000} km，路径损耗：{loss}，接收功率Pj：{Pj} W")
    return Pj

# ========================= 5. 干信比（J/S）计算 =========================
def calc_jammer_to_signal_ratio(target_pos, params, Pj, cnt = 0):
    """
    计算干信比 J/S（干扰功率/卫星信号功率）
    :param jammer: 干扰源 dict，包含 'pos' (经纬高), 'power' (W)
    :param target_pos: 目标位置(经纬高)
    :param params: 仿真参数类
    :return: J/S（线性值）、J/S_dB（dB值）
    """
    # 1. 计算干扰信号到达目标的功率 Pj (W)
    #Pj = calc_jam_power_apm(jammer, target_pos, params)
    
    # 2. 计算卫星信号到达目标的功率 Ps (W)
    # 卫星信号功率需结合传播模型（简化为自由空间传播）
    Ps_total = 0.0
    for sat_pos in params.satellite_pos:
        # 卫星位置转ECEF，计算与目标的距离
        sat_ecef = lla_to_ecef(sat_pos[0], sat_pos[1], sat_pos[2])
        target_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])
        dist_sat_target = np.linalg.norm(sat_ecef - target_ecef)  # 星地距离(米)
        
        # 自由空间损耗
        loss_sat = (4 * np.pi * dist_sat_target * params.fc / params.c) ** 2
        # 卫星发射功率（典型GPS卫星EIRP约48.5 dBW，转换为线性值：10^(48.5/10)=70794.58 W）
        sat_eirp = 70794.58 
        # sat_eirp = 48.5
        Ps_sat = sat_eirp * 100 / loss_sat  # 单颗卫星信号接收功率
        Ps_total += Ps_sat  # 多颗卫星信号功率叠加
    
    # 3. 计算干信比
    if Ps_total < 1e-20:  # 避免除零
        J_S = 1e20
        J_S_dB = 200
    else:
        J_S = Pj / Ps_total
        J_S_dB = 10 * np.log10(J_S)  # 转换为dB值

    print(f"线性干信比：{J_S}，干信比(dB)：{J_S_dB}")
    
    return J_S, J_S_dB

# ========================= 6. 跟踪环误差计算（PLL+DLL）=========================
def calc_tracking_errors(C_NJ_dB, J_S, params, cnt):
    """
    计算载波环振荡器颤动σ_JPLL和码环跟踪误差σ_JDLL(NELP)
    :param C_NJ_dB: 载噪比(dB-Hz)
    :param J_S: 干信比 J/S
    :param params: 仿真参数类
    :return: σ_JPLL(°), σ_JDLL
    """
    # 转换载噪比为线性值
    C_NJ_linear = 10 ** (C_NJ_dB / 10)
    print(f"线性载噪比：{C_NJ_linear}")
    
    # 1. 载波环振荡器颤动σ_JPLL
    term = (params.Bp / C_NJ_linear) * (1 + 1 / (2 * params.Td * C_NJ_linear))
    sigma_jpll = (360 / (2 * np.pi)) * np.sqrt(term)
    sigma_jpll = sigma_jpll % 360  # 限制在0-360度范围内

    # 2. 码环跟踪误差σ_JDLL(NELP)
    # 计算积分项
    """积分项1：∫G_J(f)G_S(f)·sin²(πf d Tc) df"""
    def integrand_sin2(f):
        GJ, GS = calc_power_spectral_density(f, params, cnt)
        return GJ * GS * (np.sin(np.pi * f * params.d * params.Tc)) ** 2
    
    """积分项2：∫f·G_S(f)·sin(πf d Tc) df"""
    def integrand_fsin(f):
        GJ, GS = calc_power_spectral_density(f, params, cnt)
        return f * GS * np.sin(np.pi * f * params.d * params.Tc)
    
    """积分项3：∫G_J(f)G_S(f)·cos²(πf d Tc) df"""
    def integrand_cos2(f):
        GJ, GS = calc_power_spectral_density(f, params, cnt)
        return GJ * GS * (np.cos(np.pi * f * params.d * params.Tc)) ** 2
    
    """积分项4：∫G_S(f)·cos(πf d Tc) df"""
    def integrand_cos(f):
        GJ, GS = calc_power_spectral_density(f, params, cnt)
        return GS * np.cos(np.pi * f * params.d * params.Tc)
    
    integral_sin2, _ = integrate.quad(integrand_sin2, params.fc - params.beta/2, params.fc + params.beta/2)
    integral_fsin, _ = integrate.quad(integrand_fsin, params.fc - params.beta/2, params.fc + params.beta/2)
    integral_cos2, _ = integrate.quad(integrand_cos2, params.fc - params.beta/2, params.fc + params.beta/2)
    integral_cos, _ = integrate.quad(integrand_cos, params.fc - params.beta/2, params.fc + params.beta/2)
    
    # 码环误差公式（简化参数，Pj已包含在C/NJ中）
    numerator = np.sqrt(params.Bd * J_S * integral_sin2) 
    denominator = 2 * np.pi * integral_fsin
    term2 = (J_S * integral_cos2) / (params.Td * (integral_cos ** 2))
    sigma_jdll = (numerator / denominator) * np.sqrt(1 + term2) if denominator != 0 else 0

    return sigma_jpll, sigma_jdll

# ========================= 7. GDOP计算（几何精度因子）=========================
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
        sat_ecef = lla_to_ecef(sat[0], sat[1], sat[2])
        target_pos_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])
        delta = sat_ecef - target_pos_ecef
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

# ========================= 8. 结果可视化（轨迹+误差+载噪比）=========================
def visualize_results(true_traj, error_traj, errors_m, cnr_list, unlock_flags, c_nj_flags, js_ratio_list, sigma_jdll_list, params):
    time = np.arange(len(errors_m)) / params.sampling_freq
    plt.figure(figsize=(15, 10))
    
    # 1. 2D轨迹对比（经纬度平面）
    plt.subplot(3, 2, 1)
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'g-', label='预定轨迹', linewidth=2)
    plt.plot(error_traj[:, 0], error_traj[:, 1], 'r-', label='受干扰轨迹', linewidth=2)
    # 绘制所有干扰源
    for j in params.jammers:
        plt.scatter(j['pos'][0], j['pos'][1], c='blue', marker='x', s=100, label=f"干扰源 ({j.get('type','')})")
    # 单个干扰源绘制
    # plt.scatter(params.jammers[0]['pos'][0], params.jammers[0]['pos'][1], c='blue', marker='x', s=100, label=f"干扰源 ({params.jammers[0].get('type','')})")
    # 打印干扰源位置
    print(f"干扰源{1}：位置 = {params.jammers[0]['pos'][0]}, {params.jammers[0]['pos'][1]}, {params.jammers[0]['pos'][2]}")
    
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
    plt.subplot(3, 2, 2)
    plt.plot(time, errors_m[:, 0], 'r-', label='经度误差(m)')
    plt.plot(time, errors_m[:, 1], 'g-', label='纬度误差(m)')
    plt.plot(time, errors_m[:, 2], 'b-', label='高度误差(m)')
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
    plt.subplot(3, 2, 3)
    plt.plot(time, cnr_list, 'orange', linewidth=2)
    plt.axhline(y=40, color='blue', linestyle='--', label='干扰有效阈值(30 dB-Hz)')
    plt.plot(time, c_nj_flags*10, 'r-', label='载噪比有效时段')
    plt.xlabel('时间(s)')
    plt.ylabel('载噪比(dB-Hz)')
    plt.legend()
    plt.title('载噪比变化曲线')
    plt.grid(True)

    # 4. 干信比变化曲线（新增）
    plt.subplot(3, 2, 4)
    plt.plot(time, js_ratio_list, 'purple', linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--', label='J/S=0dB（干信等功率）')
    plt.axhline(y=10, color='green', linestyle='--', label='J/S=10dB（干扰占优）')
    plt.xlabel('时间(s)')
    plt.ylabel('干信比(J/S) (dB)')
    plt.legend()
    plt.title('干信比变化曲线')
    plt.grid(True)

    # 5. 码环误差变化曲线（新增）
    plt.subplot(3, 2, 5)
    plt.plot(time, sigma_jdll_list, 'brown', linewidth=2)
    plt.xlabel('时间(s)')
    plt.ylabel('码环误差(°)')
    plt.title('码环误差变化曲线')
    plt.grid(True)

    # 6. 误差统计直方图
    plt.subplot(3, 2, 6)
    plt.plot(time, np.sqrt(errors_m[:, 0]**2 + errors_m[:, 1]**2 + errors_m[:, 2]**2), 'r-')
    plt.xlabel('时间(s)')
    plt.ylabel('总定位误差(m)')
    plt.title('总定位误差变化曲线')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# ========================= 9. 主仿真流程=========================
def main_simulation():
    # 初始化参数
    params = SimParams()
    time_steps = int(params.sim_time * params.sampling_freq)
    target_positions = []  # 目标真实轨迹
    error_positions = []   # 受干扰后轨迹
    errors = []            # 定位误差序列
    errors_m = []          # 定位误差序列（米）
    cnr_list = []          # 载噪比序列
    unlock_flags = []      # 失锁标志序列
    unlock_ts = []         # 失锁标志序列
    c_nj_flags = []        # 载噪比有效标志序列
    js_ratio_list = []     # 干信比序列
    sigma_jdll_list = []   # 码环误差序列
    unlock_st_end = []    # 失锁时间段记录
    
    # 仿真主循环
    for t in range(time_steps):
        errors_total = [0.0, 0.0, 0.0]            # 所有干扰源定位误差
        errors_m_total = [0.0, 0.0, 0.0]          # 所有干扰源定位误差（米）
        lock_errors_total = [0.0, 0.0, 0.0]            # 所有未失锁干扰源定位误差
        lock_errors_m_total = [0.0, 0.0, 0.0]          # 所有未失锁干扰源定位误差（米）
        sigma_jdll_total = 0.0
        unlock_flag_final = False
        c_nj_flag_final = False
        C_NJ_dB_total = 0.0
        J_S_dB_total = 0.0
        lock_num = 0  # 未失锁的干扰源个数
        unlock_num = 0  # 失锁的干扰源个数

        # 1. 计算目标实时位置
        target_pos = params.target_init_pos + params.target_velocity * t
        target_positions.append(target_pos)

        cnt = 0
        # jammer = params.jammers[0]  # 目前仅支持单个干扰源计算

        # 分别计算每个干扰源的功率和影响
        for jammer in params.jammers:
            # unlock_pos_error = np.array([0.0, 0.0, 0.0])
            # unlock_pos_error_m = np.array([0.0, 0.0, 0.0])
            lock_pos_error = np.array([0.0, 0.0, 0.0])
            lock_pos_error_m = np.array([0.0, 0.0, 0.0])
            invalid_pos_error = np.array([0.0, 0.0, 0.0])
            invalid_pos_error_m = np.array([0.0, 0.0, 0.0])
            sigma_jdll = 0.0
            unlock_flag = False

            # 2. 计算干扰信号接收功率
            Pj = calc_jam_power_apm(jammer, target_pos, params)

            # 3. 计算载噪比
            C_NJ_dB = calc_cnr(Pj, params)
            # C_NJ_dB = calc_cn0_jamming(params, Pj)
            print(f"C_NJ_dB = {C_NJ_dB}")

            # 计算干信比 J/S
            J_S, J_S_dB = calc_jammer_to_signal_ratio(target_pos, params, Pj, cnt)

            c_nj_flag = False
            if C_NJ_dB < 80:  # 干扰有效（载噪比低于正常阈值） 原来是30
                # C_NJ_dB_ = 20
                # 标记干扰有效
                c_nj_flag = True

                # 4. 计算跟踪环误差
                sigma_jpll, sigma_jdll = calc_tracking_errors(C_NJ_dB, J_S, params, cnt)
                print(f"sigma_jpll_ = {sigma_jpll}, sigma_jdll_ = {sigma_jdll}")
                
                # 5. 判定失锁状态
                pll_unlock = sigma_jpll > params.pll_unlock_thresh
                dll_unlock = sigma_jdll > params.dll_unlock_thresh
                unlock_flag = pll_unlock or dll_unlock
                
                # 6. 求解定位误差（使用所有干扰源的总接收功率）
                if unlock_flag:
                    unlock_num += 1
                    # 场景1：失锁，使用惯导定位，误差随时间累积
                    # unlock_pos_error = np.array([params.ins_drift/111, (params.ins_drift/111), 0.1])  # 简化累积偏差
                    # pos_error = ecef_to_lla(params.ins_drift, params.ins_drift, 100)
                    # unlock_pos_error_m = np.array([params.ins_drift * 1000, params.ins_drift * 1000, params.ins_drift * 1000])
                else:
                    lock_num += 1
                    # 场景2：未失锁，伪距误差×GDOP
                    pseudo_range_error = sigma_jdll * params.Tc * params.c
                    # pseudo_range_error = sigma_jdll * params.c  # 码跟踪误差转换为伪距误差(米)
                    gdop = calc_gdop(target_pos, params.satellite_pos)
                    # 转换为经纬高误差（简化：米→度，1度≈111km）
                    # pos_error = (pseudo_range_error / 1000 / 111) * gdop  # 经度/纬度误差(°)
                    pos_error_single = (pseudo_range_error) * gdop  # 经度/纬度误差(m)
                    lock_pos_error = np.array([pos_error_single/(1000 * 111), pos_error_single/(1000 * 111), pos_error_single / 1000])  # 高度误差(km)
                    lock_pos_error_m = np.array([pos_error_single, pos_error_single, pos_error_single])  # 高度误差(m)
            else:
                # 干扰无效，定位误差为正常GNSS误差
                # pos_error = np.array([0.0001, 0.0001, 0.001])
                invalid_pos_error = np.array([0.0, 0.0, 0.0])
                invalid_pos_error_m = np.array([0.0, 0.0, 0.0])
                sigma_jdll = 0.0

            unlock_flag_final = unlock_flag_final or unlock_flag
            c_nj_flag_final = c_nj_flag_final or c_nj_flag

            # 累加所有干扰源的误差贡献
            lock_errors_total += lock_pos_error
            lock_errors_m_total += lock_pos_error_m
            lock_pos_error = np.array([0.0, 0.0, 0.0])
            lock_pos_error_m = np.array([0.0, 0.0, 0.0])
            sigma_jdll_total += sigma_jdll
            C_NJ_dB_total += C_NJ_dB  # 简单累加（可改为更复杂的合成方式）
            J_S_dB_total += J_S_dB
            # 干扰源序号递增
            cnt += 1
        
        # 7. 综合所有干扰源的定位误差
        # 存在失锁干扰源
        if unlock_num > 0:
            # 场景1：失锁，使用惯导定位，误差随时间累积
            unlock_pos_error = np.array([params.ins_drift/111, (params.ins_drift/111), 0.1])  # 简化累积偏差
            # pos_error = ecef_to_lla(params.ins_drift, params.ins_drift, 100)
            unlock_pos_error_m = np.array([params.ins_drift * 1000, params.ins_drift * 1000, params.ins_drift * 1000])
            errors_total = unlock_pos_error
            errors_m_total = unlock_pos_error_m
        else:
            # 不存在未失锁干扰源
            if lock_num > 0:
                errors_total = lock_errors_total
                errors_m_total = lock_errors_m_total

        errors.append(errors_total)
        errors_m.append(errors_m_total)
        error_positions.append(target_pos + errors_total)
        sigma_jdll_list.append(sigma_jdll_total/len(params.jammers))  # 平均码跟踪误差
        unlock_flags.append(unlock_flag_final)
        if unlock_flag_final:
            unlock_ts.append(unlock_flag_final)
        cnr_list.append(C_NJ_dB_total/len(params.jammers))  # 平均载噪比
        c_nj_flags.append(c_nj_flag_final)
        js_ratio_list.append(J_S_dB_total/len(params.jammers))  # 平均干信比
        

    # 转换为numpy数组
    target_positions = np.array(target_positions)
    error_positions = np.array(error_positions)
    errors = np.array(errors)
    errors_m = np.array(errors_m)
    cnr_list = np.array(cnr_list)
    unlock_flags = np.array(unlock_flags)
    c_nj_flags = np.array(c_nj_flags)
    

    print(f"仿真结束：总时刻={params.sim_time}, 失锁时刻={np.sum(unlock_ts)}")

    # 7. 结果可视化
    visualize_results(target_positions, error_positions, errors_m,cnr_list, unlock_flags, c_nj_flags, js_ratio_list, sigma_jdll_list,params)

# 运行仿真
if __name__ == "__main__":
    main_simulation()