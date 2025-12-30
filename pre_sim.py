import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 设置中文字体与负号正常显示（Windows 常用字体：Microsoft YaHei / SimHei）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 1. 核心参数配置（完全匹配文章场景）=========================
class SimParams:
    def __init__(self):
        # 基础物理参数
        self.c = 3e8  # 光速(m/s)
        self.sat_elevation_thresh = 15  # 卫星高度角阈值(°)
        
        # 卫星参数（文章示例：4星+5星场景）
        self.satellites_4 = np.array([[10000, 10000, 0],    # S1(1万km,1万km,0)
                                      [10000, -10000, 0],   # S2(1万km,-1万km,0)
                                      [-10000, -10000, 0],  # S3(-1万km,-1万km,0)
                                      [-10000, 10000, 0]])  # S4(-1万km,1万km,0)
        self.satellites_5 = np.vstack([self.satellites_4, [0, 0, 0]])  # 新增S5(0,0,0)
        
        # 接收机参数
        self.receiver_area = np.linspace(-200, 200, 50)  # 接收机工作区域X/Y∈[-200km,200km]
        self.b_k = 0  # 接收机钟差等效距离(文章忽略，设为0)
        self.delta_s = 20e-9  # 卫星钟差(20ns)
        
        # 干扰参数（伴星+转发延时）
        self.companion_sat = np.array([0, 0, 300])  # 伴星位置(0,0,300km)
        self.alpha_tau = 1e-6  # 人为转发延时(1μs，文章示例)
        self.sigma_alpha_tau = 0.1e-6  # 延时标准差(服从零均值正态分布)
        
        # 误差参数（文章忽略次要误差，设为0）
        self.ion_delay = 0  # 电离层延迟
        self.tropo_delay = 0  # 对流层延迟
        self.v_kj = 0  # 观测随机误差

# ========================= 2. 基础伪距计算（无干扰）=========================
def calc_pseudorange_no_jam(receiver_pos, satellite_pos, params):
    """
    计算无干扰时的伪距（文章公式3）
    :param receiver_pos: 接收机位置(km)
    :param satellite_pos: 卫星位置(km)
    :param params: 仿真参数类
    :return: 伪距ρ(km)
    """
    # 转换为米计算几何距离R
    receiver_pos_m = receiver_pos * 1000
    satellite_pos_m = satellite_pos * 1000
    R_m = np.linalg.norm(satellite_pos_m - receiver_pos_m)  # 几何距离(m)
    R_km = R_m / 1000  # 转换为km
    
    # 伪距公式：ρ = R + b_k - c·δ^s（单位统一为km）
    c_delta_s_km = (params.c * params.delta_s) / 1000  # c·δ^s转换为km
    rho = R_km + params.b_k - c_delta_s_km
    return rho

# ========================= 3. 转发式欺骗干扰伪距计算（含延时）=========================
def calc_pseudorange_jam(receiver_pos, satellite_pos, companion_sat, params, is_interfere_sat=True):
    """
    计算受干扰后的伪距（文章公式5、7）
    :param is_interfere_sat: 是否对该卫星施加干扰
    :return: 干扰后伪距ρ_jam(km)
    """
    # 无干扰伪距
    rho_no_jam = calc_pseudorange_no_jam(receiver_pos, satellite_pos, params)
    
    if not is_interfere_sat:
        return rho_no_jam
    
    # 计算总延时Δτ_total = 伴星位置延时Δτ_r + 人为延时Δτ（文章简化模型）
    # 1. 伴星位置延时Δτ_r：卫星→伴星→接收机的额外路径延时
    sat2comp_m = np.linalg.norm((satellite_pos - companion_sat) * 1000)  # 卫星到伴星距离(m)
    comp2recv_m = np.linalg.norm((companion_sat - receiver_pos) * 1000)  # 伴星到接收机距离(m)
    delta_tau_r = (sat2comp_m + comp2recv_m - np.linalg.norm((satellite_pos - receiver_pos) * 1000)) / params.c
    
    # 2. 人为延时Δτ（服从零均值正态分布）
    delta_tau = np.random.normal(loc=params.alpha_tau, scale=params.sigma_alpha_tau)
    
    # 3. 总延时贡献：c·Δτ_total（转换为km）
    c_tau_total_km = (params.c * (delta_tau_r + delta_tau)) / 1000
    
    # 干扰后伪距（叠加延时贡献，忽略伴星钟差Δτ_c）
    rho_jam = rho_no_jam + c_tau_total_km + params.ion_delay + params.tropo_delay + params.v_kj
    return rho_jam

# ========================= 4. 几何矩阵H与定位误差解算（文章公式8-13）=========================
def solve_position_error(receiver_true_pos, satellites, companion_sat, params, interfere_sat_idx=0):
    """
    解算定位误差（最小二乘法）
    :param interfere_sat_idx: 被干扰卫星的索引（0-based）
    :return: 定位误差标准差σ_R(km)（水平方向：√(σ_X²+σ_Y²)）
    """
    N = len(satellites)  # 参与定位的卫星数
    if N < 4:
        return np.inf  # 卫星数不足，返回无穷大误差
    
    # 1. 计算各卫星的干扰后伪距dP（伪距误差=干扰伪距-无干扰伪距）
    dP = []
    R_km_list = []  # 卫星到接收机真实距离(km)
    for i, sat in enumerate(satellites):
        rho_no_jam = calc_pseudorange_no_jam(receiver_true_pos, sat, params)
        rho_jam = calc_pseudorange_jam(receiver_true_pos, sat, companion_sat, params, is_interfere_sat=(i==interfere_sat_idx))
        dP.append(rho_jam - rho_no_jam)
        R_km = np.linalg.norm(sat - receiver_true_pos)
        R_km_list.append(R_km)
    dP = np.array(dP).reshape(-1, 1)
    
    # 2. 构建几何矩阵H（文章公式9）
    H = []
    for i, sat in enumerate(satellites):
        delta_x = receiver_true_pos[0] - sat[0]
        delta_y = receiver_true_pos[1] - sat[1]
        delta_z = receiver_true_pos[2] - sat[2]
        R_km = R_km_list[i]
        if R_km == 0:
            R_km = 1e-6
        H.append([delta_x/R_km, delta_y/R_km, delta_z/R_km])
    H = np.array(H)
    
    # 3. 最小二乘解算定位误差dR（文章公式11）
    try:
        HTH_inv = np.linalg.inv(H.T @ H)
        dR_hat = HTH_inv @ H.T @ dP
    except np.linalg.LinAlgError:
        return np.inf  # 矩阵奇异，返回无穷大误差
    
    # 4. 计算协方差阵与水平定位误差（文章公式13、17）
    sigma_p_sq = (params.c * params.sigma_alpha_tau) ** 2 / (1000 ** 2)  # 伪距误差方差(km²)
    cov_dR = HTH_inv * sigma_p_sq  # 定位误差协方差阵
    sigma_R = np.sqrt(cov_dR[0,0] + cov_dR[1,1])  # 水平定位误差标准差(km)
    
    return sigma_R

# ========================= 5. 模拟试验（4星/5星干扰场景，文章图2、3）=========================
def simulation_test(params):
    """
    遍历接收机工作区域，计算每个点位的定位误差
    """
    # 场景1：4星定位，干扰S1（索引0）
    error_4sat = np.zeros((len(params.receiver_area), len(params.receiver_area)))
    # 场景2：5星定位，干扰S5（索引4）
    error_5sat = np.zeros((len(params.receiver_area), len(params.receiver_area)))
    
    for i, x in enumerate(params.receiver_area):
        for j, y in enumerate(params.receiver_area):
            receiver_pos = np.array([x, y, 0])  # 接收机Z坐标设为0
        
            # 4星场景误差
            sigma_R_4 = solve_position_error(receiver_pos, params.satellites_4, params.companion_sat, params, interfere_sat_idx=0)
            error_4sat[i, j] = sigma_R_4 if sigma_R_4 < 500 else 500  # 限制最大误差便于可视化
        
            # 5星场景误差
            sigma_R_5 = solve_position_error(receiver_pos, params.satellites_5, params.companion_sat, params, interfere_sat_idx=4)
            error_5sat[i, j] = sigma_R_5 if sigma_R_5 < 1 else 1  # 5星误差较小，缩放显示
    
    return error_4sat, error_5sat

# ========================= 6. 结果可视化（复刻文章图2、3：定位误差等值线图）=========================
def visualize_results(params, error_4sat, error_5sat):
    X, Y = np.meshgrid(params.receiver_area, params.receiver_area)
    
    plt.figure(figsize=(12, 5))
    
    # 子图1：4星定位误差等值线图
    plt.subplot(1, 2, 1)
    contour1 = plt.contourf(X, Y, error_4sat, cmap=cm.jet, levels=20)
    plt.colorbar(contour1, label='定位误差标准差(km)')
    plt.scatter(params.companion_sat[0], params.companion_sat[1], c='white', marker='*', s=200, label='伴星')
    plt.xlabel('X坐标(km)')
    plt.ylabel('Y坐标(km)')
    plt.title('4星定位+干扰S1 定位误差分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：5星定位误差等值线图
    plt.subplot(1, 2, 2)
    contour2 = plt.contourf(X, Y, error_5sat, cmap=cm.jet, levels=20)
    plt.colorbar(contour2, label='定位误差标准差(km)')
    plt.scatter(params.companion_sat[0], params.companion_sat[1], c='white', marker='*', s=200, label='伴星')
    plt.xlabel('X坐标(km)')
    plt.ylabel('Y坐标(km)')
    plt.title('5星定位+干扰S5 定位误差分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 输出误差统计（匹配文章表1）
    print("4星定位干扰S1：")
    print(f"误差最小值：{np.min(error_4sat):.1f} km，误差最大值：{np.max(error_4sat):.1f} km")
    print("5星定位干扰S5：")
    print(f"误差最小值：{np.min(error_5sat)*1000:.1f} m，误差最大值：{np.max(error_5sat)*1000:.1f} m")

# ========================= 7. 主函数运行 =========================
if __name__ == "__main__":
    # 初始化参数
    params = SimParams()
    
    # 执行模拟试验
    error_4sat, error_5sat = simulation_test(params)
    
    # 可视化结果
    visualize_results(params, error_4sat, error_5sat)