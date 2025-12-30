import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体与负号正常显示（Windows 常用字体：Microsoft YaHei / SimHei）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 1. 核心参数配置（参考仿真系统取值）=========================
class SimParams:
    def __init__(self):
        # 基础物理参数
        self.c = 3e8  # 光速(m/s)
        self.GDOP_threshold = 3  # 最优GDOP阈值
        self.power_ratio_threshold = 5  # 欺骗信号功率比阈值(dB)
        
        # 干扰装备参数
        self.jammer_num = 4  # 转发器数量（固定4套）
        self.jammer_pos = np.array([[115.2, 29.0, 1.0],  # 转发器经纬高(°/km)
                                    [115.3, 29.1, 1.0],
                                    [115.1, 29.2, 1.0],
                                    [115.4, 29.0, 1.0]])
        # self.jammer_pos = np.array([[115.10, 28.90, 0.8],
        #                             [115.12, 29.12, 1.2],
        #                             [115.15, 29.18, 1.0],
        #                             [115.13, 29.02, 0.9]])
        #self.jammer_pos_ecef = np.array([lla_to_ecef(sat[0], sat[1], sat[2]) for sat in self.jammer_pos])
        
        # 目标平台参数（巡航导弹示例）
        self.target_init_pos = np.array([115.193, 29.027, 0.5])  # 初始经纬高(°/km)
        #self.target_init_pos_ecef = np.array([lla_to_ecef(self.target_init_pos[0], self.target_init_pos[1], self.target_init_pos[2])])
        self.target_velocity = np.array([0.001, 0.0005, 0])  # 速度(°/s, °/s, km/s)
        #self.target_velocity_ecef = np.array([lla_to_ecef(self.target_velocity[0], self.target_velocity[1], self.target_velocity[2])])
        self.sim_time = 100  # 仿真时长(s)
        self.sampling_freq = 1  # 采样频率(Hz)
        
        # 欺骗点序列（引导路径，从起始欺骗点到最终欺骗点）
        # 格式为 [[lon, lat, h_km], ...]，仿真中会沿该路径平滑移动以引导目标到最终欺骗点
        # self.deception_points = np.array([
        #     [115.193, 29.027, 0.5],
        #     [115.24, 29.08, 0.5],
        #     [115.28, 29.12, 0.5],
        #     [115.32, 29.15, 0.5]
        # ])

        # 兼容字段：最终欺骗点（序列的最后一项）
        self.deception_pos = [115.32, 29.15, 0.5]
        #self.deception_pos_ecef = np.array([lla_to_ecef(pt[0], pt[1], pt[2]) for pt in self.deception_points])
        
        # 卫星参数（模拟10颗可供筛选的卫星，实际可通过星历计算）
        # 形式： [longitude(°), latitude(°), height(km)]
        self.satellite_pos = np.array([
            [120.0, 30.0, 20000],  # 卫星经纬高(经度, 纬度, 高度(km))
            [110.0, 35.0, 20000],
            [130.0, 25.0, 20000],
            [105.0, 28.0, 20000],
            [140.0, 32.0, 20000],
            [100.0, 22.0, 20000],
            [125.0, 40.0, 20000],
            [115.0, 20.0, 20000],
            [135.0, 27.0, 20000],
            [95.0, 33.0, 20000]
        ])
        # 将卫星经纬高转换为 ECEF（米），并保存为新属性 satellite_pos_ecef
        # 注意：lla_to_ecef(lon_deg，lat_deg, h_km) 的参数顺序为 (经度, 纬度, 高度_km)
        # self.satellite_pos_ecef = np.array([lla_to_ecef(sat[0], sat[1], sat[2]) for sat in self.satellite_pos])

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


def interpolate_deception_points(deception_points, t, total_steps):
    """
    沿着 deception_points 序列做分段线性插值，根据当前时间步 t 返回瞬时欺骗点坐标。
    t: 当前时刻索引（0..total_steps-1）
    total_steps: 总时间步数
    返回一个长度为3的数组 [lon, lat, h_km]
    """
    pts = np.asarray(deception_points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("deception_points 必须为 (N,3) 数组")
    n = pts.shape[0]
    if n == 1:
        return pts[0]
    s = t / max(1, total_steps - 1)
    seg_len = 1.0 / (n - 1)
    idx = min(int(s / seg_len), n - 2)
    local_s = (s - idx*seg_len) / seg_len
    return (1 - local_s) * pts[idx] + local_s * pts[idx+1]


# ========================= 2. 卫星筛选（GDOP最优组合，ECEF 计算）=========================
def select_optimal_satellites(target_pos, satellite_list, params):
    """
    从卫星列表中筛选GDOP最优的4颗卫星。
    输入：target_pos 与 satellite_list 均为经纬高格式 (lon, lat, h_km)。
    在内部将坐标转换为 ECEF（米）进行几何矩阵与 GDOP 计算，最后返回原始经纬高格式的最优卫星集合（shape (4,3)）。
    """
    # 先把目标与所有卫星都转换为 ECEF（米）以获得正确的几何关系
    target_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])
    sat_ecef_list = np.array([lla_to_ecef(sat[0], sat[1], sat[2]) for sat in satellite_list])

    min_gdop = float('inf')
    best_idx = None

    # 遍历所有4颗卫星的索引组合（实际可优化为贪心算法）
    from itertools import combinations
    for idx_comb in combinations(range(len(satellite_list)), 4):
        G = []
        valid = True
        for i in idx_comb:
            sat_ecef = sat_ecef_list[i]
            # 强制为一维数组并检查长度
            sat_ecef = np.asarray(sat_ecef).reshape(-1)
            tgt = np.asarray(target_ecef).reshape(-1)
            if sat_ecef.size != 3 or tgt.size != 3:
                valid = False
                print(f"Warning: invalid satellite/target vector shape: sat.size={sat_ecef.size}, tgt.size={tgt.size}; skipping combo {idx_comb}")
                break

            delta = sat_ecef - tgt  # 单位：米
            dist = np.linalg.norm(delta)
            if dist == 0 or np.isnan(dist):
                valid = False
                break
            unit_vec = delta / dist
            if unit_vec.size < 3:
                valid = False
                print(f"Warning: computed unit vector has size {unit_vec.size}; skipping combo {idx_comb}")
                break
            G.append([unit_vec[0], unit_vec[1], unit_vec[2], 1])
        if not valid:
            continue
        G = np.array(G)

        # 稳健计算逆矩阵：先检查条件数，必要时使用伪逆
        A = G.T @ G
        try:
            cond = np.linalg.cond(A)
            if cond < 1e12:
                A_inv = np.linalg.inv(A)
            else:
                A_inv = np.linalg.pinv(A)
            gdop = np.sqrt(np.trace(A_inv))
            # 记录最小 GDOP（不再强制与阈值比较；阈值可用于后续判定）
            if gdop < min_gdop:
                min_gdop = gdop
                best_idx = idx_comb
        except np.linalg.LinAlgError:
            continue

    # 若没有找到可用组合则退回默认前 4 个卫星
    if best_idx is None:
        return satellite_list[:4]

    # 返回原始经纬高格式的最优卫星集合
    optimal_sats = np.array([satellite_list[i] for i in best_idx])

    return optimal_sats

# ========================= 3. 欺骗信号有效性判定（功率比）=========================
def is_deception_valid(jammer_pos, target_pos, satellite_pos, params):
    """
    判断目标是否捕获欺骗信号（功率比>5dB）
    :return: bool: 欺骗信号是否有效
    """
    # 简化功率计算（实际需结合APM电磁传播模型计算路径损耗）当前freq 北斗B1为载波频率，实际以卫星信号实际载波设置
    def calc_signal_power(trans_pos, recv_pos, freq=1561.09e6):
        trans_pos_ecef = lla_to_ecef(trans_pos[0], trans_pos[1], trans_pos[2])
        recv_pos_ecef = lla_to_ecef(recv_pos[0], recv_pos[1], recv_pos[2])
        """计算信号接收功率（自由空间传播模型简化）"""
        dist = np.linalg.norm(recv_pos_ecef - trans_pos_ecef)
        if dist == 0:
            return 0
        # 自由空间损耗公式：L = (4πdf/c)^2，功率与损耗成反比
        loss = (4 * np.pi * dist * freq / params.c) ** 2
        return 1 / loss  # 相对功率（无需绝对功率，仅需比值）
    
    # 计算真实卫星信号功率（4颗卫星平均功率）
    real_power = np.mean([calc_signal_power(sat, target_pos) for sat in satellite_pos])
    # 计算欺骗信号功率（4个转发器平均功率）
    deception_power = np.mean([calc_signal_power(jammer, target_pos) for jammer in jammer_pos])
    
    # 功率比转换为dB：10*log10(欺骗功率/真实功率)
    power_ratio_dB = 10 * np.log10(deception_power / real_power)
    return power_ratio_dB > params.power_ratio_threshold

# ========================= 4. 欺骗时延计算与修正=========================
def calculate_deception_delay(jammer_pos, satellite_pos, target_pos, deception_pos, params):
    """
    计算各转发器需注入的时延Δτ_i，并修正为非负值
    :return: 时延向量T(4,)
    """
    deception_pos_ecef = lla_to_ecef(deception_pos[0], deception_pos[1], deception_pos[2])
    target_pos_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])
    delays = []
    for i in range(params.jammer_num):
        jammer = lla_to_ecef(jammer_pos[i][0], jammer_pos[i][1], jammer_pos[i][2])
        sat = lla_to_ecef(satellite_pos[i][0], satellite_pos[i][1], satellite_pos[i][2])
        
        # 计算各距离（经纬度转距离简化处理，实际需用ECEF坐标系）
        R_SD = np.linalg.norm(sat - deception_pos_ecef)  # 卫星到欺骗点(米)
        R_SJ = np.linalg.norm(sat - jammer)  # 卫星到转发器(米)
        R_JR = np.linalg.norm(jammer - target_pos_ecef) # 转发器到目标(米)
        
        # 时延公式：Δτ_i = (R_SD - R_SJ - R_JR) / c
        tau = (R_SD - R_SJ - R_JR) / params.c
        delays.append(tau)
    
    # 时延修正：若存在负值，整体偏移使最小值为0
    delays = np.array(delays)
    min_tau = np.min(delays)
    for num in range(len(delays)):
        if delays[num] < 0:
            delays[num] = delays[num] + abs(min_tau)
    return delays

# ========================= 5. 定位误差求解（最小二乘法）=========================
def solve_position_error(satellite_pos, target_pos, delays, params):
    """
    通过最小二乘求解定位偏差ΔX = (M^T M)^-1 M^T (T+A)
    :return: 三维定位误差(Δx, Δy, Δz)（单位：°/km）
    """
    target_pos_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])

    # 构建观测矩阵M
    M = []
    R_SR_list = []  # 卫星到目标真实距离
    for sat in satellite_pos:
        sat_ecef = np.asarray(lla_to_ecef(sat[0], sat[1], sat[2])).reshape(-1)
        tgt = np.asarray(target_pos_ecef).reshape(-1)
        if sat_ecef.size != 3 or tgt.size != 3:
            print(f"Warning: invalid sat/target shape in solve_position_error: sat.size={sat_ecef.size}, tgt.size={tgt.size}; returning zero error")
            return np.array([0, 0, 0])
        delta = sat_ecef - tgt
        R_SR = np.linalg.norm(delta)  # 米
        if R_SR == 0 or np.isnan(R_SR):
            print("Warning: zero or NaN range R_SR encountered; returning zero error")
            return np.array([0, 0, 0])
        R_SR_list.append(R_SR)
        unit_vec = delta / R_SR  # m单位的单位向量
        if unit_vec.size < 3:
            print("Warning: unit vector size < 3 in solve_position_error; returning zero error")
            return np.array([0, 0, 0])
        M.append([unit_vec[0]/params.c, unit_vec[1]/params.c, unit_vec[2]/params.c, 1])
    M = np.array(M)
    
    # 构建时延向量T
    T = delays.reshape(-1, 1)
    
    # 构建修正向量A
    A = []
    for i in range(params.jammer_num):
        sat = lla_to_ecef(satellite_pos[i][0], satellite_pos[i][1], satellite_pos[i][2])
        jammer = lla_to_ecef(params.jammer_pos[i][0], params.jammer_pos[i][1], params.jammer_pos[i][2])
        R_SJ = np.linalg.norm(sat - jammer)  # 米
        R_JR = np.linalg.norm(jammer - target_pos_ecef)  # 米
        R_SR = R_SR_list[i]
        a = (R_SJ + R_JR - R_SR) / params.c
        A.append(a)
    A = np.array(A).reshape(-1, 1)
    
    # 最小二乘求解
    try:
        MtM_inv = np.linalg.inv(M.T @ M)
        delta_X = MtM_inv @ M.T @ (T + A)
        # delta_X = [Δx, Δy, Δz, τ]^T，提取前三维定位误差（单位：米，ECEF）
        pos_error_m = delta_X[:3].flatten()

        # 将位置误差从 ECEF(m) 恢复为经纬高差（单位：度, 度, km）
        new_pos_ecef = target_pos_ecef + pos_error_m
        new_lon_deg, new_lat_deg, new_h_km = ecef_to_lla(new_pos_ecef[0], new_pos_ecef[1], new_pos_ecef[2])

        # 输入目标位置为 (lon_deg, lat_deg, h_km)
        delta_lon_deg = new_lon_deg - target_pos[0]
        delta_lat_deg = new_lat_deg - target_pos[1]
        delta_h_km = new_h_km - target_pos[2]

        pos_error = np.array([delta_lon_deg, delta_lat_deg, delta_h_km])
        return pos_error
    except np.linalg.LinAlgError:
        return np.array([0, 0, 0])  # 求解失败返回零误差

# ========================= 6. 主仿真流程=========================
def main_simulation():
    # 初始化参数
    params = SimParams()
    time_steps = int(params.sim_time * params.sampling_freq)
    target_positions = []  # 目标真实轨迹
    error_positions = []   # 目标受干扰后轨迹
    error_positions.append(params.target_init_pos)
    errors = []            # 定位误差序列
    
    # 仿真主循环
    for t in range(time_steps):
        # 1. 计算目标实时位置
        target_pos = params.target_init_pos + params.target_velocity * t #目标预定实时位置
        target_positions.append(target_pos)

        # if t == 0:
        #     error_target_pos = target_pos
        # else:
        #     error_target_pos = error_positions[t] + params.target_velocity * t #目标受干扰后实时位置
        
        # 2. 筛选GDOP最优卫星
        optimal_sats = select_optimal_satellites(target_pos, params.satellite_pos, params)
        
        # 2.1 计算当前时刻的欺骗点（沿着 deception_points 序列平滑移动以引导目标）
        # current_deception = interpolate_deception_points(params.deception_points, t, time_steps)

        # 3. 判定欺骗信号有效性
        if not is_deception_valid(params.jammer_pos, target_pos, optimal_sats, params):
            # 欺骗无效，定位误差为0（或按正常GNSS误差处理）
            error = np.array([0, 0, 0])
            errors.append(error)
            error_positions.append(target_pos)
            continue
        
        # 4. 计算欺骗时延（使用当前插值得到的欺骗点）
        delays = calculate_deception_delay(params.jammer_pos, optimal_sats, target_pos, 
                                          params.deception_pos, params)
        
        # 5. 求解定位误差
        pos_error = solve_position_error(optimal_sats, target_pos, delays, params)
        errors.append(pos_error)
        
        # 6. 计算受干扰后的目标位置（真实位置 + 定位误差）
        error_pos = target_pos - pos_error
        error_positions.append(error_pos)
    
    # 转换为numpy数组便于处理
    target_positions = np.array(target_positions)
    error_positions = np.array(error_positions)
    errors = np.array(errors)
    
    # 7. 结果可视化
    visualize_results(target_positions, error_positions, errors, params)

# ========================= 7. 结果可视化（2D/3D轨迹+误差曲线）=========================
def visualize_results(true_traj, error_traj, errors, params):
    # 绘制2D轨迹对比（经纬度平面）
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for i in range(len(params.jammer_pos)):
        plt.scatter(params.jammer_pos[i][0], params.jammer_pos[i][1], c='r', marker='*', s=200, label='干扰源' + str(i+1))

    plt.scatter(params.target_init_pos[0], params.target_init_pos[1], c='k', marker='*', s=200, label='起始点')
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'y-', label='预定轨迹', linewidth=2)
    plt.plot(error_traj[:, 0], error_traj[:, 1], 'b-', label='受干扰轨迹', linewidth=2)
    # 绘制欺骗路径或单点
    if hasattr(params, 'deception_points'):
        dp = np.array(params.deception_points)
        plt.plot(dp[:, 0], dp[:, 1], 'g--', label='欺骗路径')
        plt.scatter(dp[:, 0], dp[:, 1], c='g', marker='o', s=80)
        plt.scatter(dp[-1, 0], dp[-1, 1], c='g', marker='*', s=200, label='最终欺骗点')
    else:
        plt.scatter(params.deception_pos[0], params.deception_pos[1], c='g', marker='*', s=200, label='欺骗点')
    plt.xlabel('经度(°)')
    plt.ylabel('纬度(°)')
    plt.legend()
    plt.title('2D轨迹对比')
    plt.grid(True)
    
    # 绘制3D轨迹对比
    ax = plt.subplot(2, 2, 2, projection='3d')
    for i in range(len(params.jammer_pos)):
        ax.scatter(params.jammer_pos[i][0], params.jammer_pos[i][1], params.jammer_pos[i][2], c='r', marker='*', s=200, label='干扰源' + str(i+1))
    ax.scatter(params.target_init_pos[0], params.target_init_pos[1], params.target_init_pos[2], 
               c='k', marker='*', s=200, )
    ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 'y-')
    ax.plot(error_traj[:, 0], error_traj[:, 1], error_traj[:, 2], 'b-')
    if hasattr(params, 'deception_points'):
        dp = np.array(params.deception_points)
        ax.plot(dp[:, 0], dp[:, 1], dp[:, 2], 'g--')
        ax.scatter(dp[:, 0], dp[:, 1], dp[:, 2], c='g', marker='o', s=80)
        ax.scatter(dp[-1, 0], dp[-1, 1], dp[-1, 2], c='g', marker='*', s=200, label='最终欺骗点')
    else:
        ax.scatter(params.deception_pos[0], params.deception_pos[1], params.deception_pos[2], 
                   c='g', marker='*', s=200)
    ax.set_xlabel('经度(°)')
    ax.set_ylabel('纬度(°)')
    ax.set_zlabel('高度(km)')
    ax.legend()
    ax.set_title('3D轨迹对比')
    
    # 绘制定位误差时间序列
    time = np.arange(len(errors)) / params.sampling_freq
    plt.subplot(2, 2, 3)
    plt.plot(time, errors[:, 0], 'r-', label='经度误差(°)')
    plt.plot(time, errors[:, 1], 'g-', label='纬度误差(°)')
    plt.plot(time, errors[:, 2], 'b-', label='高度误差(km)')
    plt.xlabel('时间(s)')
    plt.ylabel('误差值')
    plt.legend()
    plt.title('定位误差随时间变化')
    plt.grid(True)
    
    # 绘制误差统计直方图
    plt.subplot(2, 2, 4)
    total_error = np.linalg.norm(errors, axis=1)
    plt.hist(total_error, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('总定位误差(°/km)')
    plt.ylabel('频次')
    plt.title('定位误差分布')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 运行仿真
if __name__ == "__main__":
    main_simulation()