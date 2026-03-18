import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

# 设置中文字体与负号正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 1. 核心参数配置 =========================
class SimParams:
    def __init__(self):
        self.c = 3e8
        self.GDOP_threshold = 3
        self.power_ratio_threshold = 5
        
        self.jammer_num = 4
        self.jammer_pos = np.array([
            [115.2, 29.0, 1.0],
            [115.3, 29.1, 1.0],
            [115.1, 29.2, 1.0],
            [115.4, 29.0, 1.0]
        ])
        
        self.target_init_pos = np.array([115.193, 29.027, 1])
        self.target_velocity = np.array([0.002, 0.001, 0])
        self.sim_time = 400
        self.sampling_freq = 1
        
        self.deception_pos = np.array([116.055, 30.33, 1.0])

        self.satellite_pos = np.array([
            [120.0, 30.0, 20000],
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

def save_all_deception_results(all_data):
    with open("deception_print.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

# ========================= 经纬度 ↔ ECEF 转换 =========================
def lla_to_ecef(lon_deg, lat_deg, h_km):
    a = 6378137.0
    f = 1/298.257223563
    e2 = 2*f - f*f
    h_m = h_km * 1000.0

    φ = np.deg2rad(lat_deg)
    λ = np.deg2rad(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(φ)**2)
    X = (N + h_m) * np.cos(φ) * np.cos(λ)
    Y = (N + h_m) * np.cos(φ) * np.sin(λ)
    Z = ((1 - e2) * N + h_m) * np.sin(φ)
    return np.array([X, Y, Z])        

def ecef_to_lla(X, Y, Z):
    a = 6378137.0
    f = 1/298.257223563
    b = a * (1 - f)
    e2 = 2*f - f*f
    ep2 = (a**2 - b**2) / (b**2)
    p = np.sqrt(X*X + Y*Y)
    if p < 1e-12:
        return 0.0, np.sign(Z)*90.0, (abs(Z)-b)/1000.0
    theta = np.arctan2(Z * a, p * b)
    lon = np.arctan2(Y, X)
    lat = np.arctan2(Z + ep2 * b * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = (p / np.cos(lat) - N) / 1000.0
    return np.rad2deg(lon), np.rad2deg(lat), h

# ========================= 卫星筛选 =========================
def select_optimal_satellites(target_pos, satellite_list, params):
    target_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])
    sat_ecef_list = np.array([lla_to_ecef(s[0], s[1], s[2]) for s in satellite_list])
    min_gdop = float('inf')
    best_idx = None
    for idx_comb in combinations(range(len(satellite_list)), 4):
        G = []
        valid = True
        for i in idx_comb:
            d = sat_ecef_list[i] - target_ecef
            dist = np.linalg.norm(d)
            if dist < 1e-3:
                valid = False
                break
            uv = d / dist
            G.append([uv[0], uv[1], uv[2], 1])
        if not valid:
            continue
        G = np.array(G)
        A = G.T @ G
        try:
            inv = np.linalg.inv(A) if np.linalg.cond(A) < 1e12 else np.linalg.pinv(A)
            gdop = np.sqrt(np.trace(inv))
            if gdop < min_gdop:
                min_gdop = gdop
                best_idx = idx_comb
        except:
            continue
    if best_idx is None:
        return satellite_list[:4]
    return np.array([satellite_list[i] for i in best_idx])

# ========================= ✅ 终极完美版本：按你的数学公式实现 =========================
def interpolate_deception_points(start_pos, target_deception_pos, t, target_velocity):
    lon0, lat0, h0 = start_pos
    dx = target_deception_pos[0] - lon0
    dy = target_deception_pos[1] - lat0
    dist_total = np.sqrt(dx**2 + dy**2)
    if dist_total < 1e-12:
        return start_pos.copy()
    ux = dx / dist_total
    uy = dy / dist_total

    vx, vy = target_velocity[0], target_velocity[1]
    dist_real = np.sqrt((vx * t)** 2 + (vy * t)** 2)

    lon = lon0 + ux * dist_real
    lat = lat0 + uy * dist_real
    h = h0
    return np.array([lon, lat, h])

# ========================= 欺骗有效性判断 =========================
def is_deception_valid(jammer_pos, target_pos, satellite_pos, params):
    def calc_power(tx, rx, freq = 1561.09e6):
        tx_ecef = lla_to_ecef(tx[0], tx[1], tx[2])
        rx_ecef = lla_to_ecef(rx[0], rx[1], rx[2])
        dist = np.linalg.norm(tx_ecef - rx_ecef)
        if dist < 1: return 0.0
        loss = (4 * np.pi * dist * freq / params.c)**2
        return 1.0 / loss
    p_real = np.mean([calc_power(s, target_pos) for s in satellite_pos])
    p_decep = np.mean([calc_power(j, target_pos, 100.03e3) for j in jammer_pos])
    return 10*np.log10(p_decep / p_real) > params.power_ratio_threshold

# ========================= 欺骗时延计算 =========================
def calculate_deception_delay(jammer_pos, satellite_pos, target_pos, deception_pos, params):
    dec_ecef = lla_to_ecef(deception_pos[0], deception_pos[1], deception_pos[2])
    tgt_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])
    delays = []
    for i in range(params.jammer_num):
        j_ecef = lla_to_ecef(jammer_pos[i][0], jammer_pos[i][1], jammer_pos[i][2])
        s_ecef = lla_to_ecef(satellite_pos[i][0], satellite_pos[i][1], satellite_pos[i][2])
        R_SD = np.linalg.norm(s_ecef - dec_ecef)
        R_SJ = np.linalg.norm(s_ecef - j_ecef)
        R_JR = np.linalg.norm(j_ecef - tgt_ecef)
        tau = (R_SD - R_SJ - R_JR) / params.c
        delays.append(tau)
    delays = np.array(delays)
    min_tau = delays.min()
    if min_tau < 0:
        delays -= min_tau
    return delays

# ========================= 最小二乘定位误差 =========================
def solve_position_error(satellite_pos, target_pos, delays, params):
    tgt_ecef = lla_to_ecef(target_pos[0], target_pos[1], target_pos[2])
    M = []
    R_list = []
    for s in satellite_pos:
        s_ecef = lla_to_ecef(s[0], s[1], s[2])
        d = s_ecef - tgt_ecef
        r = np.linalg.norm(d)
        R_list.append(r)
        uv = d / r
        M.append([uv[0]/params.c, uv[1]/params.c, uv[2]/params.c, 1])
    M = np.array(M)
    T = delays.reshape(-1,1)
    A = []
    for i in range(params.jammer_num):
        s = lla_to_ecef(satellite_pos[i][0], satellite_pos[i][1], satellite_pos[i][2])
        j = lla_to_ecef(params.jammer_pos[i][0], params.jammer_pos[i][1], params.jammer_pos[i][2])
        r_sj = np.linalg.norm(s-j)
        r_jr = np.linalg.norm(j-tgt_ecef)
        A.append((r_sj + r_jr - R_list[i])/params.c)
    A = np.array(A).reshape(-1,1)
    try:
        dx = np.linalg.inv(M.T@M) @ M.T @ (T+A)
        new_ecef = tgt_ecef + dx[:3].flatten()
        lon, lat, h = ecef_to_lla(new_ecef[0], new_ecef[1], new_ecef[2])
        return np.array([lon-target_pos[0], lat-target_pos[1], h-target_pos[2]])
    except:
        return np.array([0,0,0])

# ========================= 主仿真 =========================
def main_simulation():
    params = SimParams()
    time_steps = int(params.sim_time * params.sampling_freq)
    all_frame_data = []

    true_traj = []
    error_traj = []
    errors = []
    
    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2, projection='3d')
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    for j in params.jammer_pos:
        ax1.scatter(j[0], j[1], c='r', marker='*', s=150, zorder=5)
        ax2.scatter(j[0], j[1], j[2], c='r', marker='*', s=150, zorder=5)
    
    ax1.scatter(params.target_init_pos[0], params.target_init_pos[1], c='k', marker='*', s=200, label="起点")
    ax1.scatter(params.deception_pos[0], params.deception_pos[1], c='g', marker='*', s=200, label="欺骗点")
    
    ax2.scatter(params.target_init_pos[0], params.target_init_pos[1], params.target_init_pos[2], c='k', marker='*', s=200)
    ax2.scatter(params.deception_pos[0], params.deception_pos[1], params.deception_pos[2], c='g', marker='*', s=200)

    ax1.set_xlabel("经度(°)")
    ax1.set_ylabel("纬度(°)")
    ax1.set_title("【实时动画】目标航行轨迹")
    ax1.grid(True)
    ax1.legend()

    for t in range(time_steps):
        tgt = params.target_init_pos + params.target_velocity * t
        true_traj.append(tgt)
        sats = select_optimal_satellites(tgt, params.satellite_pos, params)
        valid = is_deception_valid(params.jammer_pos, tgt, sats, params)
        
        if not valid:
            err = np.array([0,0,0])
            err_tgt = tgt
            current_deception = params.target_init_pos
        else:
            # ✅ 调用终极完美插值
            current_deception = interpolate_deception_points(
                start_pos=params.target_init_pos,
                target_deception_pos=params.deception_pos,
                t=t,
                target_velocity=params.target_velocity
            )
            delay = calculate_deception_delay(params.jammer_pos, sats, tgt, current_deception, params)
            err = solve_position_error(sats, tgt, delay, params)
            err = np.array([-err[0], -err[1], -err[2]])
            err_tgt = tgt + err

        # 保存所有帧
        frame = {
            "time_step": int(t),
            "deception_point_now": {
                "lon_deg": float(current_deception[0]),
                "lat_deg": float(current_deception[1]),
                "h_km": float(current_deception[2])
            },
            "target_after_deception": {
                "lon_deg": float(err_tgt[0]),
                "lat_deg": float(err_tgt[1]),
                "h_km": float(err_tgt[2])
            }
        }
        all_frame_data.append(frame)

        error_traj.append(err_tgt)
        errors.append(err)

        tr = np.array(true_traj)
        er = np.array(error_traj)
        
        ax1.cla()
        ax1.grid(True)
        for j in params.jammer_pos:
            ax1.scatter(j[0], j[1], c='r', marker='*', s=150, zorder=5)
        ax1.scatter(params.target_init_pos[0], params.target_init_pos[1], c='k', marker='*', s=200)
        ax1.scatter(params.deception_pos[0], params.deception_pos[1], c='g', marker='*', s=200)
        ax1.plot(tr[:,0], tr[:,1], 'y-', linewidth=2, label="预定轨迹")
        ax1.plot(er[:,0], er[:,1], 'b-', linewidth=2, label="受干扰轨迹")
        ax1.scatter(er[-1,0], er[-1,1], c='blue', s=80, zorder=10, label="当前目标")
        ax1.set_xlabel("经度(°)")
        ax1.set_ylabel("纬度(°)")
        ax1.set_title(f"实时仿真 | 时间：{t}s | 欺骗有效：{valid}")
        ax1.legend()

        ax2.cla()
        ax2.plot(tr[:,0], tr[:,1], tr[:,2], 'y-')
        ax2.plot(er[:,0], er[:,1], er[:,2], 'b-')
        ax2.scatter(er[-1,0], er[-1,1], er[-1,2], c='blue', s=80)
        ax2.set_xlabel("经度(°)")
        ax2.set_ylabel("纬度(°)")
        ax2.set_zlabel("高度(km)")

        err_arr = np.array(errors)
        ts = np.arange(len(err_arr))
        ax3.cla()
        ax3.grid(True)
        ax3.plot(ts, err_arr[:,0], 'r-', label="经度误差")
        ax3.plot(ts, err_arr[:,1], 'g-', label="纬度误差")
        ax3.plot(ts, err_arr[:,2], 'b-', label="高度误差")
        ax3.legend()
        ax3.set_title("定位误差")

        plt.pause(0.05)

    save_all_deception_results(all_frame_data)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main_simulation()