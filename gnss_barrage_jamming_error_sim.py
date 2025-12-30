import math
import numpy as np
from scipy.linalg import inv
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

# 接收机参数
Bn = 2.046e6   # 噪声带宽 (Hz, GPS L1 频段)
sigma_rho_nom = 0.2  # 标称伪距基线误差 (m)
G_ant = 10     # 抗干扰波束成形增益 (dB)

# 干扰参数
P_j = -100     # 干扰功率 (dBm)
P_s = -130     # 卫星信号功率 (dBm)
B_j = 2e6      # 干扰带宽 (Hz, 窄带干扰)

# 卫星几何参数 (示例: 4颗卫星的方位角/俯仰角，实际需从星历获取)
sat_az = [30, 120, 210, 300]    # 卫星方位角 (°)
sat_el = [45, 45, 45, 45]       # 卫星俯仰角 (°)
user_pos = [0, 0, 0]            # 用户接收机位置 (x,y,z 笛卡尔坐标, m)

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
    # 步骤1: 计算干扰后载噪比
    C_N0_j, delta_CN0 = calc_cn0_jamming(C_N0_nom, P_s, P_j, Bn, B_j, G_ant)
    print(f"干扰后载噪比: {C_N0_j:.2f} dB-Hz")
    print(f"载噪比衰减量: {delta_CN0:.2f} dB")

    # 步骤2: 计算伪距误差
    sigma_rho_total = calc_sigma_rho_jamming(C_N0_j, Bn, c, sigma_rho_nom)
    if sigma_rho_total == np.inf:
        print("部分卫星信号失锁，无法定位")
        return
    print(f"总伪距误差: {sigma_rho_total:.4f} m")

    # 步骤3: 计算 PDOP
    pdop = calc_pdop(sat_az, sat_el, user_pos)
    print(f"PDOP 值: {pdop:.2f}")

    # 步骤4: 计算三维定位误差
    sigma_pos = pdop * sigma_rho_total
    print(f"干扰下三维定位误差: {sigma_pos:.4f} m")
    return sigma_pos

# -------------------------- 6. 可视化分析函数 --------------------------
def visualize_jamming_analysis():
    """绘制干扰影响的多维度分析图"""
    
    # 数据采样范围
    P_j_range = np.linspace(-120, -80, 30)  # 干扰功率范围 (dBm)
    C_N0_j_list = []
    sigma_rho_list = []
    sigma_pos_list = []
    
    for pj in P_j_range:
        C_N0_j, _ = calc_cn0_jamming(C_N0_nom, P_s, pj, Bn, B_j, G_ant)
        sigma_rho = calc_sigma_rho_jamming(C_N0_j, Bn, c, sigma_rho_nom)
        pdop = calc_pdop(sat_az, sat_el, user_pos)
        
        C_N0_j_list.append(C_N0_j)
        sigma_rho_list.append(sigma_rho if sigma_rho != np.inf else np.nan)
        sigma_pos_list.append(pdop * sigma_rho if sigma_rho != np.inf else np.nan)
    
    # 创建子图布局
    fig = plt.figure(figsize=(15, 10))
    
    # 子图1: 干扰功率 vs 载噪比
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(P_j_range, C_N0_j_list, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=C_N0_th, color='r', linestyle='--', label='失锁阈值')
    ax1.axhline(y=C_N0_nom, color='g', linestyle='--', label='标称值')
    ax1.set_xlabel('干扰功率 (dBm)', fontsize=11)
    ax1.set_ylabel('载噪比 C/N₀ (dB-Hz)', fontsize=11)
    ax1.set_title('干扰功率 vs 载噪比', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2: 干扰功率 vs 伪距误差
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(P_j_range, sigma_rho_list, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.axhline(y=sigma_rho_nom, color='orange', linestyle='--', label='基线误差')
    ax2.set_xlabel('干扰功率 (dBm)', fontsize=11)
    ax2.set_ylabel('伪距误差 σ_ρ (m)', fontsize=11)
    ax2.set_title('干扰功率 vs 伪距误差', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    # 子图3: 干扰功率 vs 三维定位误差
    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogy(P_j_range, sigma_pos_list, 'r-', linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('干扰功率 (dBm)', fontsize=11)
    ax3.set_ylabel('定位误差 σ_pos (m)', fontsize=11)
    ax3.set_title('干扰功率 vs 三维定位误差', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    
    # 子图4: 卫星几何配置（极坐标）
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    az_rad = [math.radians(az) for az in sat_az]
    el_rad = [math.radians(el) for el in sat_el]
    r = [90 - el for el in sat_el]  # 转换为天顶角
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for i, (az, r_val) in enumerate(zip(az_rad, r)):
        ax4.scatter(az, r_val, s=300, c=colors[i], marker='*', edgecolors='black', linewidth=1.5)
        ax4.text(az, r_val + 5, f'Sat{i+1}', ha='center', fontsize=10, fontweight='bold')
    
    ax4.set_ylim(0, 90)
    ax4.set_yticks([0, 30, 60, 90])
    ax4.set_yticklabels(['90°', '60°', '30°', '0°'])
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)
    ax4.set_title('卫星几何配置（天顶图）', fontsize=12, fontweight='bold', pad=20)
    ax4.grid(True)
    
    # 子图5: 参数对比表格
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    C_N0_j_final, delta_CN0_final = calc_cn0_jamming(C_N0_nom, P_s, P_j, Bn, B_j, G_ant)
    sigma_rho_final = calc_sigma_rho_jamming(C_N0_j_final, Bn, c, sigma_rho_nom)
    pdop_final = calc_pdop(sat_az, sat_el, user_pos)
    sigma_pos_final = pdop_final * sigma_rho_final
    
    table_data = [
        ['参数', '数值', '单位'],
        ['标称载噪比', f'{C_N0_nom:.1f}', 'dB-Hz'],
        ['干扰后载噪比', f'{C_N0_j_final:.2f}', 'dB-Hz'],
        ['衰减量', f'{delta_CN0_final:.2f}', 'dB'],
        ['伪距误差', f'{sigma_rho_final:.4f}', 'm'],
        ['PDOP值', f'{pdop_final:.2f}', '—'],
        ['定位误差', f'{sigma_pos_final:.4f}', 'm'],
        ['干扰功率', f'{P_j}', 'dBm'],
        ['信号功率', f'{P_s}', 'dBm'],
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.35, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 交替行颜色
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    ax5.set_title('仿真参数摘要', fontsize=12, fontweight='bold', pad=10)
    
    # 子图6: 误差趋势统计
    ax6 = plt.subplot(2, 3, 6)
    
    # 统计误差增长倍数
    sigma_rho_no_jam = sigma_rho_nom  # 无干扰伪距误差
    rho_increase = np.array(sigma_rho_list) / sigma_rho_no_jam
    
    ax6.fill_between(range(len(rho_increase)), 1, rho_increase, alpha=0.3, color='red')
    ax6.plot(range(len(rho_increase)), rho_increase, 'r-', linewidth=2.5, marker='o', markersize=5)
    ax6.axhline(y=1, color='g', linestyle='--', label='无干扰基线', linewidth=1.5)
    ax6.set_xlabel('干扰功率等级', fontsize=11)
    ax6.set_ylabel('误差增长倍数', fontsize=11)
    ax6.set_title('伪距误差增长倍数', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
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
    sigma_pos = calc_position_error()
    export_analysis_report()
    print("生成可视化分析图表...")
    visualize_jamming_analysis()
