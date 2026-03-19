import math
import threading
import json
import matplotlib
# 指定Matplotlib后端，解决多线程绘图兼容性问题
matplotlib.use('TkAgg')
# matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Callable

# 设置中文字体和负号显示解决绘图中文乱码问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 全局基础参数 ======================
R_EARTH = 637813.0          # 地球半径（米），用于经纬度和平面坐标转换
WINDOW_RANGE = 60            # 可视化窗口范围（米），以主节点为中心±60米
TRANSITION_SPEED = 0.05       # 队形切换平滑过渡系数（仅切换时生效，越小越慢）

# ====================== 避碰算法参数（对应C++源码参数） ======================
COLLISION_RADIUS = 4.0       # 单个节点的安全碰撞半径（米）
MAX_COLLISION_ITER = 15      # 避碰算法最大迭代次数
MAX_ADJUST_STEP = 1.0        # 单次避碰调整的最大步长（防止移动过快）


ERROR_STABLE_THRESHOLD = 0.02  # 队形误差小于此值认为变换完成

# ====================== 数据结构定义 ======================
@dataclass
class Point2D:
    """
    2D点/向量类，复刻C++ Point2D，实现向量运算
    用于避碰算法的坐标计算、向量加减/归一化/模长计算
    """
    x: float  # X轴坐标
    y: float  # Y轴坐标

    # 向量减法运算
    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)
    
    # 向量加法运算
    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)
    
    # 向量 × 浮点数
    def __mul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)
    
    # 浮点数 × 向量
    __rmul__ = __mul__

    # 向量 / 浮点数
    def __truediv__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x / scalar, self.y / scalar)

    # 计算向量的模长（两点间距离）
    def norm(self) -> float:
        return math.hypot(self.x, self.y)
    
    # 向量归一化（单位向量），避免零向量除零错误
    def normalized(self) -> 'Point2D':
        n = self.norm()
        if n < 1e-6:
            return Point2D(0.0, 0.0)
        return Point2D(self.x / n, self.y / n)

@dataclass
class UUVNode:
    """
    无人载具节点类，存储每个节点的所有属性
    包含地理坐标、运动参数、队形相对坐标、目标坐标
    """
    id: int                 # 节点唯一编号（0为主节点）
    lon: float              # 经度（地理坐标）
    lat: float              # 纬度（地理坐标）
    speed: float            # 运动速度
    heading: float          # 航向角
    rel_x: float = 0.0      # 当前相对主节点的X坐标
    rel_y: float = 0.0      # 当前相对主节点的Y坐标
    target_x: float = 0.0   # 目标队形的X坐标（用于平滑过渡）
    target_y: float = 0.0   # 目标队形的Y坐标（用于平滑过渡）

@dataclass
class FormationConfig:
    """
    编队仿真配置类，统一管理所有仿真参数
    【约束】node_num 需满足 4 ≤ node_num ≤ 10（对应从节点数3~9）
    """
    formation_type: str     # 当前队形类型
    node_num: int           # 总节点数量（1主 + n从，需4≤node_num≤10）
    main_lon: float         # 主节点初始经度
    main_lat: float         # 主节点初始纬度
    rel_distance: float     # 节点间相对间距
    init_speed: float       # 初始运动速度
    init_heading: float     # 初始航向角
    turn_radius: float = float('inf')  # 转弯半径（inf为直线行驶）
    acceleration: float = 0.0          # 加速度
    sim_step: float = 0.1              # 仿真步长
    output_interval: float = 1.0       # 【新增】状态信息输出间隔（秒），默认1秒输出一次

# ====================== 编队仿真核心类 ======================
class UUVFormationSimulator:
    def __init__(self, config: FormationConfig):
        """
        仿真器初始化函数
        :param config: 编队配置参数
        """
        self.config = config                # 仿真配置
        self.nodes: List[UUVNode] = []     # 所有节点列表
        self.current_time = 0.0            # 仿真时间
        self.last_output_time = 0.0        # 【新增】上次输出状态信息的时间

        # ====================== 【修复】这里补上缺失的3个变量 ======================
        self.is_transition = False         # 是否正在队形变换
        self.last_formation = config.formation_type  # 上一个队形
        self.transition_data = []          # 变换过程数据

        # 队形映射表：键盘输入 → 队形类型
        self.formation_map = {
            "1": "line",       # 竖直线
            "2": "rect",       # 矩形
            "3": "circle",     # 圆形
            "4": "diamond",    # 菱形
            "5": "triangle"    # 三角形
        }
        # 初始化校验 + 节点初始化
        self._validate_config()
        self._init_nodes()

    def _validate_config(self):
        """校验配置参数合法性，特别是节点数量约束"""
        if not (4 <= self.config.node_num <= 10):
            raise ValueError(f"总节点数 node_num 必须满足 4 ≤ node_num ≤ 10（当前值：{self.config.node_num}）")
        if self.config.rel_distance <= 0:
            raise ValueError(f"节点间距 rel_distance 必须大于0（当前值：{self.config.rel_distance}）")
        if self.config.output_interval < self.config.sim_step:
            raise ValueError(f"输出间隔不能小于仿真步长（当前步长：{self.config.sim_step}s）")

    def _init_nodes(self):
        """初始化主节点+从节点，直接生成最终队形（无初始过渡）"""
        # 1. 初始化主节点（ID=0）
        main_node = UUVNode(0, self.config.main_lon, self.config.main_lat, self.config.init_speed, self.config.init_heading)
        self.nodes.append(main_node)
        # 2. 初始化从节点（ID=1~n）
        for i in range(1, self.config.node_num):
            self.nodes.append(UUVNode(i, self.config.main_lon, self.config.main_lat, self.config.init_speed, self.config.init_heading))
        # 3. 设置目标队形 + 直接赋值初始坐标（无过渡）
        self._set_target_formation()
        self._set_initial_position()

    def _set_initial_position(self):
        """初始队形直接显示，无平滑过渡：主节点固定中心，从节点直接赋值"""
        # 主节点初始坐标强制为0
        self.nodes[0].rel_x = 0.0
        self.nodes[0].rel_y = 0.0
        for node in self.nodes[1:]:
            node.rel_x = node.target_x
            node.rel_y = node.target_y

    def switch_formation(self, cmd: str):
        """
        切换队形函数
        :param cmd: 键盘输入的队形编号
        """
        if cmd in self.formation_map:
            new_form = self.formation_map[cmd]
            if new_form == self.config.formation_type:
                print("✅ 当前已是该队形")
                return

            # 开始新变换
            self.last_formation = self.config.formation_type
            self.config.formation_type = new_form
            self._set_target_formation()
            self.is_transition = True
            self.transition_data = []  # 清空重写
            print(f"\n✅ 切换队形：{self.last_formation} → {new_form}，开始记录...")

    def _set_target_formation(self):
        """为所有从节点设置目标队形坐标"""
        slave_count = len(self.nodes) - 1  # 从节点数量
        if slave_count <= 0:
            return
        # 生成目标队形坐标
        positions = self._generate_formation_positions(slave_count)
        # 为每个从节点绑定目标坐标
        for i, node in enumerate(self.nodes[1:]):
            if i < len(positions):
                node.target_x, node.target_y = positions[i]

    def _transition_formation(self):
        """队形平滑过渡函数：仅切换队形时生效，逐步逼近目标坐标"""
        for node in self.nodes[1:]:
            node.rel_x += (node.target_x - node.rel_x) * TRANSITION_SPEED
            node.rel_y += (node.target_y - node.rel_y) * TRANSITION_SPEED

    # ====================== 核心：C++避碰算法1:1移植（包含主节点） ======================
    def checkCollision1(self, positions: List[Point2D]) -> List[Point2D]:
        """
        实时避碰算法：主节点是静态障碍物，不移动
        只有从节点会躲避主节点和其他从节点
        """
        adjusted = positions.copy()
        iter_num = 0
        has_collision = True
        num_uavs = len(adjusted)

        # 主节点坐标固定
        main_pos = adjusted[0]

        while has_collision and iter_num < MAX_COLLISION_ITER:
            has_collision = False
            collision_pairs = []

            # 检测所有碰撞
            for i in range(num_uavs):
                for j in range(i + 1, num_uavs):
                    diff = adjusted[i] - adjusted[j]
                    distance = diff.norm()
                    if distance <= (COLLISION_RADIUS * 2):
                        collision_pairs.append((distance, i, j))
                        has_collision = True

            if not has_collision:
                break

            collision_pairs.sort(key=lambda x: x[0])

            # 处理碰撞：主节点不动
            for distance, i, j in collision_pairs:
                diff = adjusted[i] - adjusted[j]
                dir_vec = diff.normalized()
                need_adjust = (COLLISION_RADIUS * 2) - distance
                adjust_step = min(need_adjust / 2.0, MAX_ADJUST_STEP)

                if i == 0:
                    # 主节点碰从节点 → 只移从节点
                    adjusted[j] = adjusted[j] - dir_vec * adjust_step
                elif j == 0:
                    # 从节点碰主节点 → 只移从节点
                    adjusted[i] = adjusted[i] + dir_vec * adjust_step
                else:
                    # 从节点之间互相移
                    adjusted[i] = adjusted[i] + dir_vec * adjust_step
                    adjusted[j] = adjusted[j] - dir_vec * adjust_step

            iter_num += 1

        if iter_num >= MAX_COLLISION_ITER and has_collision:
            print("⚠️ 避碰警告：达到最大迭代次数")
        return adjusted
    
    def apply_collision_avoidance(self):
        """
        避碰执行：
        主节点 = 静态障碍物，绝对不移动
        从节点 = 动态躲避
        """
        if len(self.nodes) <= 1:
            return

        positions = [Point2D(node.rel_x, node.rel_y) for node in self.nodes]
        adjusted_pos = self.checkCollision1(positions)

        # 强制锁定主节点不动
        self.nodes[0].rel_x = 0.0
        self.nodes[0].rel_y = 0.0

        # 只更新从节点
        for i in range(1, len(self.nodes)):
            self.nodes[i].rel_x = adjusted_pos[i].x
            self.nodes[i].rel_y = adjusted_pos[i].y

    # ====================== 新增：队形误差计算 ======================
    def _calculate_formation_error(self, node: UUVNode) -> float:
        """
        计算单个节点的队形误差
        误差定义：当前相对坐标 (rel_x, rel_y) 与目标坐标 (target_x, target_y) 的欧氏距离
        :param node: 节点实例
        :return: 误差距离（米）
        """
        if node.id == 0:
            return 0.0  # 主节点固定在中心，无队形误差
        dx = node.rel_x - node.target_x
        dy = node.rel_y - node.target_y
        return math.hypot(dx, dy)

    # ====================== 【核心】每一步记录变换数据 ======================
    def _record_transition_step(self):
        if not self.is_transition:
            return

        # 本步所有节点信息
        step_data = {
            "sim_time": round(self.current_time, 3),
            "from_formation": self.last_formation,
            "to_formation": self.config.formation_type,
            "nodes": []
        }

        max_error = 0.0
        for node in self.nodes:
            dx = node.rel_x - node.target_x
            dy = node.rel_y - node.target_y
            err = math.hypot(dx, dy)
            max_error = max(max_error, err)
            step_data["nodes"].append({
                "id": node.id,
                "lon": round(node.lon, 6),
                "lat": round(node.lat, 6),
                "speed": round(node.speed, 3),
                "heading": round(node.heading, 3),
                "rel_x": round(node.rel_x, 3),
                "rel_y": round(node.rel_y, 3),
                "target_x": round(node.target_x, 3),
                "target_y": round(node.target_y, 3),
                "formation_error": round(err, 4)
            })

        self.transition_data.append(step_data)

        # 误差稳定 → 结束变换，写入文件
        if max_error < ERROR_STABLE_THRESHOLD:
            self.is_transition = False
            with open("trans.json", "w", encoding="utf-8") as f:
                json.dump(self.transition_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 变换完成！共 {len(self.transition_data)} 步，已覆盖写入 trans.json")

    # ====================== 优化后的队形生成函数（适配3≤cnt≤9） ======================
    def _generate_formation_positions(self, cnt: int) -> List[Tuple[float, float]]:
        """
        生成指定队形的相对坐标（主节点固定在原点(0,0)）
        【约束】3 ≤ cnt ≤ 9（对应总节点4~10个）
        :param cnt: 从节点数量
        :return: 从节点坐标列表
        """
        # 入参校验
        if not (3 <= cnt <= 9):
            raise ValueError(f"从节点数量cnt必须满足 3 ≤ cnt ≤9（当前值：{cnt}）")
        
        d = self.config.rel_distance
        f_type = self.config.formation_type

        # 队形策略映射
        strategy_map: dict[str, Callable[[int, float], List[Tuple[float, float]]]] = {
            "line": self._generate_line,
            "rect": self._generate_rect,
            "circle": self._generate_circle,
            "diamond": self._generate_diamond,
            "triangle": self._generate_triangle
        }

        strategy = strategy_map.get(f_type)
        if not strategy:
            raise ValueError(f"不支持的队形类型：{f_type}")
        
        return strategy(cnt, d)

    # ====================== 具体队形生成策略（均适配3≤cnt≤9） ======================
    @staticmethod
    def _generate_line(cnt: int, d: float) -> List[Tuple[float, float]]:
        """竖直线队形：主节点(0,0)在最上方，从节点垂直向下等距排列"""
        return [(0.0, -i * d) for i in range(1, cnt + 1)]

    @staticmethod
    def _generate_rect(cnt: int, d: float) -> List[Tuple[float, float]]:
        """矩形队形：主节点(0,0)在最上方中心，从节点组成居中对称的矩形网格"""
        res = []
        # 针对3≤cnt≤9固定最优行列数
        row_col_map = {
            3: (2, 2), 4: (2, 2), 5: (2, 3), 6: (2, 3),
            7: (3, 3), 8: (3, 3), 9: (3, 3)
        }
        rows, cols = row_col_map[cnt]
        
        # 计算偏移量，保证网格整体居中
        x_offset = - (cols - 1) * d / 2.0
        y_offset = -d
        
        idx = 0
        for row in range(rows):
            current_y = y_offset - row * d
            for col in range(cols):
                if idx >= cnt:
                    break
                current_x = x_offset + col * d
                res.append((current_x, current_y))
                idx += 1
        return res

    @staticmethod
    def _generate_circle(cnt: int, d: float) -> List[Tuple[float, float]]:
        """圆形队形：主节点(0,0)在圆心，从节点均匀分布在圆周上"""
        circle_radius = d * 2.0
        res = []
        for i in range(cnt):
            angle = 2 * math.pi * i / cnt
            x = circle_radius * math.sin(angle)
            y = circle_radius * math.cos(angle)
            res.append((x, y))
        return res

    @staticmethod
    def _generate_diamond(cnt: int, d: float) -> List[Tuple[float, float]]:
        """菱形队形：主节点(0,0)在最上方，从节点按对称结构排布"""
        res = []
        # 针对3≤cnt≤9固定对称行结构
        row_nodes_map = {
            3: [2, 1], 4: [2, 2], 5: [2, 2, 1], 6: [2, 2, 2],
            7: [2, 3, 2], 8: [2, 4, 2], 9: [2, 4, 2, 1]
        }
        row_nodes_list = row_nodes_map[cnt]
        
        idx = 0
        for row_idx, nodes_in_row in enumerate(row_nodes_list):
            current_y = - (row_idx + 1) * d
            x_offset = - (nodes_in_row - 1) * d / 2.0
            for col_idx in range(nodes_in_row):
                if idx >= cnt:
                    break
                current_x = x_offset + col_idx * d
                res.append((current_x, current_y))
                idx += 1
        return res

    @staticmethod
    def _generate_triangle(cnt: int, d: float) -> List[Tuple[float, float]]:
        """正三角形队形：主节点(0,0)在最上方，从节点向下逐行递增1个节点"""
        res = []
        idx = 0
        row_idx = 1
        vertical_coeff = 1.2
        
        while idx < cnt:
            nodes_in_row = row_idx
            current_y = - row_idx * d * vertical_coeff
            x_offset = - (nodes_in_row - 1) * d / 2.0
            
            for col_idx in range(nodes_in_row):
                if idx >= cnt:
                    break
                current_x = x_offset + col_idx * d
                res.append((current_x, current_y))
                idx += 1
            row_idx += 1
        return res

    # ====================== 坐标转换函数 ======================
    def _geo2enu(self, lon, lat, ref_lon, ref_lat):
        """地理坐标（经纬度）转局部平面坐标（ENU东北天）"""
        lon_r = math.radians(lon)
        lat_r = math.radians(lat)
        r_lon_r = math.radians(ref_lon)
        r_lat_r = math.radians(ref_lat)
        x = R_EARTH * (lon_r - r_lon_r) * math.cos(r_lat_r)
        y = R_EARTH * (lat_r - r_lat_r)
        return x, y

    def _enu2geo(self, x, y, ref_lon, ref_lat):
        """局部平面坐标转回地理坐标（经纬度）"""
        r_lon_r = math.radians(ref_lon)
        r_lat_r = math.radians(ref_lat)
        d_lon = x / (R_EARTH * math.cos(r_lat_r))
        d_lat = y / R_EARTH
        return math.degrees(r_lon_r + d_lon), math.degrees(r_lat_r + d_lat)

    # ====================== 运动更新函数 ======================
    def _update_maneuver(self):
        """更新所有节点的运动状态、队形、避碰"""
        main = self.nodes[0]  # 主节点
        # 计算当前速度（加速度积分）
        current_v = self.config.init_speed + self.config.acceleration * self.current_time
        current_v = max(0.1, current_v)  # 最低速度限制

        # 计算转弯角度
        if self.config.turn_radius != float('inf'):
            omega = current_v / self.config.turn_radius
            turn_angle = omega * self.current_time
        else:
            turn_angle = 0.0

        # 更新主节点航向和速度
        main_hdg_rad = math.radians(self.config.init_heading) + turn_angle
        main.heading = math.degrees(main_hdg_rad) % 360.0
        main.speed = current_v

        # 主节点地理坐标更新
        dx = current_v * math.sin(main_hdg_rad) * self.config.sim_step
        dy = current_v * math.cos(main_hdg_rad) * self.config.sim_step
        main.lon, main.lat = self._enu2geo(dx, dy, main.lon, main.lat)

        # 核心逻辑：1.队形平滑过渡 2.实时避碰
        self._transition_formation()
        self.apply_collision_avoidance()

        # 更新所有从节点的运动参数和地理坐标
        for node in self.nodes[1:]:
            node.speed = current_v
            node.heading = main.heading
            # 坐标旋转变换（跟随主节点转弯）
            rx = node.rel_x * math.cos(turn_angle) - node.rel_y * math.sin(turn_angle)
            ry = node.rel_x * math.sin(turn_angle) + node.rel_y * math.cos(turn_angle)
            # 从节点地理坐标更新
            node.lon, node.lat = self._enu2geo(rx, ry, main.lon, main.lat)

    def step_simulation(self):
        """单步仿真：时间+1，更新所有状态，并定时输出节点信息"""
        self.current_time += self.config.sim_step
        self._update_maneuver()
        self._record_transition_step()  # 每步记录
        
        return self.nodes

# ====================== 可视化渲染类 ======================
class FormationVisualizer:
    def __init__(self, sim: UUVFormationSimulator):
        """
        可视化初始化
        :param sim: 编队仿真器实例
        """
        self.sim = sim
        self.fig, self.ax = plt.subplots(figsize=(8, 8))  # 创建画布
        self.ax.set_aspect('equal')  # 等比例坐标轴
        self.ax.set_title("UUV 集群编队实时仿真（含避碰）", fontsize=14)
        self.ax.grid(True, alpha=0.3)  # 显示网格

    def update_frame(self, frame):
        """动画帧更新函数：每帧重绘所有节点"""
        nodes = self.sim.step_simulation()  # 获取最新节点状态
        main = nodes[0]

        xs, ys, colors = [], [], []
        # 转换为相对主节点的坐标（主节点永远在中心）
        for node in nodes:
            dx, dy = self.sim._geo2enu(node.lon, node.lat, main.lon, main.lat)
            xs.append(dx)
            ys.append(dy)
            # 主节点绿色，从节点红色
            colors.append('#00FF00' if node.id == 0 else '#FF6666')

        # 清空画布并重绘
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        # 固定窗口范围，主节点永远在中心
        self.ax.set_xlim(-WINDOW_RANGE, WINDOW_RANGE)
        self.ax.set_ylim(-WINDOW_RANGE, WINDOW_RANGE)

        # 设置坐标轴标签
        self.ax.set_xlabel(f"东向 X (m) | 中心(主节点)", fontsize=12)
        self.ax.set_ylabel(f"北向 Y (m) | 中心(主节点)", fontsize=12)

        # 绘制节点
        self.ax.scatter(xs, ys, c=colors, s=100, alpha=1.0)
        # 为每个节点标注ID
        for i, node in enumerate(nodes):
            self.ax.annotate(f"ID{node.id}", (xs[i], ys[i]), fontsize=10, color='white',
                             bbox=dict(boxstyle="round", fc="black", alpha=0.7))

        # 显示队形和主节点坐标
        info = f"队形: {self.sim.config.formation_type} | 主节点: {main.lon:.6f}, {main.lat:.6f}"
        self.ax.text(0.01, 0.98, info, transform=self.ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle="round", facecolor='black', alpha=0.7), color='cyan')
        return self.ax,

    def start(self):
        """启动动画"""
        ani = FuncAnimation(self.fig, self.update_frame, interval=50, blit=False)
        plt.tight_layout()
        plt.show()

# ====================== 终端输入线程 ======================
def input_worker(sim: UUVFormationSimulator):
    """
    独立线程处理键盘输入，不阻塞动画渲染
    :param sim: 编队仿真器实例
    """
    print("\n===== 队形切换命令 =====")
    print("  1 = 竖直线  2 = 矩形  3 = 圆形  4 = 菱形  5 = 三角形")
    print("=========================\n")
    while True:
        try:
            cmd = input("输入命令切换队形：").strip()
            sim.switch_formation(cmd)
        except:
            continue

# ====================== 主函数 ======================
def main():
    """程序入口：初始化配置→启动仿真→启动输入线程→显示动画"""
    # 仿真参数配置（node_num=10，符合4≤node_num≤10约束）
    config = FormationConfig(
        formation_type="line",
        node_num=10,
        main_lon=120.0,
        main_lat=30.0,
        rel_distance=10.0,
        init_speed=2.0,
        init_heading=0.0,
        turn_radius=float('inf'),
        acceleration=0.0,
        sim_step=0.1,
        output_interval=1.0  # 设置每1秒输出一次状态
    )

    try:
        sim = UUVFormationSimulator(config)    # 创建仿真器
        viz = FormationVisualizer(sim)        # 创建可视化器

        # 启动输入线程（守护线程，程序退出时自动关闭）
        threading.Thread(target=input_worker, args=(sim,), daemon=True).start()
        viz.start()  # 启动可视化动画
    except ValueError as e:
        print(f"❌ 仿真启动失败：{e}")

# 程序入口
if __name__ == "__main__":
    main()