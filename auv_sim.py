import math
import threading
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Callable

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 全局基础参数 ======================
R_EARTH = 6378137.0
WINDOW_RANGE = 80
TRANSITION_SPEED = 0.08
COLLISION_RADIUS = 4.0
MAX_COLLISION_ITER = 15
MAX_ADJUST_STEP = 1.0
ERROR_STABLE_THRESHOLD = 0.02

# ====================== 运动学物理约束 ======================
MAX_SPEED = 5.0
MAX_TURN_RATE = 45.0     # 物理上的最大转向速率限制 (度/秒)

# ====================== 数据结构定义 ======================
@dataclass
class Point2D:
    x: float
    y: float
    def __sub__(self, other: 'Point2D') -> 'Point2D': return Point2D(self.x - other.x, self.y - other.y)
    def __add__(self, other: 'Point2D') -> 'Point2D': return Point2D(self.x + other.x, self.y + other.y)
    def __mul__(self, scalar: float) -> 'Point2D': return Point2D(self.x * scalar, self.y * scalar)
    __rmul__ = __mul__
    def __truediv__(self, scalar: float) -> 'Point2D': return Point2D(self.x / scalar, self.y / scalar)
    def norm(self) -> float: return math.hypot(self.x, self.y)
    def normalized(self) -> 'Point2D':
        n = self.norm()
        return Point2D(0,0) if n < 1e-6 else Point2D(self.x/n, self.y/n)

@dataclass
class UUVNode:
    id: int
    lon: float
    lat: float
    speed: float
    heading: float
    rel_x: float = 0.0
    rel_y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    last_rel_x: float = 0.0
    last_rel_y: float = 0.0

@dataclass
class FormationConfig:
    formation_type: str
    old_num: int
    node_num: int
    main_lon: float
    main_lat: float
    rel_distance: float
    init_speed: float
    init_heading: float
    # ====================== 【修改】将 turn_radius 改为 heading_rate ======================
    # heading_rate: 航向变化率 (度/秒)
    # 正数 = 逆时针左转 (Counter-Clockwise)
    # 负数 = 顺时针右转 (Clockwise)
    # 0 = 直线航行
    heading_rate: float = 0.0      
    acceleration: float = 0.0
    sim_step: float = 0.1
    output_interval: float = 1.0

# ====================== 编队仿真核心类 ======================
class UUVFormationSimulator:
    def __init__(self, config: FormationConfig):
        self.config = config
        self.nodes: List[UUVNode] = []
        self.current_time = 0.0
        self.last_output_time = 0.0
        self.is_transition = False
        self.last_formation = config.formation_type
        self.transition_data = []

        self.formation_map = {
            "1": "line", "2": "rect", "3": "circle", "4": "diamond", "5": "triangle"
        }
        self._validate_config()
        self._init_nodes()

    def _validate_config(self):
        if not (2 <= self.config.node_num <= 10): raise ValueError(f"总节点数需2~10，当前：{self.config.node_num}")
        if self.config.rel_distance <= 0: raise ValueError("节点间距必须>0")

    def _init_nodes(self):
        main_node = UUVNode(0, self.config.main_lon, self.config.main_lat,
                           self.config.init_speed, self.config.init_heading)
        self.nodes.append(main_node)
        for i in range(1, self.config.node_num):
            self.nodes.append(UUVNode(i, self.config.main_lon, self.config.main_lat,
                                      self.config.init_speed, self.config.init_heading))
        self._set_target_formation()
        self._set_initial_position()
        for node in self.nodes:
            node.last_rel_x = node.rel_x
            node.last_rel_y = node.rel_y

    def _set_initial_position(self):
        self.nodes[0].rel_x = 0.0
        self.nodes[0].rel_y = 0.0
        for node in self.nodes[1:]:
            node.rel_x = node.target_x
            node.rel_y = node.target_y

    def switch_formation(self, cmd: str):
        if cmd in self.formation_map:
            new_form = self.formation_map[cmd]
            if new_form == self.config.formation_type:
                print("✅ 当前已是该队形")
                return
            self.last_formation = self.config.formation_type
            self.config.formation_type = new_form
            self._set_target_formation()
            self.is_transition = True
            self.transition_data = []
            print(f"\n✅ 切换队形：{self.last_formation} → {new_form}，开始记录...")

    # ====================== 【关键修改】智能切换：同数平滑 / 异数跳变 ======================
    def switch_formation_and_count(self, formation_name: str, new_count: int):
        valid_forms = ["line", "rect", "circle", "diamond", "triangle"]
        if formation_name not in valid_forms:
            print(f"❌ 无效队形：{formation_name}")
            return
        if not (3 <= new_count <= 10):
            print(f"❌ 节点数必须 4~10")
            return

        self.old_num = self.config.node_num
        self.last_formation = self.config.formation_type
        self.config.formation_type = formation_name
        self.config.node_num = new_count

        # ====================== 核心逻辑 ======================
        if new_count == self.old_num:
            # 节点数相同 → 平滑过渡
            self.switch_formation(str(new_count))
            print(f"✅ 切换：{self.last_formation} → {formation_name} | 节点数不变，平滑过渡")
        else:
            # 节点数不同 → 直接到达目标位置
            # 重新初始化节点
            self._init_nodes()
            print(f"✅ 切换：{self.last_formation} → {formation_name} | 节点数变化，直接就位")
        # ======================================================
    
    def _set_target_formation(self):
        slave_count = len(self.nodes) - 1
        if slave_count <= 0: return
        positions = self._generate_formation_positions(slave_count)
        for i, node in enumerate(self.nodes[1:]):
            if i < len(positions):
                node.target_x, node.target_y = positions[i]

    def _transition_formation(self):
        for node in self.nodes[1:]:
            node.rel_x += (node.target_x - node.rel_x) * TRANSITION_SPEED
            node.rel_y += (node.target_y - node.rel_y) * TRANSITION_SPEED

    def checkCollision1(self, positions: List[Point2D]) -> List[Point2D]:
        adjusted = positions.copy()
        iter_num = 0
        has_collision = True
        num_uavs = len(adjusted)
        while has_collision and iter_num < MAX_COLLISION_ITER:
            has_collision = False
            collision_pairs = []
            for i in range(num_uavs):
                for j in range(i+1, num_uavs):
                    diff = adjusted[i] - adjusted[j]
                    dis = diff.norm()
                    if dis <= COLLISION_RADIUS*2:
                        collision_pairs.append((dis,i,j))
                        has_collision = True
            if not has_collision: break
            collision_pairs.sort(key=lambda x:x[0])
            for dis,i,j in collision_pairs:
                diff = adjusted[i]-adjusted[j]
                dir_vec = diff.normalized()
                adj = min((COLLISION_RADIUS*2 - dis)/2, MAX_ADJUST_STEP)
                if i ==0: adjusted[j] -= dir_vec*adj
                elif j ==0: adjusted[i] += dir_vec*adj
                else:
                    adjusted[i] += dir_vec*adj
                    adjusted[j] -= dir_vec*adj
            iter_num +=1
        return adjusted

    def apply_collision_avoidance(self):
        if len(self.nodes)<=1:return
        ps = [Point2D(n.rel_x, n.rel_y) for n in self.nodes]
        adj = self.checkCollision1(ps)
        self.nodes[0].rel_x=0; self.nodes[0].rel_y=0
        for i in range(1,len(self.nodes)):
            self.nodes[i].rel_x = adj[i].x
            self.nodes[i].rel_y = adj[i].y

    def _calculate_formation_error(self, node: UUVNode) -> float:
        if node.id == 0: return 0.0
        return math.hypot(node.rel_x - node.target_x, node.rel_y - node.target_y)

    def _record_transition_step(self):
        if not self.is_transition: return

        step_data = {
            "sim_time": round(self.current_time, 3),
            "from_formation": self.last_formation,
            "to_formation": self.config.formation_type,
            "nodes": []
        }

        max_error = 0.0
        for node in self.nodes:
            err = self._calculate_formation_error(node)
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

        if max_error < ERROR_STABLE_THRESHOLD:
            self.is_transition = False
            with open("trans.json", "w", encoding="utf-8") as f:
                json.dump(self.transition_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 变换完成！共 {len(self.transition_data)} 步，已写入 trans.json")

    def _generate_formation_positions(self, cnt: int) -> List[Tuple[float, float]]:
        if not (3 <= (cnt+1) <=10): raise ValueError(f"从节点必须3~10，当前：{cnt}")
        d = self.config.rel_distance
        f = self.config.formation_type
        if f == "line": return [(0.0, -i*d) for i in range(1,cnt+1)]
        elif f == "rect":
            r = []
            num = cnt+1
            if num == 4:
                r = [(d, 0), (d, -d), (0, -d)]
            elif num == 5:
                r = [(-d, d), (-d, -d), (d, d), (d, -d)]
            elif num == 6:
                r = [(-d, 0), (d, 0), (-d, -d), (0, -d), (d, -d)]
            elif num == 7:
                r = [(-d, 0), (d, 0), (0, -d), (-d, -2*d), (0, -2*d), (d, -2*d)]
            elif num == 8:
                r = [(-d, 0), (d, 0), (-d, -d), (d, -d), (-d, -2*d), (0, -2*d), (d, -2*d)]
            elif num == 9:
                r = [(-d, 0), (d, 0), (-d, -d), (0, -d), (d, -d), (-d, -2*d), (0, -2*d), (d, -2*d)]
            elif num == 10:
                r = [(-2*d,0),(-d,0),(d,0),(2*d,0), (-2*d,-d),(-d,-d),(0,-d),(d,-d),(2*d,-d)]
            return r
        elif f == "circle":
            r = d*2; return [(r*math.sin(2*math.pi*i/cnt), r*math.cos(2*math.pi*i/cnt)) for i in range(cnt)]
        elif f == "diamond":
            r = []
            d = self.config.rel_distance
            num = cnt+1
            # cnt = 从节点数量，总节点数 = cnt+1
            if num == 4:
                # 总4节点：主在上顶点 → 从：左、下、右
                r = [
                    (-d, 0),    # 左
                    (0, -d),    # 下
                    (d, 0)      # 右
                ]
            elif num == 5:
                # 总5节点：主在中心 → 从：上、左、右、下
                r = [
                    (0, d),     # 上
                    (-d, 0),    # 左
                    (d, 0),     # 右
                    (0, -d)     # 下
                ]
            elif num == 6:
                # 总6节点：排1(主1)、排2(4)、排3(1) → 从5个
                r = [
                    (-2*d, 0), (-d, 0), (d, 0), (2*d, 0),  # 排2
                    (0, -d)                                # 排3
                ]
            elif num == 7:
                # 总7节点：排1(主1)、排2(1)、排3(3)、排4(1)、排5(1) → 从6个
                r = [
                    (0, -d),                   # 排2
                    (-d, -2*d), (0, -2*d), (d, -2*d),  # 排3
                    (0, -3*d),                 # 排4
                    (0, -4*d)                  # 排5
                ]
            elif num == 8:
                # 总8节点：排1(主1)、排2(2)、排3(2)、排4(2)、排5(1) → 从7个
                r = [
                    (-d/2, -d), (d/2, -d),     # 排2
                    (-d, -2*d), (d, -2*d),     # 排3
                    (-d/2, -3*d), (d/2, -3*d), # 排4
                    (0, -4*d)                  # 排5
                ]
            elif num == 9:
                # 总9节点：排1(主1)、排2(2)、排3(3)、排4(2)、排5(1) → 从8个
                r = [
                    (-d/2, -d), (d/2, -d),            # 排2
                    (-d, -2*d), (0, -2*d), (d, -2*d), # 排3
                    (-d/2, -3*d), (d/2, -3*d),        # 排4
                    (0, -4*d)                         # 排5
                ]
            elif num == 10:
                # 总10节点：排1(主1)、排2(2)、排3(4)、排4(2)、排5(1) → 从9个
                r = [
                    (-d/2, -d), (d/2, -d),                      # 排2
                    (-1.5*d, -2*d), (-0.5*d, -2*d), (0.5*d, -2*d), (1.5*d, -2*d), # 排3
                    (-d/2, -3*d), (d/2, -3*d),                  # 排4
                    (0, -4*d)                                   # 排5
                ]
            return r
        elif f == "triangle":
            res = []
            d = self.config.rel_distance
            num = cnt+1
            # cnt = 从节点数量（总节点数 = cnt+1）
            if num == 3:
                # 总节点3个：主在上顶点 → 从：左下、右下
                res = [
                    (-d, -d),       # 左下
                    (d, -d)         # 右下
                ]
            elif num == 4:
                # 总节点4个：主在中心 → 从：上、左下、右下
                res = [
                    (0, d),         # 上顶点
                    (-d, -d),       # 左下
                    (d, -d)         # 右下
                ]
            elif num == 5:
                # 总节点5个：主中心 → 上、左下、右下 + 第2排中间1个
                res = [
                    (-d/2, -d), (d/2, -d),            # 排2
                    (-d, -2*d), (d, -2*d) # 排3   
                ]
            elif num == 6:
                # 总节点6个：排1(1主)、排2(2)、排3(3) → 从共5个
                res = [
                    (-d/2, -d), (d/2, -d),            # 排2
                    (-d, -2*d),  (0, -2*d), (d, -2*d) # 排3
                ]
            elif num == 7:
                # 总节点7个：排1(1)、排2(2)、排3(4) → 从共6个
                res = [
                    (-d/2, -d), (d/2, -d),                      # 排2
                    (-1.5*d, -2*d), (-0.5*d, -2*d), (0.5*d, -2*d), (1.5*d, -2*d) # 排3
                ]
            elif num == 8:
                # 总节点8个：排1(1)、排2(3)、排3(4) → 从共7个
                res = [
                    (-d, -d),  (0, -d),  (d, -d),              # 排2
                    (-1.5*d, -2*d), (-0.5*d, -2*d), (0.5*d, -2*d), (1.5*d, -2*d) # 排3
                ]
            elif num == 9:
                # 总节点9个：排1(1)、排2(3)、排3(5) → 从共8个
                res = [
                    (-d, -d),  (0, -d),  (d, -d),              # 排2
                    (-2*d, -2*d), (-d, -2*d), (0, -2*d), (d, -2*d), (2*d, -2*d) # 排3
                ]
            elif num == 10:
                # 总节点10个：4排：1 + 2 + 3 + 4 → 从共9个
                res = [
                    (-d/2, -d), (d/2, -d),            # 排2
                    (-d, -2*d),  (0, -2*d), (d, -2*d), # 排3
                    (-1.5*d, -3*d), (-0.5*d, -3*d), (0.5*d, -3*d), (1.5*d, -3*d) # 排4
                ]
            return res
        return []

    def _geo2enu(self, lon, lat, rlon, rlat):
        lr = math.radians(lon); la = math.radians(lat)
        rlr = math.radians(rlon); rla = math.radians(rlat)
        x = R_EARTH * (lr-rlr) * math.cos(rla)
        y = R_EARTH * (la-rla)
        return x,y

    def _enu2geo(self, x,y,rlon,rlat):
        rlr = math.radians(rlon); rla = math.radians(rlat)
        dlon = x/(R_EARTH*math.cos(rla))
        dlat = y/R_EARTH
        return math.degrees(rlr+dlon), math.degrees(rla+dlat)

    # ====================== 【核心修改】真实编队圆弧转弯 + 整体旋转 ======================
    def _update_maneuver(self):
        main = self.nodes[0]
        dt = self.config.sim_step

        # 1. 更新主节点
        current_v_main = self.config.init_speed + self.config.acceleration * self.current_time
        current_v_main = max(0.1, current_v_main)
        
        # 航向积分
        main.heading = (main.heading + self.config.heading_rate * dt) % 360.0
        main_hdg_rad = math.radians(main.heading)
        main.speed = current_v_main

        # 主节点位移
        dx_main = current_v_main * math.sin(main_hdg_rad) * dt
        dy_main = current_v_main * math.cos(main_hdg_rad) * dt
        main.lon, main.lat = self._enu2geo(dx_main, dy_main, main.lon, main.lat)

        # 2. 队形渐变 + 避碰
        if self.is_transition:
            self._transition_formation()
        self.apply_collision_avoidance()

        # 3. 角速度（弧度/秒）
        yaw_rate_deg = self.config.heading_rate
        w = math.radians(yaw_rate_deg)

        # 4. 逐个从节点：速度分配 + 位置旋转（整体队形倾斜）
        for node in self.nodes[1:]:
            rx = node.rel_x  # 相对主节点的机体坐标系坐标
            ry = node.rel_y

            # ---------------------------
            # 关键：转弯时内外侧速度分配
            # ---------------------------
            if abs(w) < 1e-4:
                # 直行：速度 = 主节点速度
                des_vx = current_v_main * math.sin(main_hdg_rad)
                des_vy = current_v_main * math.cos(main_hdg_rad)
            else:
                # 转弯：外侧快、内侧慢
                v_rel_x = -w * ry
                v_rel_y =  w * rx
                des_vx = current_v_main * math.sin(main_hdg_rad) + v_rel_x
                des_vy = current_v_main * math.cos(main_hdg_rad) + v_rel_y

            # 期望航速、航向
            desired_speed = math.hypot(des_vx, des_vy)
            desired_heading = math.degrees(math.atan2(des_vx, des_vy)) % 360.0

            # 限幅
            desired_speed = min(desired_speed, MAX_SPEED)

            # 更新节点速度与航向
            node.speed = desired_speed
            node.heading = desired_heading

            # ---------------------------
            # 关键：整体队形旋转（倾斜）
            # 把相对坐标旋转到主节点当前航向 → 实现编队倾斜
            # ---------------------------
            x_world = rx * math.cos(main_hdg_rad) - ry * math.sin(main_hdg_rad)
            y_world = rx * math.sin(main_hdg_rad) + ry * math.cos(main_hdg_rad)

            # 用旋转后的全局相对位置计算经纬
            node.lon, node.lat = self._enu2geo(x_world, y_world, main.lon, main.lat)

            # 保存上一帧位置
            node.last_rel_x = rx
            node.last_rel_y = ry

    def step_simulation(self):
        self.current_time += self.config.sim_step
        self._update_maneuver()
        self._record_transition_step()
        return self.nodes

# ====================== 可视化渲染类 ======================
class FormationVisualizer:
    def __init__(self, sim: UUVFormationSimulator):
        self.sim = sim
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal')
        self.ax.set_title("UUV 集群编队（航向变化率控制）", fontsize=14)
        self.ax.grid(True, alpha=0.3)

    def update_frame(self, frame):
        nodes = self.sim.step_simulation()
        main = nodes[0]

        xs, ys, colors = [], [], []
        for node in nodes:
            dx, dy = self.sim._geo2enu(node.lon, node.lat, main.lon, main.lat)
            xs.append(dx)
            ys.append(dy)
            colors.append('#00FF00' if node.id == 0 else '#FF6666')

        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-WINDOW_RANGE, WINDOW_RANGE)
        self.ax.set_ylim(-WINDOW_RANGE, WINDOW_RANGE)
        self.ax.set_xlabel("东向 X (m) | 中心(主节点)", fontsize=12)
        self.ax.set_ylabel("北向 Y (m) | 中心(主节点)", fontsize=12)

        self.ax.scatter(xs, ys, c=colors, s=100, alpha=1.0)
        
        for i, node in enumerate(nodes):
            label = f"ID{node.id}\n{node.heading:.0f}°\n{node.speed:.1f}m/s"
            self.ax.annotate(label, (xs[i], ys[i]), fontsize=8, color='white',
                             bbox=dict(boxstyle="round", fc="black", alpha=0.7))

        # 显示当前的航向变化率
        info = f"队形: {self.sim.config.formation_type} | 航向变化率: {self.sim.config.heading_rate:.1f}°/s"
        self.ax.text(0.01, 0.98, info, transform=self.ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle="round", facecolor='black', alpha=0.7), color='cyan')
        return self.ax,

    def start(self):
        ani = FuncAnimation(self.fig, self.update_frame, interval=50, blit=False)
        plt.tight_layout()
        plt.show()

def input_worker(sim: UUVFormationSimulator):
    print("\n===== 队形切换命令 =====")
    print("  1 = 竖直线  2 = 矩形  3 = 圆形  4 = 菱形  5 = 三角形")
    print("=========================\n")
    while True:
        try:
            cmd = input("输入命令切换队形：").strip()
            sim.switch_formation(cmd)
        except:
            continue

# def input_worker(sim: UUVFormationSimulator):
#     print("\n===== 编队切换命令（格式：队形 节点数）=====")
#     print("示例：")
#     print("  rect 6    → 矩形6节点")
#     print("  line 8    → 直线8节点")
#     print("可选：line, rect, circle, diamond, triangle")
#     print("=============================================\n")
    
#     while True:
#         try:
#             line = input(">> 输入命令：").strip()
#             if not line: continue
#             parts = line.split()
#             if len(parts) != 2:
#                 print("❌ 格式：队形 节点数")
#                 continue
#             form_name = parts[0].lower()
#             new_cnt = int(parts[1])
#             sim.switch_formation_and_count(form_name, new_cnt)
#         except ValueError:
#             print("❌ 节点数必须是数字")
#         except Exception as e:
#             print(f"❌ 错误：{e}")

# ====================== 主函数 ======================
def main():
    config = FormationConfig(
        formation_type="line",
        old_num=8,
        node_num=8,
        main_lon=120.0,
        main_lat=30.0,
        rel_distance=10.0,
        init_speed=2.0,
        init_heading=0.0,
        # ====================== 【关键配置】航向变化率 ======================
        # 0.0 = 直线航行
        # 5.0 = 以每秒5度的速率逆时针左转
        # -5.0 = 以每秒5度的速率顺时针右转
        heading_rate=1.0, 
        acceleration=0.0,
        sim_step=0.1
    )

    try:
        sim = UUVFormationSimulator(config)
        viz = FormationVisualizer(sim)
        threading.Thread(target=input_worker, args=(sim,), daemon=True).start()
        viz.start()
    except ValueError as e:
        print(f"❌ 仿真启动失败：{e}")

if __name__ == "__main__":
    main()