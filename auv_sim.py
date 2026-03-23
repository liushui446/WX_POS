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
        if not (4 <= self.config.node_num <= 10): raise ValueError(f"总节点数需4~10，当前：{self.config.node_num}")
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
        if not (3 <= cnt <=9): raise ValueError(f"从节点必须3~9，当前：{cnt}")
        d = self.config.rel_distance
        f = self.config.formation_type
        if f == "line": return [(0.0, -i*d) for i in range(1,cnt+1)]
        elif f == "rect":
            m = {3:(2,2),4:(2,2),5:(2,3),6:(2,3),7:(3,3),8:(3,3),9:(3,3)}
            rows,cols = m[cnt]; ox=-(cols-1)*d/2; oy=-d; r=[]; idx=0
            for i in range(rows):
                y = oy -i*d
                for j in range(cols):
                    if idx>=cnt:break
                    r.append((ox+j*d, y)); idx+=1
            return r
        elif f == "circle":
            r = d*2; return [(r*math.sin(2*math.pi*i/cnt), r*math.cos(2*math.pi*i/cnt)) for i in range(cnt)]
        elif f == "diamond":
            m = {3:[2,1],4:[2,2],5:[2,2,1],6:[2,2,2],7:[2,3,2],8:[2,4,2],9:[2,4,2,1]}
            rows = m[cnt]; r=[]; idx=0
            for i,n in enumerate(rows):
                y = -(i+1)*d; ox=-(n-1)*d/2
                for j in range(n):
                    if idx>=cnt:break
                    r.append((ox+j*d, y)); idx+=1
            return r
        elif f == "triangle":
            res = []
            idx = 0
            current_row = 2
            while idx < cnt:
                nodes_in_current_row = current_row
                y = -(current_row - 1) * d * 1.2
                x_offset = -(nodes_in_current_row - 1) * d / 2.0
                for col in range(nodes_in_current_row):
                    if idx >= cnt: break
                    x = x_offset + col * d
                    res.append((x, y))
                    idx += 1
                current_row += 1
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

    # ====================== 【核心修改】基于航向变化率的更新 ======================
    def _update_maneuver(self):
        main = self.nodes[0]
        dt = self.config.sim_step

        # 1. 更新主节点（使用 heading_rate 直接积分）
        current_v_main = self.config.init_speed + self.config.acceleration * self.current_time
        current_v_main = max(0.1, current_v_main)
        
        # 【修改】直接累加航向变化率
        # 新航向 = 旧航向 + 航向变化率 * 时间步长
        main.heading = (main.heading + self.config.heading_rate * dt) % 360.0
        main_hdg_rad = math.radians(main.heading)
        main.speed = current_v_main

        # 主节点位移增量
        dx_main_global = current_v_main * math.sin(main_hdg_rad) * dt
        dy_main_global = current_v_main * math.cos(main_hdg_rad) * dt
        main.lon, main.lat = self._enu2geo(dx_main_global, dy_main_global, main.lon, main.lat)

        # 2. 队形过渡 + 避碰
        self._transition_formation()
        self.apply_collision_avoidance()

        # 3. 运动学解算：更新从节点
        for node in self.nodes[1:]:
            # 3.1 计算相对速度
            v_rel_x = (node.rel_x - node.last_rel_x) / dt
            v_rel_y = (node.rel_y - node.last_rel_y) / dt

            # 3.2 坐标旋转
            v_rel_x_g = v_rel_x * math.cos(main_hdg_rad) - v_rel_y * math.sin(main_hdg_rad)
            v_rel_y_g = v_rel_x * math.sin(main_hdg_rad) + v_rel_y * math.cos(main_hdg_rad)

            # 3.3 速度合成
            v_main_x_g = current_v_main * math.sin(main_hdg_rad)
            v_main_y_g = current_v_main * math.cos(main_hdg_rad)
            
            v_abs_x = v_main_x_g + v_rel_x_g
            v_abs_y = v_main_y_g + v_rel_y_g

            # 3.4 计算期望的航速和航向
            desired_speed = math.hypot(v_abs_x, v_abs_y)
            desired_heading_rad = math.atan2(v_abs_x, v_abs_y)
            desired_heading = math.degrees(desired_heading_rad) % 360.0

            # 3.5 物理约束平滑
            desired_speed = min(desired_speed, MAX_SPEED)
            
            heading_diff = desired_heading - node.heading
            heading_diff = (heading_diff + 180) % 360 - 180
            
            max_turn_this_step = MAX_TURN_RATE * dt
            if abs(heading_diff) > max_turn_this_step:
                desired_heading = node.heading + math.copysign(max_turn_this_step, heading_diff)

            # 3.6 更新从节点状态
            node.speed = desired_speed
            node.heading = desired_heading % 360.0

            # 3.7 更新从节点地理坐标
            rx = node.rel_x * math.cos(main_hdg_rad) - node.rel_y * math.sin(main_hdg_rad)
            ry = node.rel_x * math.sin(main_hdg_rad) + node.rel_y * math.cos(main_hdg_rad)
            node.lon, node.lat = self._enu2geo(rx, ry, main.lon, main.lat)

            # 3.8 保存当前相对位置
            node.last_rel_x = node.rel_x
            node.last_rel_y = node.rel_y

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

# ====================== 主函数 ======================
def main():
    config = FormationConfig(
        formation_type="line",
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
        heading_rate=0.0, 
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