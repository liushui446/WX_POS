import math
import threading
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 全局基础参数 ======================
R_EARTH = 6378137.0
WINDOW_RANGE = 100
TRANSITION_SPEED = 0.08
COLLISION_RADIUS = 4.0
MAX_COLLISION_ITER = 15
MAX_ADJUST_STEP = 1.0
ERROR_STABLE_THRESHOLD = 0.02

# ====================== 运动学物理约束 ======================
MAX_SPEED = 5.0
MAX_TURN_RATE = 45.0

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
    is_joining: bool = False
    join_progress: float = 0.0
    join_total_frames: int = 0
    is_leaving: bool = False
    leave_target_x: float = 0.0
    leave_target_y: float = 0.0

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
        self.max_id = 0

        self.formation_map = {
            "1": "line", "2": "rect", "3": "circle", "4": "diamond", "5": "triangle"
        }
        self._validate_config()
        self._init_nodes()

    def _validate_config(self):
        if not (2 <= self.config.node_num <= 10): raise ValueError(f"总节点数需2~10，当前：{self.config.node_num}")
        if self.config.rel_distance <= 0: raise ValueError("节点间距必须>0")

    def _init_nodes(self):
        self.nodes.clear()
        self.max_id = 0
        main_node = UUVNode(0, self.config.main_lon, self.config.main_lat, self.config.init_speed, self.config.init_heading)
        self.nodes.append(main_node)
        for i in range(1, self.config.node_num):
            self.nodes.append(UUVNode(i, self.config.main_lon, self.config.main_lat, self.config.init_speed, self.config.init_heading))
            self.max_id = i
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
            print(f"\n✅ 切换队形：{self.last_formation} → {new_form}")

    def _set_target_formation(self):
        # 正常编队节点
        valid_slaves = [n for n in self.nodes[1:] if not n.is_leaving]
        slave_count = len(valid_slaves)
        positions = self._generate_formation_positions(slave_count)
        for i, node in enumerate(valid_slaves):
            if i < len(positions):
                node.target_x, node.target_y = positions[i]

        # ✅【关键】给脱离节点设置它自己的目标（相对于主机）
        for node in self.nodes[1:]:
            if node.is_leaving:
                node.target_x = node.leave_target_x
                node.target_y = node.leave_target_y
                # print("脱离点消失位置更新为：", node.target_x, node.target_y)

    # ====================== 【支持批量添加节点】核心修改 ======================
    def add_node(self, lon: float, lat: float, speed: float, heading: float, join_frames: int = 60):
        """ 单个添加（兼容旧接口） """
        if len(self.nodes) >= 10:
            print("❌ 已达到最大节点数10个，无法添加")
            return False

        self.max_id += 1
        main = self.nodes[0]
        rel_x, rel_y = self._geo2enu(lon, lat, main.lon, main.lat)
        new_node = UUVNode(
            id=self.max_id, lon=lon, lat=lat, speed=speed, heading=heading,
            rel_x=rel_x, rel_y=rel_y, is_joining=True,
            join_total_frames=join_frames, join_progress=0.0
        )
        self.nodes.append(new_node)
        print(f"\n✅ 成功添加节点 ID:{self.max_id}")
        return True

    def add_multiple_nodes(self, add_count: int = 1):
        """ 一次添加多个节点，自动判断上限 """
        current = len(self.nodes)
        max_allow = 10 - current
        if max_allow <= 0:
            print("❌ 已达最大节点数 10，无法添加")
            return

        add_count = min(add_count, max_allow)
        success = 0

        # 批量添加，位置稍微错开避免重叠
        base_lon = 120.001
        base_lat = 30.001
        for i in range(add_count):
            lon = base_lon + i * 0.0001
            lat = base_lat + i * 0.0001
            if self.add_node(lon, lat, 1.5, 90.0, 60):
                success += 1

        self.config.node_num = len(self.nodes)
        self._set_target_formation()
        self.is_transition = True
        print(f"✅ 批量添加完成：成功添加 {success} 个节点，当前总节点：{len(self.nodes)}")

    # ====================== 【支持多节点删除】 ======================
    def remove_random_node(self, remove_count: int = 1):
        """ 一次删除多个节点 """
        max_possible_remove = len(self.nodes) - 2  # 至少保留主节点+1个
        if max_possible_remove <= 0:
            print("❌ 节点数过少，无法删除")
            return

        # 实际可删除数量
        remove_count = min(remove_count, max_possible_remove)

        # 获取所有可删除节点
        slave_nodes = [node for node in self.nodes[1:] if not node.is_leaving]
        if len(slave_nodes) < remove_count:
            print("❌ 没有足够可删除的节点")
            return

        # 取最后 N 个节点开始脱离
        leave_nodes = slave_nodes[-remove_count:]
        main = self.nodes[0]
        leave_dist = self.config.rel_distance * 15.0

        for leave_node in leave_nodes:
            leave_node.is_leaving = True

            # ====================== 安全方向：主节点右侧 90° ======================
            opposite_heading = (main.heading + 90.0) % 360.0
            opposite_rad = math.radians(opposite_heading)
            leave_node.leave_target_x = leave_dist * math.sin(opposite_rad)
            leave_node.leave_target_y = leave_dist * math.cos(opposite_rad)

            print(f"\n✅ 节点 ID:{leave_node.id} 开始脱离")
            print(f"   目标相对位置: ({leave_node.leave_target_x:.1f}, {leave_node.leave_target_y:.1f})")

        self._set_target_formation()
        self.is_transition = True
        print(f"\n✅ 已启动 {remove_count} 个节点脱离流程，剩余节点重排中")

    def _transition_formation(self):
        for node in self.nodes[1:]:
            if node.is_joining:
                node.join_progress = min(1.0, node.join_progress + 1.0 / node.join_total_frames)
                node.rel_x += (node.target_x - node.rel_x) * 0.1
                node.rel_y += (node.target_y - node.rel_y) * 0.1
                if node.join_progress >= 1.0:
                    node.is_joining = False
                    print(f"✅ 节点 ID:{node.id} 已完全加入编队")
            else:
                if node.is_leaving:
                    node.rel_x += (node.target_x - node.rel_x) * TRANSITION_SPEED * 0.6
                    node.rel_y += (node.target_y - node.rel_y) * TRANSITION_SPEED * 0.6
                else:
                    node.rel_x += (node.target_x - node.rel_x) * TRANSITION_SPEED
                    node.rel_y += (node.target_y - node.rel_y) * TRANSITION_SPEED

    def checkCollision1(self, positions: List[Point2D]) -> List[Point2D]:
        adjusted = [Point2D(p.x, p.y) for p in positions]
        iter_num = 0
        has_collision = True
        num = len(adjusted)
        while has_collision and iter_num < MAX_COLLISION_ITER:
            has_collision = False
            pairs = []
            for i in range(num):
                for j in range(i+1, num):
                    d = (adjusted[i] - adjusted[j]).norm()
                    if d < COLLISION_RADIUS * 2.0:
                        pairs.append((d,i,j))
                        has_collision = True
            if not has_collision: break
            pairs.sort(key=lambda x:x[0])
            for d,i,j in pairs:
                dir = (adjusted[i] - adjusted[j]).normalized()
                step = min((COLLISION_RADIUS*2 - d)/2, MAX_ADJUST_STEP)
                adjusted[i] += dir * step
                adjusted[j] -= dir * step
            iter_num +=1
        return adjusted

    def apply_collision_avoidance(self):
        if len(self.nodes) <= 1: return
        ps = [Point2D(n.rel_x, n.rel_y) for n in self.nodes]
        adj = self.checkCollision1(ps)
        self.nodes[0].rel_x = 0
        self.nodes[0].rel_y = 0
        for i in range(1, len(self.nodes)):
            self.nodes[i].rel_x = adj[i].x
            self.nodes[i].rel_y = adj[i].y

    def _calculate_formation_error(self, node: UUVNode) -> float:
        if node.id == 0 or node.is_leaving or node.is_joining: return 0.0
        return math.hypot(node.rel_x - node.target_x, node.rel_y - node.target_y)

    def _record_transition_step(self):
        if not self.is_transition:
            return
        any_joining = any(n.is_joining for n in self.nodes)
        if any_joining:
            return

        max_err = 0.0
        for n in self.nodes:
            if n.id == 0 or n.is_leaving or n.is_joining:
                continue
            e = math.hypot(n.rel_x - n.target_x, n.rel_y - n.target_y)
            max_err = max(max_err, e)

        if max_err < ERROR_STABLE_THRESHOLD:
            has_leaving = any(n.is_leaving for n in self.nodes)
            if not has_leaving:
                self.is_transition = False
                print("✅ 编队已稳定")

    def _generate_formation_positions(self, cnt: int) -> List[Tuple[float, float]]:
        d = self.config.rel_distance
        f = self.config.formation_type
        if f == "line":
            return [(0.0, -i * d) for i in range(1, cnt+1)]
        elif f == "rect":
            num = cnt + 1
            r = []
            if num == 4:  r = [(d,0),(d,-d),(0,-d)]
            elif num ==5: r = [(-d,d),(-d,-d),(d,d),(d,-d)]
            elif num ==6: r = [(-d,0),(d,0),(-d,-d),(0,-d),(d,-d)]
            elif num ==7: r = [(-d,0),(d,0),(0,-d),(-d,-2*d),(0,-2*d),(d,-2*d)]
            elif num ==8: r = [(-d,0),(d,0),(-d,-d),(d,-d),(-d,-2*d),(0,-2*d),(d,-2*d)]
            elif num ==9: r = [(-d,0),(d,0),(-d,-d),(0,-d),(d,-d),(-d,-2*d),(0,-2*d),(d,-2*d)]
            elif num==10: r = [(-2*d,0),(-d,0),(d,0),(2*d,0),(-2*d,-d),(-d,-d),(0,-d),(d,-d),(2*d,-d)]
            return r
        elif f == "circle":
            r = d * 2.5
            return [(r*math.sin(2*math.pi*i/cnt+0.3), r*math.cos(2*math.pi*i/cnt+0.3)) for i in range(cnt)]
        elif f == "diamond":
            num = cnt+1
            r = []
            if num==4: r = [(-d,0),(0,-d),(d,0)]
            elif num==5: r = [(0,d),(-d,0),(d,0),(0,-d)]
            elif num==6: r = [(0,d),(-d,0),(d,0),(0,-d),(0,-2*d)]
            elif num==7: r = [(0,-d),(-d,-2*d),(0,-2*d),(d,-2*d),(0,-3*d),(0,-4*d)]
            elif num==8: r = [(-d/2,-d),(d/2,-d),(-d,-2*d),(d,-2*d),(-d/2,-3*d),(d/2,-3*d),(0,-4*d)]
            elif num==9: r = [(-d/2,-d),(d/2,-d),(-d,-2*d),(0,-2*d),(d,-2*d),(-d/2,-3*d),(d/2,-3*d),(0,-4*d)]
            elif num==10:r = [(-d/2,-d),(d/2,-d),(-1.5*d,-2*d),(-0.5*d,-2*d),(0.5*d,-2*d),(1.5*d,-2*d),(-d/2,-3*d),(d/2,-3*d),(0,-4*d)]
            return r
        elif f == "triangle":
            num = cnt+1
            r = []
            if num==3:  r = [(-d,-d),(d,-d)]
            elif num==4: r = [(0,d),(-d,-d),(d,-d)]
            elif num==5: r = [(-d/2,-d),(d/2,-d),(-d,-2*d),(d,-2*d)]
            elif num==6: r = [(-d/2,-d),(d/2,-d),(-d,-2*d),(0,-2*d),(d,-2*d)]
            elif num==7: r = [(-d/2,-d),(d/2,-d),(-1.5*d,-2*d),(-0.5*d,-2*d),(0.5*d,-2*d),(1.5*d,-2*d)]
            elif num==8: r = [(-d,-d),(0,-d),(d,-d),(-1.5*d,-2*d),(-0.5*d,-2*d),(0.5*d,-2*d),(1.5*d,-2*d)]
            elif num==9: r = [(-d,-d),(0,-d),(d,-d),(-2*d,-2*d),(-d,-2*d),(0,-2*d),(d,-2*d),(2*d,-2*d)]
            elif num==10:r= [(-d/2,-d),(d/2,-d),(-d,-2*d),(0,-2*d),(d,-2*d),(-1.5*d,-3*d),(-0.5*d,-3*d),(0.5*d,-3*d),(1.5*d,-3*d)]
            return r
        return []

    def _geo2enu(self, lon, lat, rlon, rlat):
        lr = math.radians(lon)
        la = math.radians(lat)
        rlr = math.radians(rlon)
        rla = math.radians(rlat)
        x = R_EARTH * (lr - rlr) * math.cos(rla)
        y = R_EARTH * (la - rla)
        return x,y

    def _enu2geo(self, x,y,rlon,rlat):
        rlr = math.radians(rlon)
        rla = math.radians(rlat)
        dlon = x/(R_EARTH*math.cos(rla))
        dlat = y/R_EARTH
        return math.degrees(rlr+dlon), math.degrees(rla+dlat)

    def _update_maneuver(self):
        main = self.nodes[0]
        dt = self.config.sim_step
        current_v_main = self.config.init_speed + self.config.acceleration * self.current_time
        current_v_main = max(0.1, current_v_main)
        main.heading = (main.heading + self.config.heading_rate * dt) % 360.0
        main_hdg_rad = math.radians(main.heading)
        main.speed = current_v_main
        dx_main = current_v_main * math.sin(main_hdg_rad) * dt
        dy_main = current_v_main * math.cos(main_hdg_rad) * dt
        main.lon, main.lat = self._enu2geo(dx_main, dy_main, main.lon, main.lat)

        if self.is_transition:
            self._transition_formation()
        self.apply_collision_avoidance()
        w = math.radians(self.config.heading_rate)

        # ==============================================
        # 【支持多删除】倒序遍历 + 批量删除
        # ==============================================
        nodes_to_remove = []
        for node in reversed(self.nodes[1:]):
            if node.is_leaving:
                closest_node = None
                min_dist = float('inf')
                for n in self.nodes[1:]:
                    if n.is_leaving: continue
                    d = math.hypot(n.rel_x - node.rel_x, n.rel_y - node.rel_y)
                    if d < min_dist:
                        min_dist = d
                        closest_node = n

                if closest_node is not None:
                    current_dist = math.hypot(closest_node.rel_x - node.rel_x, closest_node.rel_y - node.rel_y)
                    delete_dist = 5 * self.config.rel_distance
                    if current_dist > delete_dist:
                        print(f"✅ 节点 ID:{node.id} 已远离编队，消失")
                        nodes_to_remove.append(node)

            # 正常节点
            rx, ry = node.rel_x, node.rel_y
            if abs(w) < 1e-4:
                dvx = current_v_main * math.sin(main_hdg_rad)
                dvy = current_v_main * math.cos(main_hdg_rad)
            else:
                vrx = -w * ry
                vry = w * rx
                dvx = current_v_main * math.sin(main_hdg_rad) + vrx
                dvy = current_v_main * math.cos(main_hdg_rad) + vry

            node.speed = min(math.hypot(dvx, dvy), MAX_SPEED)
            node.heading = math.degrees(math.atan2(dvx, dvy)) % 360.0
            wx = rx * math.cos(main_hdg_rad) - ry * math.sin(main_hdg_rad)
            wy = rx * math.sin(main_hdg_rad) + ry * math.cos(main_hdg_rad)
            node.lon, node.lat = self._enu2geo(wx, wy, main.lon, main.lat)

        # 批量删除（安全）
        for node in nodes_to_remove:
            if node in self.nodes:
                self.nodes.remove(node)
        self.config.node_num = len(self.nodes)
        if nodes_to_remove:
            self._set_target_formation()

    def step_simulation(self):
        self.current_time += self.config.sim_step
        self._update_maneuver()
        self._record_transition_step()
        return self.nodes

# ====================== 可视化 ======================
class FormationVisualizer:
    def __init__(self, sim: UUVFormationSimulator):
        self.sim = sim
        self.fig, self.ax = plt.subplots(figsize=(9,9))
        self.ax.set_aspect('equal')
        self.ax.set_title("UUV 集群编队仿真", fontsize=14)
        self.ax.grid(True, alpha=0.3)

    def update_frame(self, frame):
        nodes = self.sim.step_simulation()
        main = nodes[0]
        xs, ys, cs = [], [], []

        for n in nodes:
            dx, dy = self.sim._geo2enu(n.lon, n.lat, main.lon, main.lat)
            xs.append(dx)
            ys.append(dy)
            if n.id == 0:
                cs.append('#00ff00')
            elif n.is_leaving:
                cs.append('#ff0000')
            elif n.is_joining:
                cs.append('#0000ff')
            else:
                cs.append('#ff6666')

        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-WINDOW_RANGE, WINDOW_RANGE)
        self.ax.set_ylim(-WINDOW_RANGE, WINDOW_RANGE)
        self.ax.set_xlabel("东向 X (m)", fontsize=12)
        self.ax.set_ylabel("北向 Y (m)", fontsize=12)

        self.ax.scatter(xs, ys, c=cs, s=100)

        for n, x, y in zip(nodes, xs, ys):
            t = f"ID{n.id}\n{n.heading:.0f}°\n{n.speed:.1f}"
            if n.is_leaving:
                t += "\n🚀脱离中"
            if n.is_joining:
                t += f"\n🔵加入中\n{n.join_progress:.0%}"
            self.ax.annotate(t, (x, y), fontsize=8, color='white',
                            bbox=dict(boxstyle="round", fc="black", alpha=0.7))

        info = (f"队形:{self.sim.config.formation_type}  总节点:{len(self.sim.nodes)}  "
                f"脱离中:{len([n for n in self.sim.nodes if n.is_leaving])}  "
                f"航向率:{self.sim.config.heading_rate:.1f}°/s")
        self.ax.text(0.01, 0.98, info, transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor='black', alpha=0.7), color='cyan')
        return self.ax,

    def start(self):
        ani = FuncAnimation(self.fig, self.update_frame, interval=50)
        plt.tight_layout()
        plt.show()

# ====================== 命令输入 ======================
def input_worker(sim: UUVFormationSimulator):
    print("\n===== 命令面板 =====")
    print("1-5     : 切换队形")
    print("add     : 增加1个节点")
    print("add2    : 增加2个节点")
    print("add3    : 增加3个节点")
    print("remove  : 删除1个节点")
    print("remove2 : 删除2个节点")
    print("remove3 : 删除3个节点")
    print("====================\n")
    while True:
        try:
            cmd = input(">> 输入命令：").strip().lower()
            if cmd in "12345":
                sim.switch_formation(cmd)
            elif cmd == "add":
                sim.add_multiple_nodes(1)
            elif cmd == "add2":
                sim.add_multiple_nodes(2)
            elif cmd == "add3":
                sim.add_multiple_nodes(3)
            elif cmd == "remove":
                sim.remove_random_node(1)
            elif cmd == "remove2":
                sim.remove_random_node(2)
            elif cmd == "remove3":
                sim.remove_random_node(3)
        except Exception as e:
            print(f"❌ 错误：{e}")

# ====================== 主函数 ======================
def main():
    config = FormationConfig(
        formation_type="line",
        old_num=8,
        node_num=6,
        main_lon=120.0,
        main_lat=30.0,
        rel_distance=10.0,
        init_speed=2.0,
        init_heading=180.0,
        heading_rate=1.0,
        acceleration=0.0,
        sim_step=0.1
    )
    sim = UUVFormationSimulator(config)
    viz = FormationVisualizer(sim)
    threading.Thread(target=input_worker, args=(sim,), daemon=True).start()
    viz.start()

if __name__ == "__main__":
    main()