import math
import threading
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 全局基础参数 ======================
R_EARTH = 6378137.0
WINDOW_RANGE = 150
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

# ====================== 编队仿真核心类（单编队，带ID） ======================
class UUVFormationSimulator:
    def __init__(self, config: FormationConfig, formation_id: int):
        self.formation_id = formation_id  # 编队唯一ID
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
                print(f"✅ 编队[{self.formation_id}] 当前已是该队形")
                return
            self.last_formation = self.config.formation_type
            self.config.formation_type = new_form
            self._set_target_formation()
            self.is_transition = True
            self.transition_data = []
            print(f"\n✅ 编队[{self.formation_id}] 切换队形：{self.last_formation} → {new_form}")

    def _set_target_formation(self):
        valid_slaves = [n for n in self.nodes[1:] if not n.is_leaving]
        slave_count = len(valid_slaves)
        positions = self._generate_formation_positions(slave_count)
        for i, node in enumerate(valid_slaves):
            if i < len(positions):
                node.target_x, node.target_y = positions[i]

        for node in self.nodes[1:]:
            if node.is_leaving:
                node.target_x = node.leave_target_x
                node.target_y = node.leave_target_y

    def add_node(self, lon: float, lat: float, speed: float, heading: float, join_frames: int = 60):
        if len(self.nodes) >= 10:
            print(f"❌ 编队[{self.formation_id}] 已达到最大节点数10个，无法添加")
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
        print(f"\n✅ 编队[{self.formation_id}] 成功添加节点 ID:{self.max_id}")
        return True

    def add_multiple_nodes(self, add_count: int = 1):
        current = len(self.nodes)
        max_allow = 10 - current
        if max_allow <= 0:
            print(f"❌ 编队[{self.formation_id}] 已达最大节点数 10，无法添加")
            return

        add_count = min(add_count, max_allow)
        success = 0

        base_lon = self.config.main_lon + 0.001
        base_lat = self.config.main_lat + 0.001
        for i in range(add_count):
            lon = base_lon + i * 0.0001
            lat = base_lat + i * 0.0001
            if self.add_node(lon, lat, 1.5, 90.0, 60):
                success += 1

        self.config.node_num = len(self.nodes)
        self._set_target_formation()
        self.is_transition = True
        print(f"✅ 编队[{self.formation_id}] 批量添加完成：成功添加 {success} 个节点，当前总节点：{len(self.nodes)}")

    def remove_random_node(self, remove_count: int = 1):
        max_possible_remove = len(self.nodes) - 2
        if max_possible_remove <= 0:
            print(f"❌ 编队[{self.formation_id}] 节点数过少，无法删除")
            return

        remove_count = min(remove_count, max_possible_remove)
        slave_nodes = [node for node in self.nodes[1:] if not node.is_leaving]
        if len(slave_nodes) < remove_count:
            print(f"❌ 编队[{self.formation_id}] 没有足够可删除的节点")
            return

        leave_nodes = slave_nodes[-remove_count:]
        main = self.nodes[0]
        leave_dist = self.config.rel_distance * 15.0

        for leave_node in leave_nodes:
            leave_node.is_leaving = True
            opposite_heading = (main.heading + 90.0) % 360.0
            opposite_rad = math.radians(opposite_heading)
            leave_node.leave_target_x = leave_dist * math.sin(opposite_rad)
            leave_node.leave_target_y = leave_dist * math.cos(opposite_rad)
            print(f"\n✅ 编队[{self.formation_id}] 节点 ID:{leave_node.id} 开始脱离")

        self._set_target_formation()
        self.is_transition = True
        print(f"\n✅ 编队[{self.formation_id}] 已启动 {remove_count} 个节点脱离流程")

    def _transition_formation(self):
        for node in self.nodes[1:]:
            if node.is_joining:
                node.join_progress = min(1.0, node.join_progress + 1.0 / node.join_total_frames)
                node.rel_x += (node.target_x - node.rel_x) * 0.1
                node.rel_y += (node.target_y - node.rel_y) * 0.1
                if node.join_progress >= 1.0:
                    node.is_joining = False
                    print(f"✅ 编队[{self.formation_id}] 节点 ID:{node.id} 已完全加入编队")
            else:
                speed = TRANSITION_SPEED * 0.6 if node.is_leaving else TRANSITION_SPEED
                node.rel_x += (node.target_x - node.rel_x) * speed
                node.rel_y += (node.target_y - node.rel_y) * speed

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
                print(f"✅ 编队[{self.formation_id}] 编队已稳定")

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
                    delete_dist = 5 * self.config.rel_distance
                    if min_dist > delete_dist:
                        print(f"✅ 编队[{self.formation_id}] 节点 ID:{node.id} 已远离编队，消失")
                        nodes_to_remove.append(node)

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

        for node in nodes_to_remove:
            if node in self.nodes:
                self.nodes.remove(node)
        self.config.node_num = len(self.nodes)
        if nodes_to_remove:
            self._set_target_formation()

    def step_simulation(self):
        # 单编队仿真步（并行执行）
        self.current_time += self.config.sim_step
        self._update_maneuver()
        self._record_transition_step()
        return self.nodes

# ====================== 多编队管理器（核心：并行计算+ID管理） ======================
class MultiFormationManager:
    def __init__(self):
        self.formations: Dict[int, UUVFormationSimulator] = {}  # {编队ID: 仿真器}
        self.executor = ThreadPoolExecutor(max_workers=10)  # 并行线程池

    def add_formation(self, formation_id: int, config: FormationConfig):
        """添加编队（指定ID）"""
        if formation_id in self.formations:
            print(f"❌ 编队ID[{formation_id}] 已存在！")
            return
        sim = UUVFormationSimulator(config, formation_id)
        self.formations[formation_id] = sim
        print(f"✅ 编队[{formation_id}] 创建成功！")

    def get_formation(self, formation_id: int) -> UUVFormationSimulator:
        """根据ID获取编队"""
        return self.formations.get(formation_id)

    def parallel_step(self):
        """【并行计算】所有编队同时执行仿真步"""
        futures = []
        for sim in self.formations.values():
            # 提交到线程池并行执行
            future = self.executor.submit(sim.step_simulation)
            futures.append(future)
        # 等待所有编队计算完成
        for future in futures:
            future.result()

# ====================== 多编队可视化 ======================
class MultiFormationVisualizer:
    def __init__(self, manager: MultiFormationManager):
        self.manager = manager
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_aspect('equal')
        self.ax.set_title("多编队UUV集群仿真（实时地理定位）", fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.colors = ['#00FF00', '#FF6666', '#00BFFF', '#FFD700', '#FF69B4', '#32CD32']
        self.global_ref_lon = None
        self.global_ref_lat = None

    def update_frame(self, frame):
        self.manager.parallel_step()
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-WINDOW_RANGE, WINDOW_RANGE)
        self.ax.set_ylim(-WINDOW_RANGE, WINDOW_RANGE)
        self.ax.set_xlabel("东向 X (m)", fontsize=12)
        self.ax.set_ylabel("北向 Y (m)", fontsize=12)

        formation_ids = list(self.manager.formations.keys())
        if not formation_ids:
            return self.ax,

        # ==============================================
        # 🔥 修复点1：取【第一个编队主节点 实时坐标】为全局原点
        # ==============================================
        global_sim = self.manager.formations[formation_ids[0]]
        global_main_node = global_sim.nodes[0]
        self.global_ref_lon = global_main_node.lon
        self.global_ref_lat = global_main_node.lat

        # 遍历所有编队，计算【实时】全局偏移
        for idx, (fid, sim) in enumerate(self.manager.formations.items()):
            color = self.colors[idx % len(self.colors)]
            # 当前编队的主节点（实时运动后的经纬度）
            current_main_node = sim.nodes[0]
            
            # ==============================================
            # 🔥 修复点2：实时计算 → 编队主节点 相对于全局原点的米级偏移
            # ==============================================
            offset_x, offset_y = sim._geo2enu(
                current_main_node.lon, current_main_node.lat,
                self.global_ref_lon, self.global_ref_lat
            )

            # 节点最终坐标 = 编队内相对坐标 + 编队全局实时偏移
            xs, ys = [], []
            for n in sim.nodes:
                dx, dy = sim._geo2enu(n.lon, n.lat, current_main_node.lon, current_main_node.lat)
                final_x = dx + offset_x
                final_y = dy + offset_y
                xs.append(final_x)
                ys.append(final_y)

            self.ax.scatter(xs, ys, c=color, s=100, label=f"编队{fid}")
            
            # 节点标注
            for n, x, y in zip(sim.nodes, xs, ys):
                t = f"F{fid}\nID{n.id}"
                if n.is_leaving: t += "\n🚀脱离"
                if n.is_joining: t += "\n🔵加入"
                self.ax.annotate(t, (x, y), fontsize=7, color='white',
                                bbox=dict(boxstyle="round", fc="black", alpha=0.7))

        self.ax.legend(loc="upper right")
        info = (f"编队数量：{len(self.manager.formations)} | "
                f"编队1速度：2m/s | 编队2/3速度：4m/s")
        self.ax.text(0.01, 0.98, info, transform=self.ax.transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round", fc="black", alpha=0.7), color='cyan')
        return self.ax,

    def start(self):
        ani = FuncAnimation(self.fig, self.update_frame, interval=50)
        plt.tight_layout()
        plt.show()

# ====================== 多编队命令输入 ======================
def input_worker(manager: MultiFormationManager):
    print("\n===== 多编队命令面板 =====")
    print("格式：编队ID:命令   |   all:命令 = 所有编队执行")
    print("示例：1:1 → 编队1切直线 | 2:add → 编队2加节点 | all:2 → 所有编队切矩形")
    print("命令：1-5队形 | add/add2/add3 | remove/remove2/remove3")
    print("==========================\n")
    
    while True:
        try:
            cmd = input(">> 输入命令：").strip().lower()
            if ":" not in cmd:
                print("❌ 格式错误！请用 编队ID:命令")
                continue
            
            fid_str, cmd_str = cmd.split(":", 1)
            # 执行命令
            if fid_str == "all":
                # 所有编队执行
                for sim in manager.formations.values():
                    exec_command(sim, cmd_str)
            else:
                # 单个编队执行
                fid = int(fid_str)
                sim = manager.get_formation(fid)
                if not sim:
                    print(f"❌ 编队ID[{fid}] 不存在！")
                    continue
                exec_command(sim, cmd_str)
        except Exception as e:
            print(f"❌ 命令执行错误：{e}")

def exec_command(sim: UUVFormationSimulator, cmd: str):
    """执行编队命令"""
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

# ====================== 主函数 ======================
def main():
    # 1. 创建多编队管理器
    manager = MultiFormationManager()

    # 2. 创建多个编队（指定不同ID、位置、参数）
    # 编队1：ID=1，直线队形，初始位置(120.0, 30.0)
    config1 = FormationConfig(
        formation_type="line", old_num=6, node_num=6, main_lon=120.0, main_lat=30.0,
        rel_distance=10.0, init_speed=2.0, init_heading=135.0, heading_rate=1.0
    )
    manager.add_formation(1, config1)

    # 编队2：ID=2，三角形队形，初始位置(120.002, 30.01)
    config2 = FormationConfig(
        formation_type="triangle", old_num=5, node_num=5, main_lon=120.0003, main_lat=30.0001,
        rel_distance=10.0, init_speed=2.0, init_heading=45.0, heading_rate=0.0
    )
    manager.add_formation(2, config2)

    # 编队3：ID=3，三角形队形，初始位置(120.002, 30.01)
    config3 = FormationConfig(
        formation_type="triangle", old_num=5, node_num=5, main_lon=120.0003, main_lat=30.0004,
        rel_distance=10.0, init_speed=2.0, init_heading=45.0, heading_rate=0.0
    )
    manager.add_formation(3, config3)

    # 3. 启动可视化
    viz = MultiFormationVisualizer(manager)
    # 4. 启动命令输入线程
    threading.Thread(target=input_worker, args=(manager,), daemon=True).start()
    # 5. 启动绘图
    viz.start()

if __name__ == "__main__":
    main()