import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import pandas as pd
from pathlib import Path

class UAVFormationSimulator:
    def __init__(self, num_uavs=8, interval=5.0, collision_radius=2.0, switch_interval=3.0):
        """
        初始化编队模拟器
        :param num_uavs: 无人机数量
        :param interval: 队形节点间间隔（单位：米）
        :param collision_radius: 避碰半径（单位：米）
        :param switch_interval: 队形切换间隔（单位：秒）
        """
        self.num_uavs = num_uavs
        self.interval = interval
        self.collision_radius = collision_radius
        self.switch_interval = switch_interval
        
        # 队形序列：三角形→圆形→菱形→直线→矩形
        self.formation_sequence = ['rectangle','triangle', 'circle', 'diamond', 'line']
        self.current_formation_idx = 0
        self.current_formation = self.formation_sequence[self.current_formation_idx]
        
        # 初始化无人机位置（默认从三角形队形开始）
        self.uav_positions = self.generate_formation('rectangle')
        self.target_positions = self.uav_positions.copy()
        
        # 动画相关参数
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('UAV Formation Transformation Simulation')
        self.ax.grid(True)
        
        # 绘制无人机和队形中心
        self.uav_scatter = self.ax.scatter([], [], c='blue', s=100, label='UAVs')
        self.center_scatter = self.ax.scatter([], [], c='red', marker='x', s=200, label='Formation Center')
        # 修复：初始化formation_label
        self.formation_label = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                            verticalalignment='top', fontsize=10)
        self.ax.legend()
        
        # 动画控制
        self.animation = None
        self.frame_count = 0
        self.switch_frame_interval = int(switch_interval * 30)  # 30fps
        
        # 轨迹记录
        self.trajectory_data = []  # 存储每帧的轨迹数据
        self.formation_change_frames = [0]  # 记录队形变换发生的帧数
        
    def generate_formation(self, formation_type):
        """生成指定类型的队形位置"""
        positions = np.zeros((self.num_uavs, 2))
        center = np.array([0, 0])  # 原地变换，队形中心固定在原点
        
        if formation_type == 'triangle':
            # 正三角形队形
            # for i in range(self.num_uavs):
            #     angle = 2 * np.pi * i / self.num_uavs - np.pi/6  # 起始角度调整为正三角形
            #     radius = self.interval * np.sqrt(self.num_uavs) / 2
            #     positions[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])

            # # 修正：正三角形队形（顶点+边均匀分布）
            # # 1. 计算正三角形的边长（根据无人机数量和间隔）
            # side_length = self.interval * (np.ceil(np.sqrt(self.num_uavs)) - 1)
            # # 2. 正三角形的三个顶点坐标（中心在原点）
            # vertex1 = np.array([0, side_length / np.sqrt(3)])  # 上顶点
            # vertex2 = np.array([-side_length / 2, -side_length / (2 * np.sqrt(3))])  # 左下顶点
            # vertex3 = np.array([side_length / 2, -side_length / (2 * np.sqrt(3))])  # 右下顶点
            # vertices = [vertex1, vertex2, vertex3]
            
            # # 3. 分配无人机到三个边（包括顶点）
            # uavs_per_side = int(np.ceil(self.num_uavs / 3))  # 每边分配的无人机数量
            # idx = 0
            
            # for side in range(3):
            #     # 每个边的起点和终点
            #     start = vertices[side]
            #     end = vertices[(side + 1) % 3]
            #     # 均匀分布当前边的无人机
            #     for i in range(uavs_per_side):
            #         if idx >= self.num_uavs:
            #             break
            #         # 线性插值计算位置
            #         t = i / (uavs_per_side - 1) if uavs_per_side > 1 else 0.0
            #         positions[idx] = center + (1 - t) * start + t * end
            #         idx += 1

            # 等腰三角形队形（保证底边严格共线）
            # 1. 核心参数定义（可调整形状，底边始终水平）
            base_ratio = 1.6  # 底边长度比例
            height_ratio = 1  # 三角形高度比例
            base_length = self.interval * base_ratio * np.ceil(np.sqrt(self.num_uavs))
            height = self.interval * height_ratio * np.ceil(np.sqrt(self.num_uavs))
            
            # 2. 固定底边为水平直线（y坐标统一），保证所有底边节点y值相同
            base_y = -2 * height / 3  # 底边的y坐标（固定值，保证共线）
            # 三个顶点坐标（底边两个顶点y值完全一致，严格共线）
            vertex_top = np.array([0, height/3])          # 上顶点 (0, height/3)
            vertex_left = np.array([-base_length/2, base_y])  # 左下顶点 (-base_length/2, base_y)
            vertex_right = np.array([base_length/2, base_y])  # 右下顶点 (base_length/2, base_y)
            vertices = [vertex_top, vertex_left, vertex_right]
            
            # 3. 验证底边顶点共线（调试用，可保留）
            # 检查左右下顶点y坐标是否完全一致
            assert abs(vertex_left[1] - vertex_right[1]) < 1e-9, "底边顶点y坐标不一致！"
            
            # 4. 按边分配无人机（优先保证底边节点均匀分布在水平线上）
            # 计算三条边长度
            side1_len = np.linalg.norm(vertex_left - vertex_top)    # 左腰
            side2_len = np.linalg.norm(vertex_right - vertex_left)  # 底边（水平）
            side3_len = np.linalg.norm(vertex_top - vertex_right)   # 右腰
            total_side_len = side1_len + side2_len + side3_len
            
            # 按边长比例分配无人机数量
            uav_side1 = int(np.ceil(self.num_uavs * side1_len / total_side_len))  # 左腰
            uav_side2 = int(np.ceil(self.num_uavs * side2_len / total_side_len))  # 底边
            uav_side3 = self.num_uavs - uav_side1 - uav_side2                     # 右腰
            uavs_per_side = [uav_side1, uav_side2, uav_side3]
            
            idx = 0
            # 遍历三条边分配位置
            for side in range(3):
                start = vertices[side]
                end = vertices[(side + 1) % 3]
                num_uav = uavs_per_side[side]
                
                if num_uav <= 0 or idx >= self.num_uavs:
                    continue
                
                # 5. 生成边上的节点（底边节点强制统一y坐标）
                for i in range(num_uav):
                    if idx >= self.num_uavs:
                        break
                    t = i / (num_uav - 1) if num_uav > 1 else 0.0
                    # 基础插值计算
                    x = start[0] * (1 - t) + end[0] * t
                    y = start[1] * (1 - t) + end[1] * t
                    
                    # 关键：如果是底边，强制y坐标等于base_y，确保严格共线
                    if side == 1:  # side=1对应底边（左→右）
                        y = base_y  # 覆盖为固定值，彻底避免浮点误差导致的不共线
                    
                    positions[idx] = center + np.array([x, y])
                    idx += 1

        elif formation_type == 'circle':
            # 圆形队形
            # radius = self.interval * np.sqrt(self.num_uavs) / (2 * np.pi) * 2
            # for i in range(self.num_uavs):
            #     angle = 2 * np.pi * i / self.num_uavs
            #     positions[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
            for i in range(self.num_uavs):
                angle = 2 * np.pi * i / self.num_uavs - np.pi/6  # 起始角度调整为正三角形
                radius = self.interval * np.sqrt(self.num_uavs) / 2
                positions[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
                
        elif formation_type == 'diamond':
            # # 菱形队形（正方形旋转45度）
            # side_length = self.interval * np.sqrt(self.num_uavs) / 2
            # half_side = side_length / 2
            # # 生成菱形顶点和内部点
            # if self.num_uavs == 4:
            #     positions = np.array([
            #         [half_side, 0], [0, half_side],
            #         [-half_side, 0], [0, -half_side]
            #     ])
            # else:
            #     # 更多无人机时均匀分布在菱形边上
            #     for i in range(self.num_uavs):
            #         edge = i % 4
            #         t = (i // 4) / max(1, (self.num_uavs // 4))
            #         if edge == 0:  # 右→上
            #             positions[i] = [half_side*(1-t), half_side*t]
            #         elif edge == 1:  # 上→左
            #             positions[i] = [half_side*(-t), half_side*(1-t)]
            #         elif edge == 2:  # 左→下
            #             positions[i] = [half_side*(-(1-t)), half_side*(-t)]
            #         else:  # 下→右
            #             positions[i] = [half_side*t, half_side*(-(1-t))]

            # 优化后的菱形队形生成逻辑（核心）
            # 1. 定义菱形核心参数：轴长（顶点到中心的距离），保证四边等长
            # axis_length：菱形的"半轴长"（原点到顶点的距离），与间隔强关联
            axis_length = self.interval * (np.ceil(np.sqrt(self.num_uavs)) / 2)
            # 菱形4个顶点的基准坐标（上下左右对称）
            diamond_vertices = np.array([
                [axis_length, 0],   # 右顶点（x轴正方向）
                [0, axis_length],   # 上顶点（y轴正方向）
                [-axis_length, 0],  # 左顶点（x轴负方向）
                [0, -axis_length]   # 下顶点（y轴负方向）
            ])
            
            # 2. 分配无人机位置：优先顶点，剩余均匀分布在四条边上
            uav_per_edge = (self.num_uavs - 4) // 4  # 每条边额外分配的无人机数
            remaining_uavs = (self.num_uavs - 4) % 4  # 无法均分的剩余无人机数
            
            pos_idx = 0
            # 第一步：填充4个顶点
            for vertex in diamond_vertices:
                if pos_idx < self.num_uavs:
                    positions[pos_idx] = vertex
                    pos_idx += 1
            
            # 第二步：填充四条边（保证四边等长，均匀分布）
            edges = [
                (diamond_vertices[0], diamond_vertices[1]),  # 右→上
                (diamond_vertices[1], diamond_vertices[2]),  # 上→左
                (diamond_vertices[2], diamond_vertices[3]),  # 左→下
                (diamond_vertices[3], diamond_vertices[0])   # 下→右
            ]
            
            for edge_idx, (start, end) in enumerate(edges):
                # 每条边需要填充的无人机数（基础数 + 剩余数优先分配）
                num_on_edge = uav_per_edge + (1 if edge_idx < remaining_uavs else 0)
                if num_on_edge <= 0 or pos_idx >= self.num_uavs:
                    continue
                
                # 均匀插值计算边上的点（保证间距相等）
                for step in range(1, num_on_edge + 1):
                    if pos_idx >= self.num_uavs:
                        break
                    # 线性插值：t∈(0,1)，避免与顶点重合
                    t = step / (num_on_edge + 1)
                    # 计算当前点坐标（保证在边的连线上，四边等长）
                    positions[pos_idx] = start * (1 - t) + end * t
                    pos_idx += 1
                        
        elif formation_type == 'line':
            # 直线队形（沿X轴）
            start_x = -self.interval * (self.num_uavs - 1) / 2
            for i in range(self.num_uavs):
                positions[i] = [start_x + i * self.interval, 0]
                
        elif formation_type == 'rectangle':
            # 矩形队形
            cols = int(np.ceil(np.sqrt(self.num_uavs)))
            rows = int(np.ceil(self.num_uavs / cols))
            # 计算矩形边界
            start_x = -self.interval * (cols - 1) / 2
            start_y = -self.interval * (rows - 1) / 2
            idx = 0
            for y in range(rows):
                for x in range(cols):
                    if idx < self.num_uavs:
                        positions[idx] = [start_x + x * self.interval, start_y + y * self.interval]
                        idx += 1
        
        return positions
    
    def check_collision(self, positions):
        """检查无人机之间是否碰撞，若碰撞则调整位置"""
        adjusted_positions = positions.copy()
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                distance = np.linalg.norm(adjusted_positions[i] - adjusted_positions[j])
                if distance < self.collision_radius:
                    # 碰撞规避：沿两点连线方向分离
                    direction = (adjusted_positions[i] - adjusted_positions[j]) / (distance + 1e-6)
                    adjust_step = (self.collision_radius - distance) / 2
                    adjusted_positions[i] += direction * adjust_step
                    adjusted_positions[j] -= direction * adjust_step
        return adjusted_positions
    
    def switch_formation(self):
        """切换到下一个队形"""
        self.current_formation_idx = (self.current_formation_idx + 1) % len(self.formation_sequence)
        self.current_formation = self.formation_sequence[self.current_formation_idx]
        self.target_positions = self.generate_formation(self.current_formation)
        # 确保目标队形无碰撞
        self.target_positions = self.check_collision(self.target_positions)
        # 记录队形变换帧数
        self.formation_change_frames.append(self.frame_count)
        print(f"Switched to {self.current_formation} formation at frame {self.frame_count}")
    
    def update_positions(self):
        """更新无人机位置（平滑过渡到目标位置）"""
        # 采用指数平滑过渡
        alpha = 0.05  # 过渡系数，越小越平滑
        self.uav_positions = (1 - alpha) * self.uav_positions + alpha * self.target_positions
        # 实时避碰检查
        self.uav_positions = self.check_collision(self.uav_positions)
    
    def animate_frame(self, frame):
        """动画帧更新函数"""
        self.frame_count = frame
        
        # 定期切换队形
        if frame % self.switch_frame_interval == 0 and frame != 0:
            self.switch_formation()
        
        # 更新无人机位置
        self.update_positions()
        
        # 记录轨迹数据
        for uav_idx, pos in enumerate(self.uav_positions):
            self.trajectory_data.append({
                'frame': frame,
                'time': frame / 30.0,  # 按30fps计算时间
                'uav_id': uav_idx,
                'formation': self.current_formation,
                'x': pos[0],
                'y': pos[1]
            })
        
        # 更新绘图数据
        self.uav_scatter.set_offsets(self.uav_positions)
        self.center_scatter.set_offsets([[0, 0]])  # 队形中心固定在原点
        
        # 更新标题显示当前队形
        self.ax.set_title(f'UAV Formation Transformation: {self.current_formation}')

        # 实时更新队形类型标签文本
        self.formation_label.set_text(f'Type: {self.current_formation}')
        
        return self.uav_scatter, self.center_scatter, self.formation_label
    
    def save_trajectory_to_csv(self, filename='uav_transformation_trajectories.csv'):
        """保存轨迹数据到CSV文件"""
        if not self.trajectory_data:
            print("No trajectory data to save!")
            return
        
        df = pd.DataFrame(self.trajectory_data)
        output_path = Path(__file__).parent / filename
        df.to_csv(output_path, index=False)
        print(f"Trajectory data saved to: {output_path}")
        return output_path
    
    def print_trajectory_summary(self):
        """打印轨迹摘要"""
        if not self.trajectory_data:
            print("No trajectory data!")
            return
        
        df = pd.DataFrame(self.trajectory_data)
        print("\n" + "="*80)
        print("UAV FORMATION TRANSFORMATION TRAJECTORY SUMMARY")
        print("="*80)
        
        # 队形变换信息
        print("\n队形变换时间点:")
        for i, (start_frame, end_frame) in enumerate(zip(self.formation_change_frames[:-1], 
                                                           self.formation_change_frames[1:] + [len(df) // self.num_uavs])):
            formation = self.formation_sequence[i % len(self.formation_sequence)]
            time_start = start_frame / 30.0
            time_end = end_frame / 30.0
            print(f"  {formation.upper():12s}: 帧数 {start_frame:5d}-{end_frame:5d} | 时间 {time_start:7.2f}s-{time_end:7.2f}s")
        
        # 每个UAV的轨迹统计
        print("\n各UAV轨迹统计:")
        for uav_id in range(self.num_uavs):
            uav_data = df[df['uav_id'] == uav_id]
            if len(uav_data) > 0:
                x_min, x_max = uav_data['x'].min(), uav_data['x'].max()
                y_min, y_max = uav_data['y'].min(), uav_data['y'].max()
                total_distance = self._calculate_distance(uav_data)
                print(f"  UAV {uav_id:2d}: X范围[{x_min:7.2f},{x_max:7.2f}] "
                      f"Y范围[{y_min:7.2f},{y_max:7.2f}] 总移动距离:{total_distance:8.2f}m")
        
        print("\n" + "="*80 + "\n")
    
    def _calculate_distance(self, uav_data):
        """计算单个UAV的总移动距离"""
        positions = uav_data[['x', 'y']].values
        if len(positions) < 2:
            return 0.0
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        return np.sum(distances)
    
    def run_simulation(self, max_frames=1500, show_plot=True):
        """运行仿真
        
        :param max_frames: 最大仿真帧数
        :param show_plot: 是否显示动画
        """
        def frame_generator():
            frame = 0
            while frame < max_frames:
                yield frame
                frame += 1
        
        self.animation = FuncAnimation(
            self.fig, self.animate_frame,
            frames=frame_generator(), interval=33,  # ~30fps
            blit=True, cache_frame_data=False
        )
        
        if show_plot:
            plt.show()
        else:
            # 后台运行，完成所有帧
            while self.frame_count < max_frames - 1:
                pass

if __name__ == "__main__":
    # 初始化模拟器参数
    simulator = UAVFormationSimulator(
        num_uavs=20,          # 无人机数量（可调整）
        interval=5.0,        # 队形节点间隔（米）
        collision_radius=2.5, # 避碰半径（米）
        switch_interval=8.0  # 每10秒切换一次队形
    )
    
    # 运行仿真
    simulator.run_simulation(max_frames=1500, show_plot=True)
    
    # 保存轨迹数据
    simulator.save_trajectory_to_csv()
    
    # 打印轨迹摘要
    simulator.print_trajectory_summary()