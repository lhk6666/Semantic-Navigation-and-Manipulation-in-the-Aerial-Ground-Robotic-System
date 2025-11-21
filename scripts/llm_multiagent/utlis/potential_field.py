import cv2
import numpy as np
import matplotlib.pyplot as plt

class ColorGradientMapper:
    """
    将0~255的灰度值映射到自定义的 BGR 渐变（或 RGB，你可在get_color时自行调换）。
    """

    def __init__(self, color_start=(255, 0, 0), color_end=(0, 0, 255)):
        """
        初始化时生成一个从 color_start 到 color_end 的查找表 (LUT)，
        color_start, color_end: (B, G, R)，范围[0, 255]。
        """
        self.color_start = color_start
        self.color_end   = color_end
        self.lut = self._create_custom_colormap(self.color_start, self.color_end)

    def _create_custom_colormap(self, color1, color2):
        """
        在 BGR 空间里从 color1 平滑过渡到 color2，
        生成一个长度为 256 的查找表 (N×3)，可用于灰度值到彩色的映射。

        color1, color2: tuple, (B, G, R)，范围 [0,255]
        返回:
        - lut: 大小为 (256, 3) 的 uint8 数组，lut[i] = (B, G, R)
        """
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            alpha = i / 255.0  # 从 0 平滑到 1
            b = int(color1[0]*(1-alpha) + color2[0]*alpha)
            g = int(color1[1]*(1-alpha) + color2[1]*alpha)
            r = int(color1[2]*(1-alpha) + color2[2]*alpha)
            lut[i] = (b, g, r)
        return lut

    def get_color_bgr(self, value):
        """
        给定灰度值(0~255)，返回对应的 BGR 三元组 (tuple,int)。
        """
        if not (0 <= value <= 255):
            raise ValueError("value 必须在 [0, 255] 范围内")
        return tuple(self.lut[value])  # (B, G, R)

    def get_color_rgb(self, value):
        """
        如果需要 (R, G, B) 的顺序，则在这里对调即可。
        """
        b, g, r = self.get_color_bgr(value)
        return (r, g, b)

    def set_colors(self, color_start, color_end):
        """
        重新设置起止颜色，并刷新 LUT。
        """
        self.color_start = color_start
        self.color_end   = color_end
        self.lut = self._create_custom_colormap(color_start, color_end)

class PotentialField:
    def __init__(self, data):
        self.data = data
        self.object_pos = np.array(data["object position"], dtype=float)
        self.obstacles_pos = np.array(data["obstacles position"], dtype=float)
        self.target_pos = np.array(data["target position"], dtype=float)
        self.x_min, self.x_max = -5.5, 5.5
        self.y_min, self.y_max = -3.5, 3.5
        self.resolution = 100
        self.width = int((self.x_max - self.x_min) * self.resolution)
        self.height = int((self.y_max - self.y_min) * self.resolution)
        self.potential_field = np.zeros((self.height, self.width), dtype=np.float32)
        self.colorgrad = ColorGradientMapper(color_start=(255, 0, 0),
                                 color_end=(0, 0, 255))

    def attractive_potential(self, px, py, tx, ty, k=20):
        dx = px - tx
        dy = py - ty
        return k * np.sqrt(dx**2 + dy**2)

    def repulsive_potential(self, px, py, ox, oy, c=80):
        dx = px - ox
        dy = py - oy
        dist_sq = dx**2 + dy**2
        return c / (dist_sq + 1e-6)

    def compute_potential_field(self):
        # 1) 生成网格坐标
        xs = np.linspace(self.x_min, self.x_max, self.width)
        ys = np.linspace(self.y_min, self.y_max, self.height)
        # meshX, meshY 形状为 (height, width)
        meshX, meshY = np.meshgrid(xs, ys, indexing='xy')
        
        # 2) 计算对目标点的吸引势
        #    U_attr = k * sqrt( (x - tx)^2 + (y - ty)^2 )
        dx = meshX - self.target_pos[0]
        dy = meshY - self.target_pos[1]
        U_attr = 20.0 * np.sqrt(dx**2 + dy**2)

        # 3) 计算所有障碍物产生的排斥势并求和
        #    U_rep = sum( c / ((x - ox)^2 + (y - oy)^2 + 1e-6) )
        U_rep = np.zeros_like(meshX, dtype=np.float32)
        for obs in self.obstacles_pos:
            dx_obs = meshX - obs[0]
            dy_obs = meshY - obs[1]
            dist_sq = np.log1p(dx_obs**2 + dy_obs**2 + 1e-6)
            U_rep += (80.0 / dist_sq)

        # 4) 合并势场
        U_total = U_attr + U_rep

        # 5) 存储或返回结果
        # 注意：U_total 的下标 [row, col] 对应坐标 (meshX[row,col], meshY[row,col])
        self.potential_field = np.flipud(U_total)

    def visualize_potential_field(self):
        max_val = np.percentile(self.potential_field, 95)
        potential_field_clipped = np.clip(self.potential_field, 0, max_val)
        potential_norm = potential_field_clipped / max_val
        potential_gray = ((1 - potential_norm) * 255).astype(np.uint8)
        return potential_gray

    def to_image_coords(self, pos):
        col = int((pos[0] - self.x_min) * self.resolution)
        row = int((self.y_max - pos[1]) * self.resolution)
        return (col, row)

    def draw_elements(self, potential_gray):
        # Draw grid and labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255)
        thickness = 2

        for x in np.arange(self.x_min, self.x_max + 1, 1):
            col = int((x - self.x_min) * self.resolution)
            cv2.line(potential_gray, (col, 0), (col, self.height), (128), 1)
            # cv2.putText(potential_gray, f'{x + 0.5:.1f}', (col + 25, self.height - 10), font, font_scale, font_color, thickness, cv2.LINE_AA)

        for y in np.arange(self.y_min, self.y_max + 1, 1):
            row = int((self.y_max - y) * self.resolution)
            cv2.line(potential_gray, (0, row), (self.width, row), (128), 1)
            # cv2.putText(potential_gray, f'{y + 0.5:.1f}', (10, row - 45), font, font_scale, font_color, thickness, cv2.LINE_AA)
            
        for x in np.arange(self.x_min, self.x_max + 1, 1):
            for y in np.arange(self.y_min, self.y_max + 1, 1):
                col = int((x - self.x_min) * self.resolution)
                row = int((self.y_max - y) * self.resolution)
                cv2.putText(potential_gray, f'({int(x + 0.5)},{int(y + 0.5)})', (col + 10, row + 30), font, 0.5, font_color, 1, cv2.LINE_AA)

        # Draw target
        t_col, t_row = self.to_image_coords(self.target_pos)
        cv2.line(potential_gray, (t_col - 6, t_row - 6), (t_col + 6, t_row + 6), (255), 5)
        cv2.line(potential_gray, (t_col - 6, t_row + 6), (t_col + 6, t_row - 6), (255), 5)

        # Draw object
        obj_col, obj_row = self.to_image_coords(self.object_pos)
        cv2.circle(potential_gray, (obj_col, obj_row), 5, (255), 10)

    def generate_potential_field_image(self):
        self.compute_potential_field()
        potential_gray = self.visualize_potential_field()
        self.draw_elements(potential_gray)
        return potential_gray

# 示例数据
if __name__ == '__main__':
    data = {
        "object": "X cube",
        "object position": [-1.0, 0.0],
        "obstacles position": [[-1.0,1.0], [1.0,1.0]],
        "target position": [0.0, 2.0]
    }

    pf = PotentialField(data)
    potential_gray = pf.generate_potential_field_image()
    cv2.imwrite("potential_field.jpg", potential_gray)

    plt.figure(figsize=(10, 6))
    plt.imshow(potential_gray, cmap='gray', extent=[pf.x_min, pf.x_max, pf.y_min, pf.y_max])
    plt.title("Artificial Potential Field (Grayscale)")
    # plt.grid(color='white', linestyle='--', linewidth=0.5)
    # plt.xticks(np.arange(pf.x_min, pf.x_max + 1, 1))
    # plt.yticks(np.arange(pf.y_min, pf.y_max + 1, 1))
    plt.show()
