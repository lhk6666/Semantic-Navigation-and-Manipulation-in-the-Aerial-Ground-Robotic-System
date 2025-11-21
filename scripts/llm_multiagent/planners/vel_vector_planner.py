import numpy as np
import math
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

class InitialVelocityPlanner:
    def __init__(self,
                 num_directions=360,
                 angle_weight=5.0,
                 zero_point_weight=3.0,
                 obstacle_weight=15.0,
                 threshold=3.0,  
                 rot_angle_step=0.02,
                 rotate_threshold=0.1):  
        self.num_directions = num_directions
        self.angle_weight = angle_weight
        self.zero_point_weight = zero_point_weight
        self.obstacle_weight = obstacle_weight
        self.threshold = threshold
        self.rot_angle_step = rot_angle_step
        self.rotate_threshold = rotate_threshold

    def compute_initial_velocity(self, A, C, obstacles, body_length, current_orientation=0.0):
        self.A = np.array(A, dtype=float)
        self.C = np.array(C, dtype=float)
        self.obstacles = [np.array(obs, dtype=float) for obs in obstacles]

        cost_list = []

        vec_to_goal = self.C - self.A
        vec_to_zero = np.array([0.0, 0.0]) - self.A
        dist_AC = np.linalg.norm(vec_to_goal)
        manipulator = self.A + body_length * np.array([np.cos(current_orientation), np.sin(current_orientation)])
        dist_manipulator2C = np.linalg.norm(manipulator - self.C)
        if dist_AC < 1e-9:
            return np.array([1.0, 0.0]), [], "none"

        goal_dir = vec_to_goal / dist_AC
        zero_point_dir = vec_to_zero / np.linalg.norm(vec_to_zero)

        for i in range(self.num_directions):
            theta = 2 * np.pi * i / self.num_directions
            dir_vec = np.array([np.cos(theta), np.sin(theta)])
            angle_diff = abs(np.arccos(np.clip(np.dot(dir_vec, goal_dir), -1.0, 1.0)))

            zero_point_diff = abs(np.arccos(np.clip(np.dot(dir_vec, zero_point_dir), -1.0, 1.0)))
            
            obstacle_cost = 0.0
            for obs in self.obstacles:
                vec_A_obs = obs - self.A
                proj = np.dot(vec_A_obs, dir_vec)
                if 0 <= proj <= 3 + body_length:
                    closest_point = self.A + proj * dir_vec
                    dist_to_line = np.linalg.norm(obs - closest_point)
                    if dist_to_line <= self.threshold:
                        obstacle_cost += 3.0 / (dist_to_line + 1e-9)
                    # elif self.threshold < dist_to_line < 2 * self.threshold:
                    #     obstacle_cost += 0.5 / (dist_to_line + 1e-9)
            if len(self.obstacles) == 0:
                obstacle_cost = 0.0

            future_position = self.A + dir_vec * 3
            if abs(future_position[0]) > 12 or abs(future_position[1]) > 6:
                window_cost = float('inf')
            else:
                window_cost = 0.0

            if dist_manipulator2C > 10:
                bias_ = 1
            else:
                bias_ = (3 / abs(dist_manipulator2C -1)) ** 2

            

            cost = self.angle_weight * bias_ * math.sqrt(angle_diff * (angle_diff + 1)) + \
                   self.zero_point_weight * zero_point_diff + self.obstacle_weight * obstacle_cost + window_cost
            cost_list.append(cost)
        
        best_cost_idx = np.argmin(cost_list)
        candidate = best_cost_idx * 2 * np.pi / self.num_directions
        current_orientation = current_orientation if current_orientation >= 0 else current_orientation + 2 * np.pi


        def is_rotation_safe(start_angle, end_angle, rotation_direction, body_length):
            start = start_angle % (2 * np.pi)
            end = end_angle % (2 * np.pi)
            if rotation_direction == "ccw":
                if end < start:
                    end += 2 * np.pi
                angles = np.linspace(start, end, int((end - start) / self.rot_angle_step) + 2)
            elif rotation_direction == "cw":
                if start < end:
                    start += 2 * np.pi
                angles = np.linspace(start, end, int((start - end) / self.rot_angle_step) + 2)
            else:
                raise ValueError("rotation_direction have to be 'ccw' or 'cw'")

            num_body_samples = 50 
            ts = np.linspace(0, 1, num_body_samples) 

            head_points = self.A[None, :] + body_length * np.column_stack((np.cos(angles), np.sin(angles)))
            body_points = self.A[None, :] + (head_points - self.A[None, :])[:, None, :] * ts[None, :, None]

            for obs in self.obstacles:
                distances = np.linalg.norm(body_points - obs[None, None, :], axis=2) 
                if np.any(distances < self.rotate_threshold):
                    return False
            return True



        current = current_orientation % (2*np.pi)
        cand = candidate % (2*np.pi)
        delta_ccw = (cand - current) % (2*np.pi)     
        delta_cw = (current - cand) % (2*np.pi)       

        safe_ccw = is_rotation_safe(current_orientation, candidate, "ccw", body_length)
        safe_cw  = is_rotation_safe(current_orientation, candidate, "cw", body_length)
        

        if safe_ccw or safe_cw:
            if safe_ccw and safe_cw:
                chosen_rotation = "ccw" if delta_ccw <= delta_cw else "cw"
            else:
                chosen_rotation = "ccw" if safe_ccw else "cw"
            safe_candidate = candidate
        else:
            safe_candidates = []
            for i in range(self.num_directions):
                theta = 2 * np.pi * i / self.num_directions
                with ThreadPoolExecutor(max_workers=2) as executor: 
                    future1 = executor.submit(is_rotation_safe, current_orientation, theta, "ccw", body_length)
                    future2 = executor.submit(is_rotation_safe, current_orientation, theta, "cw", body_length)
                    safe_ccw_i = future1.result()
                    safe_cw_i = future2.result()

                if safe_ccw_i or safe_cw_i:
                    safe_candidates.append((theta, cost_list[i]))
            if safe_candidates:
                safe_candidate, _ = min(safe_candidates, key=lambda x: x[1])
                safe_ccw_candidate = is_rotation_safe(current_orientation, safe_candidate, "ccw", body_length)
                safe_cw_candidate  = is_rotation_safe(current_orientation, safe_candidate, "cw", body_length)
                delta_ccw_candidate = (safe_candidate - current) % (2*np.pi)
                delta_cw_candidate  = (current - safe_candidate) % (2*np.pi)
                if safe_ccw_candidate and safe_cw_candidate:
                    chosen_rotation = "ccw" if delta_ccw_candidate <= delta_cw_candidate else "cw"
                else:
                    chosen_rotation = "ccw" if safe_ccw_candidate else "cw"
            else:
                safe_candidate = candidate
                chosen_rotation = "ccw" if delta_ccw <= delta_cw else "cw"
                
        if safe_candidate > np.pi:
            safe_candidate -= 2 * np.pi

        return safe_candidate, cost_list, chosen_rotation


if __name__ == "__main__":
    A = (-1.5, -0.7)
    C = (0.15958934, -0.3588437)
    obstacles = [[9.1, 4.8], [3.2, 0.0], [3.4, 4.1]]
    body_length = 6.5

    current_orientation = 1.5

    planner = InitialVelocityPlanner()
    orientation, costs, chosen_rotation = planner.compute_initial_velocity(
        A, C, obstacles, body_length, current_orientation=current_orientation)
    print(f"Target Orientation (radians): {orientation}")
    print(f"Corresponding Unit Vector: ({np.cos(orientation):.3f}, {np.sin(orientation):.3f})")
    print(f"Chosen Rotation Direction: {chosen_rotation}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot([A[0], C[0]], [A[1], C[1]], 'bo-', label='A to C')

    ax1.quiver(A[0], A[1], np.cos(current_orientation), np.sin(current_orientation),
               angles='xy', scale_units='xy', scale=1, color='b', label='Current Orientation')

    ax1.quiver(A[0], A[1], np.cos(orientation), np.sin(orientation),
               angles='xy', scale_units='xy', scale=1, color='r', label='Target Orientation')

    for obs in obstacles:
        ax1.plot(obs[0], obs[1], 'kx', markersize=10, label='Obstacle')

    start = current_orientation % (2 * np.pi)
    end = orientation % (2 * np.pi)
    if chosen_rotation == "ccw":
        if end < start:
            end += 2 * np.pi
        arc_angles = np.linspace(start, end, 100)
    else:  # chosen_rotation == "cw"
        if start < end:
            start += 2 * np.pi
        arc_angles = np.linspace(start, end, 100)
    arc_x = A[0] + body_length * np.cos(arc_angles)
    arc_y = A[1] + body_length * np.sin(arc_angles)
    ax1.plot(arc_x, arc_y, 'g--', label=f'Rotation Path ({chosen_rotation})')

    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Initial Velocity Planner')
    ax1.grid()
    ax1.axis('equal')

    angles = np.linspace(0, 2 * np.pi, planner.num_directions)
    ax2.plot(angles, costs, 'b-')
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost Distribution by Angle')
    ax2.grid()

    plt.tight_layout()
    plt.show()

