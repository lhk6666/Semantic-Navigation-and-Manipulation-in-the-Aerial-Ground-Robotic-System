import numpy as np
from scipy.interpolate import splprep, splev, BSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

class PathPlanner:
    def __init__(self, num_ctrl_points=8, obstacle_effect_area=3):
        self.num_ctrl_points = num_ctrl_points
        self.num_points = None
        self.obstacle_effect_area = obstacle_effect_area
        self.length_weight = 10.0
        self.curvature_weight = 3.0
        self.obstacle_weight = 3.0
        self._eps = 1e-8

    # ---------- utils ----------
    def is_on_path(self, P, A, C, tolerance=0.1):
        AC = C - A
        AP = P - A
        denom = max(np.linalg.norm(AC), self._eps)
        cross_product = np.abs(np.cross(AC, AP)) / denom
        if cross_product > tolerance:
            return False
        t = np.dot(AP, AC) / max(np.dot(AC, AC), self._eps)
        return 0 <= t <= 1

    def add_small_offset(self, P, A, C):
        AC = C - A
        norm = max(np.linalg.norm(AC), self._eps)
        perpendicular = np.array([-AC[1], AC[0]]) / norm
        offset_magnitude = 0.5
        return P + perpendicular * offset_magnitude

    def _arc_length_resample(self, polyline, n_samples):
        """Resample a polyline to n_samples points by approximate arc-length."""
        if len(polyline) < 2:
            return np.repeat(polyline, n_samples, axis=0)
        seg = np.diff(polyline, axis=0)
        d = np.linalg.norm(seg, axis=1)
        s = np.insert(np.cumsum(d), 0, 0.0)
        total = s[-1] if s[-1] > self._eps else self._eps
        s_q = np.linspace(0, total, n_samples)
        # interpolate x, y separately on s axis
        x = np.interp(s_q, s, polyline[:, 0])
        y = np.interp(s_q, s, polyline[:, 1])
        return np.stack([x, y], axis=1)

    # ---------- original: spline via splprep (interpolating nodes) ----------
    def get_spline_path(self, ctrl_points_flat):
        """Build cubic spline via splprep (s=0 interpolation) from 'nodes' and resample by arc length."""
        ctrl_points = ctrl_points_flat.reshape(-1, 2)
        # splprep expects sequences; s=0 forces interpolation
        tck, _ = splprep([ctrl_points[:, 0], ctrl_points[:, 1]], k=3, s=0)
        u_dense = np.linspace(0, 1, max(self.num_points * 10, 50))
        path_dense = np.array(splev(u_dense, tck)).T
        path_uniform = self._arc_length_resample(path_dense, self.num_points)
        return path_uniform

    def _path_cost_components(self, path):
        """Compute path length, curvature (second finite diff), and obstacle cost."""
        # length
        delta = np.diff(path, axis=0)
        seg_len = np.linalg.norm(delta, axis=1, keepdims=True)
        path_length = float(np.sum(seg_len))

        # curvature: discrete 2nd difference magnitude
        if len(path) >= 3:
            curvature = float(np.sum(np.linalg.norm(np.diff(path, n=2, axis=0), axis=1)))
        else:
            curvature = 0.0

        # obstacle cost
        obstacle_cost = 0.0
        if getattr(self, "obstacles", None) and len(self.obstacles) > 0 and len(path) >= 2:
            seg_start = path[:-1]
            seg_end = path[1:]
            seg_vec = seg_end - seg_start
            seg_len = np.linalg.norm(seg_vec, axis=1, keepdims=True)
            seg_len = np.maximum(seg_len, self._eps)
            unit_seg = seg_vec / seg_len

            obstacles_array = np.array(self.obstacles, dtype=float)
            obs_exp = obstacles_array[:, None, :]                # (N, S, 2)
            seg_start_exp = seg_start[None, :, :]                # (1, S, 2)
            unit_seg_exp = unit_seg[None, :, :]                  # (1, S, 2)
            seg_len_exp = seg_len[None, :, :]                    # (1, S, 1)

            vec_to_obs = obs_exp - seg_start_exp
            proj_len = np.sum(vec_to_obs * unit_seg_exp, axis=2, keepdims=True)
            proj_len = np.clip(proj_len, 0.0, seg_len_exp)
            proj_pt = seg_start_exp + unit_seg_exp * proj_len

            dist_vec = obs_exp - proj_pt
            dist = np.linalg.norm(dist_vec, axis=2)              # (N, S)

            safe = np.full_like(dist, float(self.obstacle_effect_area))
            # hinge penalty inside safe distance
            penalty = np.maximum(0.0, safe - dist)
            obstacle_cost = float(np.sum(penalty))

        return path_length, curvature, obstacle_cost

    def cost_function(self, ctrl_points_flat):
        path = self.get_spline_path(ctrl_points_flat)
        L, K, O = self._path_cost_components(path)
        return self.length_weight * L + self.curvature_weight * K + self.obstacle_weight * O

    # ==== BSpline begin: true B-spline control-point optimization ====
    @staticmethod
    def _open_uniform_knot_vector(n_ctrl, degree):
        """
        Open-uniform knot vector in [0,1] with endpoint multiplicity = degree+1.
        Total knots = n_ctrl + degree + 1.
        """
        k = degree
        m = n_ctrl + k + 1
        # endpoints repeated k+1
        t = np.zeros(m)
        t[-(k+1):] = 1.0
        # internal knots (if any)
        n_int = m - 2*(k+1)
        if n_int > 0:
            t[k+1:-k-1] = np.linspace(0, 1, n_int+2)[1:-1]
        return t

    def _eval_bspline_path(self, ctrl_points_flat, degree=3, n_dense=None):
        """Evaluate BSpline curve from true control points (coefficients)."""
        C = ctrl_points_flat.reshape(-1, 2)  # shape (n_ctrl, 2)
        n_ctrl = C.shape[0]
        k = degree
        # knot vector
        t = self._open_uniform_knot_vector(n_ctrl=n_ctrl, degree=k)
        # separate x,y splines
        spl_x = BSpline(t, C[:, 0], k)
        spl_y = BSpline(t, C[:, 1], k)
        # dense param grid
        if n_dense is None:
            n_dense = max(self.num_points * 10, 50)
        u = np.linspace(0.0, 1.0, n_dense)
        path_dense = np.stack([spl_x(u), spl_y(u)], axis=1)
        # arc-length resample to self.num_points
        path_uniform = self._arc_length_resample(path_dense, self.num_points)
        return path_uniform

    def cost_function_bspline(self, ctrl_points_flat, degree=3):
        path = self._eval_bspline_path(ctrl_points_flat, degree=degree)
        L, K, O = self._path_cost_components(path)
        return self.length_weight * L + self.curvature_weight * K + self.obstacle_weight * O

    def generate_safe_path_bspline(self, A, C, obstacles, degree=3):
        """
        True B-spline control-point optimization.
        - Optimizes control-point coefficients (BSpline) with fixed endpoints.
        - Keeps the same objective (length/curvature/obstacle) as original.
        """
        # init shared fields
        distance = np.linalg.norm(np.array(A) - np.array(C))
        self.num_points = int(np.clip(distance * 5, 30, 100))
        self.A = np.array(A, dtype=float)
        self.C = np.array(C, dtype=float)
        self.obstacles = [np.array(P, dtype=float) for P in obstacles]
        # avoid target-as-obstacle
        self.obstacles = [obs for obs in self.obstacles if not np.array_equal(obs, self.C)]
        # n_ctrl = user-specified
        n_ctrl = self.num_ctrl_points
        # initialize control points roughly along line A->C
        A = self.A; C_ = self.C
        line = [A + (i/(n_ctrl-1))*(C_ - A) for i in range(n_ctrl)]
        C_init = np.vstack(line)

        # (optional) small obstacle-aware offsets to interior control points
        if len(self.obstacles) > 0:
            path_dir = C_ - A
            path_len = max(np.linalg.norm(path_dir), self._eps)
            for i in range(1, n_ctrl-1):
                base = C_init[i].copy()
                total_offset = np.zeros(2)
                for obs in self.obstacles:
                    vec = obs - base
                    dist = max(np.linalg.norm(vec), self._eps)
                    # simple repulsion within effect area
                    if dist < self.obstacle_effect_area:
                        # side-dependent mild shaping (kept from your heuristic)
                        if path_dir[0] > 0 and np.cross(path_dir, vec) > 0:
                            offset = vec / (dist**2)
                        elif path_dir[0] > 0 and np.cross(path_dir, vec) <= 0:
                            offset = vec / dist
                        elif path_dir[0] <= 0 and np.cross(path_dir, vec) > 0:
                            offset = vec / dist
                        else:
                            offset = vec / (dist**2)
                        total_offset -= offset
                # clamp offset
                mag = np.linalg.norm(total_offset)
                max_off = path_len * 0.3
                if mag > max_off:
                    total_offset *= (max_off / mag)
                C_init[i] = base + total_offset

        x0 = C_init.flatten()

        # bounds: fix endpoints as A, C
        bounds = [(None, None)] * x0.size
        bounds[0:2] = [(A[0], A[0]), (A[1], A[1])]
        bounds[-2:] = [(C_[0], C_[0]), (C_[1], C_[1])]

        res = minimize(
            fun=lambda z: self.cost_function_bspline(z, degree=degree),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )
        if not res.success:
            print("BSpline optimizer: Not converged:", res.message)

        path_opt = self._eval_bspline_path(res.x, degree=degree)
        return [tuple(p) for p in path_opt]
    # ==== BSpline end ====

    # ---------- high-level API (original) ----------
    def generate_safe_path(self, A, C, obstacles):
        """
        Original interface:
        - Optimizes 'interpolating nodes' for a cubic spline built by splprep (s=0).
        """
        distance = np.linalg.norm(np.array(A) - np.array(C))
        self.num_points = int(np.clip(distance * 5, 30, 100))
        self.A = np.array(A, dtype=float)
        self.C = np.array(C, dtype=float)
        self.obstacles = [np.array(P, dtype=float) for P in obstacles]
        self.obstacles = [obs for obs in self.obstacles if not np.array_equal(obs, self.C)]

        if self.obstacles:
            for idx, P in enumerate(self.obstacles):
                if self.is_on_path(P, self.A, self.C):
                    self.obstacles[idx] = self.add_small_offset(P, self.A, self.C)

        path_direction = self.C - self.A
        path_length = max(np.linalg.norm(path_direction), self._eps)

        # initialize 'nodes' along line A->C, with small obstacle-aware offsets
        control_points = [self.A]
        for i in range(1, self.num_ctrl_points - 1):
            t = i / (self.num_ctrl_points - 1)
            base_point = self.A + t * path_direction
            total_offset = np.zeros(2)
            for obs in self.obstacles:
                vec_to_obs = obs - base_point
                dist = max(np.linalg.norm(vec_to_obs), self._eps)
                if dist < self.obstacle_effect_area:
                    if path_direction[0] > 0 and np.cross(path_direction, vec_to_obs) > 0:
                        offset = vec_to_obs / (dist ** 2)
                    elif path_direction[0] > 0 and np.cross(path_direction, vec_to_obs) <= 0:
                        offset = vec_to_obs / (dist)
                    elif path_direction[0] <= 0 and np.cross(path_direction, vec_to_obs) > 0:
                        offset = vec_to_obs / (dist)
                    else:
                        offset = vec_to_obs / (dist ** 2)
                    total_offset -= offset
            mag = np.linalg.norm(total_offset)
            if mag > path_length * 0.3:
                total_offset *= (path_length * 0.3 / mag)
            control_points.append(base_point + total_offset)
        control_points.append(self.C)
        control_points = np.array(control_points)

        initial_guess = control_points.flatten()
        bounds = [(None, None)] * len(initial_guess)
        bounds[0:2] = [(self.A[0], self.A[0]), (self.A[1], self.A[1])]
        bounds[-2:] = [(self.C[0], self.C[0]), (self.C[1], self.C[1])]

        result = minimize(
            self.cost_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )
        if not result.success:
            print("Not converged:", result.message)

        optimized_path = self.get_spline_path(result.x)
        return [tuple(point) for point in optimized_path]

    # ---------- viz ----------
    def plot_path(self, path, obstacles, A, C, title='Result'):
        path_points = np.array(path)
        plt.figure(figsize=(8, 6))
        plt.plot(path_points[:, 0], path_points[:, 1], '-', linewidth=2, label='Path')
        plt.scatter(path_points[:, 0], path_points[:, 1], c='y', s=16, label='Samples')

        for obstacle in obstacles:
            plt.scatter(obstacle[0], obstacle[1], c='r', marker='x', s=80)
            safety_circle = plt.Circle(obstacle, self.obstacle_effect_area, color='r', fill=False, linestyle='--', alpha=0.5)
            plt.gca().add_artist(safety_circle)

        plt.scatter([A[0]], [A[1]], c='g', marker='o', s=80, label='Start')
        plt.scatter([C[0]], [C[1]], c='r', marker='o', s=80, label='Goal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.title(title)
        plt.xlabel('X'); plt.ylabel('Y')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    A = (-0.625670555331895, 0.34496478618988025)
    C = (2.78, 3.65)
    obstacles = [[2.6, 8.7], [4.78, 3.65], [0.52, -0.24], [-1.05, 9.95], [-9.6, 8.7], [-1, 1.25]]

    planner = PathPlanner(num_ctrl_points=8, obstacle_effect_area=3)

    time_start = time.time()
    path_interp = planner.generate_safe_path(A, C, obstacles)
    print(f"[splprep] points: {len(path_interp)}, Time: {time.time() - time_start:.2f} s")

    time_start = time.time()
    path_bs = planner.generate_safe_path_bspline(A, C, obstacles, degree=3)
    print(f"[BSpline] points: {len(path_bs)}, Time: {time.time() - time_start:.2f} s")

    planner.plot_path(path_interp, obstacles, A, C, title='Interpolating Spline (splprep)')
    planner.plot_path(path_bs, obstacles, A, C, title='BSpline Control-Point Optimization')
