# slam.py
import numpy as np
import math
from settings import *


class EKF_SLAM:
    def __init__(self, start_x, start_y):
        self.xEst = np.array([[start_x], [start_y], [0.0]])
        self.PEst = np.eye(3) * 0.01
        self.lm_dict = {}
        self.Q = np.diag([ROBOT_STD_V, math.radians(ROBOT_STD_W)]) ** 2
        self.R = np.diag(
            [SENSOR_STD_DISTANCE, math.radians(SENSOR_STD_ANGLE)]) ** 2

    def predict(self, v, w, dt):
        theta = self.xEst[2, 0]

        dx = v * dt * math.cos(theta)
        dy = v * dt * math.sin(theta)
        dtheta = w * dt

        self.xEst[0, 0] += dx
        self.xEst[1, 0] += dy
        self.xEst[2, 0] = (self.xEst[2, 0] + dtheta +
                           math.pi) % (2 * math.pi) - math.pi
        G_x = np.array([
            [1.0, 0.0, -dy],
            [0.0, 1.0,  dx],
            [0.0, 0.0,  1.0]
        ])

        V = np.array([
            [dt * math.cos(theta), 0.0],
            [dt * math.sin(theta), 0.0],
            [0.0, dt]
        ])

        self.PEst[0:3, 0:3] = G_x @ self.PEst[0:3,
                                              0:3] @ G_x.T + V @ self.Q @ V.T

        if self.PEst.shape[0] > 3:
            self.PEst[0:3, 3:] = G_x @ self.PEst[0:3, 3:]
            self.PEst[3:, 0:3] = self.PEst[0:3, 3:].T

    def update(self, measurements):
        """Krok 2: Aktualizacja na podstawie pomiarów (Zamykanie pętli)."""
        for dist, angle, lm_id in measurements:
            if lm_id not in self.lm_dict:
                self._add_new_landmark(dist, angle, lm_id)
            else:
                self._update_landmark(dist, angle, lm_id)

    def _add_new_landmark(self, dist, angle, lm_id):
        """Inicjalizacja nowego landmarku w macierzach."""
        theta = self.xEst[2, 0]
        lm_x = self.xEst[0, 0] + dist * math.cos(theta + angle)
        lm_y = self.xEst[1, 0] + dist * math.sin(theta + angle)

        self.xEst = np.vstack((self.xEst, [[lm_x], [lm_y]]))
        self.lm_dict[lm_id] = len(self.xEst) - 2

        G_r = np.array([
            [1.0, 0.0, -dist * math.sin(theta + angle)],
            [0.0, 1.0,  dist * math.cos(theta + angle)]
        ])
        G_z = np.array([
            [math.cos(theta + angle), -dist * math.sin(theta + angle)],
            [math.sin(theta + angle),  dist * math.cos(theta + angle)]
        ])

        P_rr = self.PEst[0:3, 0:3]
        P_rm = self.PEst[0:3, 3:]
        P_ll = G_r @ P_rr @ G_r.T + G_z @ self.R @ G_z.T
        P_lr = G_r @ P_rr

        if P_rm.shape[1] > 0:
            P_lm = G_r @ P_rm
            P_top = np.hstack((self.PEst, np.vstack((P_lr.T, P_lm.T))))
            P_bot = np.hstack((P_lr, P_lm, P_ll))
        else:
            P_top = np.hstack((self.PEst, P_lr.T))
            P_bot = np.hstack((P_lr, P_ll))

        self.PEst = np.vstack((P_top, P_bot))

    def _update_landmark(self, dist, angle, lm_id):
        """Korekcja stanu na podstawie znanego landmarku."""
        idx = self.lm_dict[lm_id]

        dx = self.xEst[idx, 0] - self.xEst[0, 0]
        dy = self.xEst[idx+1, 0] - self.xEst[1, 0]
        q = dx**2 + dy**2
        sq = math.sqrt(q)

        z_pred_dist = sq
        z_pred_angle = math.atan2(dy, dx) - self.xEst[2, 0]
        z_pred_angle = (z_pred_angle + math.pi) % (2 * math.pi) - math.pi

        H = np.zeros((2, len(self.xEst)))
        H[0, 0:3] = [-dx/sq, -dy/sq, 0.0]
        H[1, 0:3] = [dy/q, -dx/q, -1.0]
        H[0, idx:idx+2] = [dx/sq, dy/sq]
        H[1, idx:idx+2] = [-dy/q, dx/q]

        y = np.array([[dist - z_pred_dist], [angle - z_pred_angle]])
        y[1, 0] = (y[1, 0] + math.pi) % (2 * math.pi) - math.pi

        S = H @ self.PEst @ H.T + self.R
        K = self.PEst @ H.T @ np.linalg.inv(S)

        self.xEst = self.xEst + K @ y
        self.PEst = (np.eye(len(self.xEst)) - K @ H) @ self.PEst
