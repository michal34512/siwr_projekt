# slam_fast.py
import numpy as np
import math
import random
from settings import *

# ==========================================
# 1. STRUKTURY DANYCH DLA CZĄSTECZEK
# ==========================================


class LandmarkMiniEKF:
    """Mały, niezależny EKF tylko dla JEDNEGO drzewa."""

    def __init__(self, mu_x, mu_y):
        self.mu = np.array([[mu_x], [mu_y]])
        # Początkowa niepewność pozycji tego drzewa (tylko 2x2!)
        self.Sigma = np.eye(2) * 50.0


class Particle:
    """Pojedyncza cząsteczka zgadująca trasę robota."""

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = 1.0

        # Mapa dla tej konkretnej cząsteczki: id_drzewa -> obiekt LandmarkMiniEKF
        self.lm_dict = {}

# ==========================================
# 2. GŁÓWNY ALGORYTM FASTSLAM
# ==========================================


class FastSLAM:
    def __init__(self, start_x, start_y, num_particles=50):
        self.num_particles = num_particles

        # Tworzymy chmurę cząsteczek. Na starcie wszystkie są w tym samym miejscu.
        self.particles = [Particle(start_x, start_y, 0.0)
                          for _ in range(num_particles)]

        # Szumy
        # Szum odometrii (std v, std w)
        self.Q = np.array([ROBOT_STD_V, math.radians(ROBOT_STD_W)])
        self.R = np.diag(
            [SENSOR_STD_DISTANCE, math.radians(SENSOR_STD_ANGLE)]) ** 2

        # Zmienne dla main.py (żeby rysowanie działało bez zmian)
        self.xEst = np.zeros((3, 1))
        self.lm_dict = {}

    def predict(self, v, w, dt):
        """Krok 1: PREDYKCJA RUCHU (Odometria z szumem)"""
        for p in self.particles:
            # Każdą cząsteczkę poruszamy trochę inaczej! Dodajemy szum do komend.
            noisy_v = v + random.gauss(0, self.Q[0])
            noisy_w = w + random.gauss(0, self.Q[1])

            p.theta += noisy_w * dt
            p.theta = (p.theta + math.pi) % (2 * math.pi) - math.pi

            p.x += noisy_v * dt * math.cos(p.theta)
            p.y += noisy_v * dt * math.sin(p.theta)

    def update(self, measurements):
        """Krok 2: AKTUALIZACJA MAPY I WAG"""
        if not measurements:
            self._update_best_estimate()
            return

        for p in self.particles:
            for dist, angle, lm_id in measurements:
                if lm_id not in p.lm_dict:
                    self._add_new_landmark(p, dist, angle, lm_id)
                else:
                    self._update_landmark(p, dist, angle, lm_id)

        self._resample()
        self._update_best_estimate()

    def _add_new_landmark(self, p, dist, angle, lm_id):
        """Inicjalizacja drzewa dla danej cząsteczki."""
        # Skoro ufamy, że pozycja cząsteczki p jest idealna, to gdzie jest drzewo?
        lm_x = p.x + dist * math.cos(p.theta + angle)
        lm_y = p.y + dist * math.sin(p.theta + angle)

        p.lm_dict[lm_id] = LandmarkMiniEKF(lm_x, lm_y)
        # Waga cząsteczki się nie zmienia przy nowym drzewie

    def _update_landmark(self, p, dist, angle, lm_id):
        """Aktualizacja Mini-EKF i zmiana wagi cząsteczki."""
        lm = p.lm_dict[lm_id]

        # Oczekiwany pomiar (gdzie to drzewo powinno być z perspektywy tej cząsteczki)
        dx = lm.mu[0, 0] - p.x
        dy = lm.mu[1, 0] - p.y
        q = dx**2 + dy**2
        sq = math.sqrt(q)

        if sq < 1e-5:
            sq = 1e-5

        z_pred_dist = sq
        z_pred_angle = math.atan2(dy, dx) - p.theta
        z_pred_angle = (z_pred_angle + math.pi) % (2 * math.pi) - math.pi

        # Jakobian H względem MAPY (nie względem robota! Rozmiar 2x2)
        H = np.array([
            [dx/sq, dy/sq],
            [-dy/q, dx/q]
        ])

        # Innowacja
        y = np.array([[dist - z_pred_dist], [angle - z_pred_angle]])
        y[1, 0] = (y[1, 0] + math.pi) % (2 * math.pi) - math.pi

        # Równania Kalmana dla macierzy 2x2
        S = H @ lm.Sigma @ H.T + self.R
        K = lm.Sigma @ H.T @ np.linalg.inv(S)

        lm.mu = lm.mu + K @ y
        lm.Sigma = (np.eye(2) - K @ H) @ lm.Sigma

        # OBLICZANIE WAGI
        detS = np.linalg.det(S)
        invS = np.linalg.inv(S)

        # Obliczamy wykładnik jako skalar
        exponent = (-0.5 * y.T @ invS @ y).item()

        # Prawdopodobieństwo Gaussa
        likelihood = math.exp(exponent) / math.sqrt((2 * math.pi)**2 * detS)
        p.weight *= likelihood

    def _resample(self):
        """Krok 3: RESAMPLING (Ruletka Darwina)"""
        weights = [p.weight for p in self.particles]
        sum_w = sum(weights)

        if sum_w == 0:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles
            return

        # Normalizacja wag
        weights = [w / sum_w for w in weights]

        # Losowanie nowych cząsteczek na podstawie wag
        new_particles = []
        indices = np.random.choice(
            range(self.num_particles), size=self.num_particles, p=weights)

        for idx in indices:
            old_p = self.particles[idx]
            # Klony muszą mieć nową pamięć, inaczej będą modyfikować te same obiekty!
            new_p = Particle(old_p.x, old_p.y, old_p.theta)
            for lm_id, lm in old_p.lm_dict.items():
                # Kopiujemy macierze
                new_lm = LandmarkMiniEKF(lm.mu[0, 0], lm.mu[1, 0])
                new_lm.Sigma = lm.Sigma.copy()
                new_p.lm_dict[lm_id] = new_lm
            new_particles.append(new_p)

        self.particles = new_particles

    def _update_best_estimate(self):
        """
        Krok 4: Trik inżynierski. 
        Zmieniamy wygraną cząsteczkę w strukturę `self.xEst`, aby main.py
        mógł to narysować bez zmieniania ani jednej linijki kodu wizualizacji!
        """
        best_p = max(self.particles, key=lambda p: p.weight)

        self.xEst = np.array([[best_p.x], [best_p.y], [best_p.theta]])
        self.lm_dict = {}

        map_state = []
        idx = 3
        for lm_id, lm in best_p.lm_dict.items():
            self.lm_dict[lm_id] = idx
            map_state.extend([lm.mu[0, 0], lm.mu[1, 0]])
            idx += 2

        if map_state:
            self.xEst = np.vstack(
                (self.xEst, np.array(map_state).reshape(-1, 1)))

    def get_particles(self):
        return [(p.x, p.y, p.theta) for p in self.particles]
