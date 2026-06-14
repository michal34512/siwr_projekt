# robot.py
import pygame
import math
import random
from settings import *


class UGV:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.theta = 0.0
        self.v = 0.0
        self.w = 0.0
        self.radius = 15

    def update(self, dt):
        self.theta += self.w * dt

        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt

    def draw(self, surface):
        pygame.draw.circle(
            surface, BLUE, (int(self.x), int(self.y)), self.radius)

        end_x = self.x + math.cos(self.theta) * self.radius * 1.5
        end_y = self.y + math.sin(self.theta) * self.radius * 1.5
        pygame.draw.line(surface, BLACK, (int(self.x), int(
            self.y)), (int(end_x), int(end_y)), 3)

    def get_measurements(self, landmarks, surface=None):
        measurements = []
        for lm_id, (lx, ly) in enumerate(landmarks):
            dx = lx - self.x
            dy = ly - self.y
            true_dist = math.hypot(dx, dy)
            if true_dist <= SENSOR_RANGE:
                true_angle = math.atan2(dy, dx) - self.theta
                noisy_dist = true_dist + random.gauss(0, SENSOR_STD_DISTANCE)
                noisy_angle = true_angle + \
                    random.gauss(0, math.radians(SENSOR_STD_ANGLE))
                noisy_angle = (noisy_angle + math.pi) % (2 * math.pi) - math.pi
                measurements.append((noisy_dist, noisy_angle, lm_id))
                if surface:
                    pygame.draw.line(surface, SENSOR_RAY_COLOR, (int(
                        self.x), int(self.y)), (int(lx), int(ly)), 1)

        return measurements
