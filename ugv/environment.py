# environment.py
import pygame
import random
from settings import *


def generate_landmarks(num_landmarks):
    landmarks = []
    for _ in range(num_landmarks):
        lx = random.randint(50, WIDTH - 50)
        ly = random.randint(50, HEIGHT - 50)
        landmarks.append((lx, ly))
    return landmarks


def draw_landmarks(surface, landmarks):
    for lx, ly in landmarks:
        pygame.draw.circle(surface, GREEN, (int(lx), int(ly)), 6)
        pygame.draw.circle(surface, BLACK, (int(lx), int(ly)), 6, 1)  # Ramka
