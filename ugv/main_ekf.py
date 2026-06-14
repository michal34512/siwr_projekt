# main.py
import pygame
import math
from settings import *
from robot import UGV
from environment import generate_landmarks, draw_landmarks
from slam_ekf import EKF_SLAM
from fast_slam import FastSLAM


def robot_control(robot):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        robot.v = MAX_SPEED
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        robot.v = -MAX_SPEED / 2.0
    else:
        robot.v = 0.0

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        robot.w = -MAX_ROTATION
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        robot.w = MAX_ROTATION
    else:
        robot.w = 0.0


def draw_slam_estimate_aligned(surface, slam, true_robot):
    rx_est, ry_est = slam.xEst[0, 0], slam.xEst[1, 0]
    rtheta_est = slam.xEst[2, 0]

    dtheta = true_robot.theta - rtheta_est

    for lm_id, idx in slam.lm_dict.items():
        ex, ey = slam.xEst[idx, 0], slam.xEst[idx+1, 0]
        local_x = ex - rx_est
        local_y = ey - ry_est
        rot_x = local_x * math.cos(dtheta) - local_y * math.sin(dtheta)
        rot_y = local_x * math.sin(dtheta) + local_y * math.cos(dtheta)
        final_x = true_robot.x + rot_x
        final_y = true_robot.y + rot_y

        # Rysujemy czerwony krzyżyk
        pygame.draw.line(surface, RED, (final_x - 5, final_y - 5),
                         (final_x + 5, final_y + 5), 2)
        pygame.draw.line(surface, RED, (final_x - 5, final_y + 5),
                         (final_x + 5, final_y - 5), 2)
    pygame.draw.circle(
        surface, RED, (int(true_robot.x), int(true_robot.y)), 17, 2)
    end_x = true_robot.x + math.cos(true_robot.theta) * 20
    end_y = true_robot.y + math.sin(true_robot.theta) * 20
    pygame.draw.line(surface, RED, (int(true_robot.x), int(
        true_robot.y)), (int(end_x), int(end_y)), 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("UGV EKF-SLAM")
    clock = pygame.time.Clock()

    robot = UGV(START_X, START_Y)
    # slam = FastSLAM(START_X, START_Y)
    slam = EKF_SLAM(START_X, START_Y)
    landmarks = generate_landmarks(NUM_LANDMARKS)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        robot_control(robot)
        robot.update(dt)
        screen.fill(WHITE)
        slam.predict(robot.v, robot.w, dt)
        measurements = robot.get_measurements(landmarks, surface=screen)
        slam.update(measurements)
        draw_landmarks(screen, landmarks)
        robot.draw(screen)
        # particles = slam.get_particles()
        # for px, py, ptheta in particles:
        #     pygame.draw.circle(screen, (255, 20, 147), (int(px), int(py)), 1)
        draw_slam_estimate_aligned(screen, slam, robot)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
