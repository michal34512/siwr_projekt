import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ==========================================
# 1. PARAMETRY MODELU
# ==========================================
DT = 0.1
SIM_TIME = 70.0 # Czas wystarczający na przejechanie pełnego koła

# Zwiększone szumy, by zamykanie pętli było spektakularne
Q = np.diag([0.2, np.deg2rad(4.0)]) ** 2 # Szum Odometrii
R = np.diag([0.3, np.deg2rad(3.0)]) ** 2 # Szum Sensora (Dystans, Kąt)

# Prawdziwe otoczenie (Robot nie ma do tego dostępu! Służy tylko do generowania pomiarów)
TRUE_LANDMARKS = np.array([
    [10.0, -2.0], [15.0, 10.0], [3.0, 15.0], [-5.0, 5.0]
])

MAX_RANGE = 8.0 # Krótki zasięg radaru zmusza robota do odkrywania mapy po kolei

# ==========================================
# 2. FIZYKA I SYMULACJA (Środowisko)
# ==========================================
def motion_model(x, u):
    theta = x[2, 0]
    return x + np.array([
        [u[0, 0] * DT * math.cos(theta)],
        [u[0, 0] * DT * math.sin(theta)],
        [u[1, 0] * DT]
    ])

def get_observations(xTrue):
    z = []
    for i, lm in enumerate(TRUE_LANDMARKS):
        dx = lm[0] - xTrue[0, 0]
        dy = lm[1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        if d < MAX_RANGE:
            angle = (math.atan2(dy, dx) - xTrue[2, 0] + math.pi) % (2*math.pi) - math.pi
            # Dodanie szumu do pomiaru
            d_noisy = d + np.random.randn() * math.sqrt(R[0,0])
            a_noisy = angle + np.random.randn() * math.sqrt(R[1,1])
            z.append(np.array([[d_noisy], [a_noisy], [i]])) # Zwracamy pomiar i ID landmarku
    return z

# ==========================================
# 3. PEŁNY EKF SLAM (Mózg Robota)
# ==========================================
# Początkowy wektor stanu to tylko robot (x, y, theta). Mapa jest pusta!
xEst = np.zeros((3, 1))
PEst = np.eye(3) * 0.001 
landmark_dict = {} # Słownik: ID landmarku -> indeks w wektorze xEst

def ekf_predict(xEst, PEst, u):
    # Predykcja pozycji robota (pierwsze 3 elementy wektora stanu)
    xEst[0:3] = motion_model(xEst[0:3], u)
    
    # Jakobian G_x modelu ruchu
    theta = xEst[2, 0]
    G_x = np.array([
        [1.0, 0.0, -u[0, 0] * DT * math.sin(theta)],
        [0.0, 1.0,  u[0, 0] * DT * math.cos(theta)],
        [0.0, 0.0, 1.0]
    ])
    
    # Transformacja szumu na przestrzeń stanu
    V = np.array([[DT * math.cos(theta), 0], [DT * math.sin(theta), 0], [0, DT]])
    R_movement = V @ Q @ V.T
    
    # Aktualizacja kowariancji robota P_rr
    PEst[0:3, 0:3] = G_x @ PEst[0:3, 0:3] @ G_x.T + R_movement
    
    # Aktualizacja korelacji między robotem a mapą (P_rm) - KLUCZ DO ZAMYKANIA PĘTLI!
    if PEst.shape[0] > 3:
        PEst[0:3, 3:] = G_x @ PEst[0:3, 3:]
        PEst[3:, 0:3] = PEst[0:3, 3:].T
        
    return xEst, PEst

def add_new_landmark(xEst, PEst, z):
    r = z[0, 0]
    phi = z[1, 0]
    lm_id = int(z[2, 0])
    
    # Inicjalizacja nowej pozycji X, Y dla znalezionego landmarku
    lm_x = xEst[0, 0] + r * math.cos(xEst[2, 0] + phi)
    lm_y = xEst[1, 0] + r * math.sin(xEst[2, 0] + phi)
    
    # Rozszerzenie wektora stanu
    xEst = np.vstack((xEst, [[lm_x], [lm_y]]))
    landmark_dict[lm_id] = len(xEst) - 2 # Zapisz indeks
    
    # Jakobiany inicjalizacji: względem robota (G_r) i sensora (G_z)
    G_r = np.array([
        [1, 0, -r * math.sin(xEst[2, 0] + phi)],
        [0, 1,  r * math.cos(xEst[2, 0] + phi)]
    ])
    G_z = np.array([
        [math.cos(xEst[2, 0] + phi), -r * math.sin(xEst[2, 0] + phi)],
        [math.sin(xEst[2, 0] + phi),  r * math.cos(xEst[2, 0] + phi)]
    ])
    
    # Rozszerzanie Macierzy Kowariancji PEst
    P_rr = PEst[0:3, 0:3]
    P_rm = PEst[0:3, 3:]
    
    P_ll = G_r @ P_rr @ G_r.T + G_z @ R @ G_z.T
    P_lr = G_r @ P_rr
    
    if P_rm.shape[1] > 0:
        P_lm = G_r @ P_rm
        P_top = np.hstack((PEst, np.vstack((P_lr.T, P_lm.T))))
        P_bot = np.hstack((P_lr, P_lm, P_ll))
    else:
        P_top = np.hstack((PEst, P_lr.T))
        P_bot = np.hstack((P_lr, P_ll))
        
    PEst = np.vstack((P_top, P_bot))
    return xEst, PEst

def ekf_update(xEst, PEst, z):
    lm_id = int(z[2, 0])
    lm_idx = landmark_dict[lm_id]
    
    # Oczekiwany pomiar
    dx = xEst[lm_idx, 0] - xEst[0, 0]
    dy = xEst[lm_idx+1, 0] - xEst[1, 0]
    q = dx**2 + dy**2
    sq = math.sqrt(q)
    
    z_pred = np.array([[sq], [math.atan2(dy, dx) - xEst[2, 0]]])
    z_pred[1, 0] = (z_pred[1, 0] + math.pi) % (2*math.pi) - math.pi
    
    # Jakobian H rozciągnięty na CAŁY wektor stanu
    H = np.zeros((2, len(xEst)))
    H[0, 0:3] = [-dx/sq, -dy/sq, 0]
    H[1, 0:3] = [dy/q, -dx/q, -1]
    H[0, lm_idx:lm_idx+2] = [dx/sq, dy/sq]
    H[1, lm_idx:lm_idx+2] = [-dy/q, dx/q]
    
    # Innowacja
    y = z[0:2] - z_pred
    y[1, 0] = (y[1, 0] + math.pi) % (2*math.pi) - math.pi
    
    # Kalman Gain i Aktualizacja
    S = H @ PEst @ H.T + R
    K = PEst @ H.T @ np.linalg.inv(S)
    
    xEst = xEst + K @ y
    PEst = (np.eye(len(xEst)) - K @ H) @ PEst
    return xEst, PEst

# ==========================================
# 4. GŁÓWNA PĘTLA
# ==========================================
xTrue = np.zeros((3, 1))
xDR = np.zeros((3, 1)) # Ślepa odometria

hxTrue, hxEst, hxDR = [], [], []

for t in np.arange(0, SIM_TIME, DT):
    u_cmd = np.array([[1.0], [0.1]]) # Jazda po okręgu
    
    # 1. Rzeczywistość
    u_noisy = u_cmd + np.array([[np.random.randn() * math.sqrt(Q[0,0])], [np.random.randn() * math.sqrt(Q[1,1])]])
    xTrue = motion_model(xTrue, u_noisy)
    xDR = motion_model(xDR, u_cmd)
    
    # 2. PGM SLAM: Predykcja
    xEst, PEst = ekf_predict(xEst, PEst, u_cmd)
    
    # 3. PGM SLAM: Mapowanie i Korekcja (Obserwacje)
    z_list = get_observations(xTrue)
    for z in z_list:
        lm_id = int(z[2, 0])
        if lm_id not in landmark_dict:
            xEst, PEst = add_new_landmark(xEst, PEst, z) # DODAWANIE DO MAPY
        else:
            xEst, PEst = ekf_update(xEst, PEst, z)       # AKTUALIZACJA (Fuzja/Loop Closure)
            
    hxTrue.append(xTrue.copy())
    hxEst.append(xEst[0:3].copy())
    hxDR.append(xDR.copy())

# ==========================================
# 5. WIZUALIZACJA (Wykreślanie elips)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Pełny EKF-SLAM: Zamykanie pętli i sieć korelacji", fontsize=14)

hxTrue = np.array(hxTrue)
hxEst = np.array(hxEst)
hxDR = np.array(hxDR)

ax.plot(hxTrue[:, 0, 0], hxTrue[:, 1, 0], "-g", label="Prawdziwa Trasa")
ax.plot(hxDR[:, 0, 0], hxDR[:, 1, 0], "--r", label="Ślepa Odometria")
ax.plot(hxEst[:, 0, 0], hxEst[:, 1, 0], "-b", label="Trasa SLAM")

ax.scatter(TRUE_LANDMARKS[:, 0], TRUE_LANDMARKS[:, 1], marker='*', s=150, c='black', label="Prawdziwe Landmarki")

# Rysowanie zmapowanych landmarków i ich elips niepewności
for lm_id, idx in landmark_dict.items():
    ax.scatter(xEst[idx, 0], xEst[idx+1, 0], marker='o', s=50, c='blue')
    # Obliczanie elipsy na podstawie bloku macierzy kowariancji dla danego landmarku
    P_lm = PEst[idx:idx+2, idx:idx+2]
    eigvals, eigvecs = np.linalg.eig(P_lm)
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    # Elipsa 3-sigma (99% pewności)
    width, height = 3 * np.sqrt(eigvals) 
    ellipse = Ellipse((xEst[idx, 0], xEst[idx+1, 0]), width, height, angle=angle, edgecolor='blue', fill=False, lw=2)
    ax.add_patch(ellipse)

ax.legend()
ax.grid(True)
ax.axis("equal")
plt.show()