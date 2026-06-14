import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

# ==========================================
# 1. PARAMETRY SIECI STACJI (TDoA / ToA)
# ==========================================
STATIONS = np.array([
    [3000.0, 3000.0, 50.0],
    [-3000.0, 3000.0, 10.0],
    [-3000.0, -3000.0, 120.0],
    [3000.0, -3000.0, 5.0]
])

# Zwiększamy szum, by wyraźnie pokazać jak bardzo EKF wygładza surowe namiary
NOISE_STD_TDOA = 30.0


def lon_lat_to_cartesian(lon, lat, alt, lon0, lat0, alt0):
    R = 6371000.0
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    z = alt - alt0
    return x, y, z


def prepare_ground_truth(gps_path):
    gps = pd.read_csv(gps_path)
    lon0, lat0, alt0 = gps['lon'].iloc[0], gps['lat'].iloc[0], gps['height'].iloc[0]
    x, y, z = lon_lat_to_cartesian(
        gps['lon'], gps['lat'], gps['height'], lon0, lat0, alt0)

    # Twarda podłoga
    z = np.maximum(z, 0.0)

    gps['x'], gps['y'], gps['z'] = x, y, z
    gps = gps[gps['time_s'] >= 0].reset_index(drop=True)
    return gps


def simulate_tdoa_measurements(ground_truth, outage_start, outage_end):
    measurements = []
    for idx, row in ground_truth.iterrows():
        t = row['time_s']
        pos_true = np.array([row['x'], row['y'], row['z']])
        z_step = []
        for st_idx, st in enumerate(STATIONS):
            if outage_start <= t <= outage_end and st_idx in [0, 1]:
                z_step.append(np.nan)
            else:
                dist = np.linalg.norm(pos_true - st)
                noisy_dist = dist + np.random.normal(0, NOISE_STD_TDOA)
                z_step.append(noisy_dist)
        measurements.append(z_step)
    return measurements

# ==========================================
# 2. ROZSZERZONY FILTR KALMANA (EKF)
# ==========================================


def run_ekf_tdoa(ground_truth, measurements):
    X_ekf = np.zeros((3, 1))
    X_ekf[0, 0], X_ekf[1, 0], X_ekf[2, 0] = 0.1, 0.1, 10.0

    P_ekf = np.eye(3) * 10.0
    R_tdoa = NOISE_STD_TDOA**2
    Q_val = 1.0

    ekf_path = []

    for step in range(len(ground_truth)):
        row = ground_truth.iloc[step]
        dt = 0.1

        v_north = row['veln']
        v_east = row['vele']
        v_up = -row['veld']

        X_ekf[0, 0] += v_east * dt
        X_ekf[1, 0] += v_north * dt
        X_ekf[2, 0] += v_up * dt

        P_ekf += np.eye(3) * Q_val * dt

        # --- KROK 2: AKTUALIZACJA ---
        z_meas = measurements[step]
        for st_idx, st in enumerate(STATIONS):
            if np.isnan(z_meas[st_idx]):
                continue

            ex, ey, ez = X_ekf[0, 0], X_ekf[1, 0], X_ekf[2, 0]
            dx, dy, dz = ex - st[0], ey - st[1], ez - st[2]
            dist_pred = np.sqrt(dx**2 + dy**2 + dz**2)

            H = np.array([[dx/dist_pred, dy/dist_pred, dz/dist_pred]])

            y = z_meas[st_idx] - dist_pred
            S = H @ P_ekf @ H.T + R_tdoa
            K = (P_ekf @ H.T) / S

            X_ekf = X_ekf + K * y
            P_ekf = (np.eye(3) - K @ H) @ P_ekf

        if X_ekf[2, 0] < 0.0:
            X_ekf[2, 0] = 0.0

        ekf_path.append(X_ekf.flatten())

    return np.array(ekf_path)

# ==========================================
# 3. ZWYKŁA MULTILATERACJA (Metoda Najmniejszych Kwadratów)
# ==========================================


def run_multilateration(measurements):
    """
    Surowe wyliczanie pozycji z odległości.
    Jeśli dostępnych jest mniej niż 3 stacji, funkcja nie może wyliczyć 3D i zwraca NaN.
    """
    ml_path = []
    guess = np.array([0.0, 0.0, 10.0])  # Początkowe "zgadywanie"

    # Definiujemy granice: ( [min_X, min_Y, min_Z], [max_X, max_Y, max_Z] )
    # X i Y mogą być od -nieskończoności do +nieskończoności.
    # Z musi być >= 0 (Twarda podłoga)
    bounds = ([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])

    for z_meas in measurements:
        valid_idx = ~np.isnan(z_meas)

        # Jeśli mamy mniej niż 3 stacje (np. podczas awarii)
        if np.sum(valid_idx) < 3:
            ml_path.append([np.nan, np.nan, np.nan])
            continue

        active_stations = STATIONS[valid_idx]
        active_dists = np.array(z_meas)[valid_idx]

        # Funkcja błędu do minimalizacji
        def residual(p):
            return np.linalg.norm(active_stations - p, axis=1) - active_dists

        # Szukanie optymalnego punktu przecinania się sfer (z uwzględnieniem granic!)
        res = least_squares(residual, guess, bounds=bounds)

        if res.success:
            ml_path.append(res.x)
            # Lekkie zabezpieczenie - jeśli z jakiegoś powodu Z utknęło na 0,
            # to podbijamy guess odrobinę do góry, żeby algorytm w kolejnym kroku
            # nie "ślizgał się" po dolnej granicy, jeśli rakieta zacznie lecieć w górę.
            guess = res.x.copy()
            if guess[2] == 0.0:
                guess[2] = 1.0
        else:
            ml_path.append([np.nan, np.nan, np.nan])

    return np.array(ml_path)

# ==========================================
# 4. WIZUALIZACJA I PORÓWNANIE
# ==========================================


def plot_comparison_results(ground_truth, ekf_path, ml_path, outage_start, outage_end):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(STATIONS[:, 0], STATIONS[:, 1], STATIONS[:, 2],
               c='black', marker='^', s=200, label='Stacje Naziemne (TDoA)')

    ax.plot(ground_truth['x'], ground_truth['y'], ground_truth['z'],
            color='green', linewidth=4, alpha=0.5, label='Prawdziwa Rakieta (Ground Truth)')

    # Rysujemy surową multilaterację (kropki, żeby pokazać jak jest poszarpana i gdzie znika)
    ax.scatter(ml_path[:, 0], ml_path[:, 1], ml_path[:, 2],
               color='gray', s=10, alpha=0.5, label='Surowa Multilateracja (Szum + Brak danych w awarii)')

    ax.plot(ekf_path[:, 0], ekf_path[:, 1], ekf_path[:, 2],
            color='red', linestyle='-', linewidth=2.5, label='EKF (Wygładzanie + Predykcja z Pitota)')

    times = ground_truth['time_s'].values
    outage_mask = (times >= outage_start) & (times <= outage_end)
    if np.any(outage_mask):
        ax.plot(ekf_path[outage_mask, 0], ekf_path[outage_mask, 1], ekf_path[outage_mask, 2],
                color='orange', linewidth=5, alpha=0.8, label='EKF w trakcie Awarii')

    ax.set_title("Nawigacja: EKF vs Surowa Multilateracja", fontsize=16)
    ax.set_xlabel('Oś X (Wschód) [m]')
    ax.set_ylabel('Oś Y (Północ) [m]')
    ax.set_zlabel('Wysokość Z [m]')
    ax.legend(loc='upper left', fontsize=11)

    ax.view_init(elev=20, azim=-45)
    plt.show()


# ==========================================
# 5. GŁÓWNA PĘTLA
# ==========================================
if __name__ == "__main__":
    gps_filename = 'gps_data_synced.csv'

    # Ustawiamy awarię w środku lotu (Dostosuj sekundy do swojego pliku csv!)
    OUTAGE_START = 30.0
    OUTAGE_END = 50.0

    print("Wczytywanie i przygotowywanie danych...")
    try:
        ground_truth = prepare_ground_truth(gps_filename)
        measurements = simulate_tdoa_measurements(
            ground_truth, OUTAGE_START, OUTAGE_END)

        print("Obliczanie Surowej Multilateracji (bez EKF)...")
        ml_path = run_multilateration(measurements)

        print("Obliczanie EKF (z fuzją prędkości)...")
        ekf_path = run_ekf_tdoa(ground_truth, measurements)

        print("Rysowanie wyników...")
        plot_comparison_results(ground_truth, ekf_path,
                                ml_path, OUTAGE_START, OUTAGE_END)

    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{gps_filename}'.")
