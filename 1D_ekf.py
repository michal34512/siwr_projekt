import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETRY SYMULACJI 1D
# ==========================================
# Prawdziwe, nieznane robotowi pozycje
true_robot_x = 0.0
true_tree_x = 10.0

# Inicjalizacja "Mózgu" robota (Wektor Stanu)
# Robot myśli, że jest w 0.0, a drzewo widzi na 10.0
X_est = np.array([
    [0.0],  # Pozycja robota (x)
    [10.0]  # Pozycja drzewa (m)
])

# Początkowa macierz niepewności (Kowariancja P)
P_est = np.array([
    [0.0, 0.0],  # Jesteśmy w 100% pewni, że startujemy z zera
    # Ale czujnik miał błąd, więc pozycja drzewa ma niepewność (wariancję = 1.0)
    [0.0, 1.0]
])

# Szumy
Q = 0.5  # Szum kół (Predykcja) - z każdym metrem rośnie nasz błąd
R = 0.2  # Szum czujnika laserowego (Korekcja) - dalmierz myli się o ok. 0.2m

history_robot_err = []
history_tree_err = []

print("START: Robot w x=0, Drzewo w x=10")
print(
    f"Niepewność Robota: {P_est[0, 0]:.2f}, Niepewność Drzewa: {P_est[1, 1]:.2f}\n")

# ==========================================
# 2. PĘTLA SLAM (5 kroków w stronę drzewa)
# ==========================================
for step in range(1, 6):
    # --- Prawdziwy świat ---
    u = 1.0  # Robot chce jechać 1 metr do przodu
    true_robot_x += u  # W idealnym świecie przesuwa się o 1m

    # Wykonanie pomiaru zaszumionym laserem
    z_true = true_tree_x - true_robot_x  # Prawdziwa odległość do drzewa
    z_meas = z_true + np.random.randn() * np.sqrt(R)  # Odczyt z szumem

    # ----------------------------------------------------
    # RÓWNANIE 4: TIME-UPDATE (Predykcja - jazda w ciemno)
    # ----------------------------------------------------
    # Robot przesuwa się o 'u'. Drzewo stoi w miejscu.
    X_est[0, 0] += u

    # Nasza niepewność co do pozycji ROBOTA rośnie (bo koła się ślizgają)
    P_est[0, 0] += Q

    # ----------------------------------------------------
    # RÓWNANIE 5: MEASUREMENT-UPDATE (Korekcja z lasera)
    # ----------------------------------------------------
    # Czego się spodziewaliśmy?
    # Spodziewana odległość = (Drzewo - Robot)
    z_pred = X_est[1, 0] - X_est[0, 0]

    # Jaki popełniliśmy błąd w przewidywaniu?
    y = z_meas - z_pred

    # Macierz H mówi, jak pomiar zależy od stanu: z = m - x
    # Czyli pochodna po x to -1, a po m to 1.
    H = np.array([[-1.0, 1.0]])

    # Klasyczne Wzmocnienie Kalmana (Kalman Gain)
    S = H @ P_est @ H.T + R
    K = P_est @ H.T @ np.linalg.inv(S)

    # Złota reguła SLAM: Aktualizacja STANU i NIEPEWNOŚCI
    X_est = X_est + K * y
    P_est = (np.eye(2) - K @ H) @ P_est

    # Zapis do wykresu
    history_robot_err.append(P_est[0, 0])
    history_tree_err.append(P_est[1, 1])

    print(f"KROK {step}:")
    print(
        f"  Stan: Robot szacuje x={X_est[0, 0]:.2f}, Drzewo szacuje m={X_est[1, 0]:.2f}")
    print(
        f"  Kowariancja: Błąd Robota={P_est[0, 0]:.2f}, Błąd Drzewa={P_est[1, 1]:.2f}")
    print(f"  Korelacja (Sprężyna): P_xm = {P_est[0, 1]:.2f}\n")

# ==========================================
# 3. WYKRES ZBIEŻNOŚCI
# ==========================================
plt.plot(range(1, 6), history_robot_err, '-o',
         label='Niepewność Robota (P_xx)')
plt.plot(range(1, 6), history_tree_err, '-s', label='Niepewność Drzewa (P_mm)')
plt.title("Malejąca niepewność w 1D SLAM")
plt.xlabel("Kroki w czasie")
plt.ylabel("Wariancja (Błąd)")
plt.legend()
plt.grid()
plt.show()
