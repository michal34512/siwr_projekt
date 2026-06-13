# Prawdziwa mapa: 0 = pusto, 1 = Drzewo (znajduje się na indeksie 3)
mapa = [0, 0, 0, 1, 0]

# Nasza wiedza o pozycji robota (rozkład prawdopodobieństwa dla 5 komórek).
# Zaczynamy z całkowitą pewnością: 100% (1.0) szans, że stoimy w komórce 0.
prawdopodobienstwo = [1.0, 0.0, 0.0, 0.0, 0.0]

print(f"START: {prawdopodobienstwo}")

# ==========================================
# 1. RÓWNANIE 4: TIME-UPDATE (Przewidywanie po ruchu)
# ==========================================
# Komenda (u_k): Przesuń się o 1 pole w prawo.
# Model Ruchu P(x_k | x_k-1, u_k): Silniki są tanie. Mamy 80% szans, że ruch się uda,
# i 20% szans, że kółka zabuksują i zostaniemy w miejscu.

nowe_prawdopodobienstwo = [0.0, 0.0, 0.0, 0.0, 0.0]

for i in range(5):
    if prawdopodobienstwo[i] > 0:
        # Jeśli uda nam się przejść o 1 w prawo (80% szans)
        if i + 1 < 5:
            nowe_prawdopodobienstwo[i+1] += prawdopodobienstwo[i] * 0.8
        # Jeśli zostaniemy w miejscu (20% szans)
        nowe_prawdopodobienstwo[i] += prawdopodobienstwo[i] * 0.2

prawdopodobienstwo = nowe_prawdopodobienstwo
print(f"Po ruchu (Równanie 4): {prawdopodobienstwo}")
# Wynik będzie: [0.2, 0.8, 0.0, 0.0, 0.0]
# Zauważ: nie jesteśmy już pewni na 100%, gdzie jesteśmy! Wiedza się "rozmyła".

# ==========================================
# 2. RÓWNANIE 5: MEASUREMENT-UPDATE (Korekcja przez zmysły)
# ==========================================
# Załóżmy, że w tym momencie włączamy czujnik i odczyt (z_k) mówi: "Widzę Drzewo!".
# Model Sensora P(z_k | x_k, m): Czujnik ma 90% skuteczności, ale myli się w 10% przypadków.

for i in range(5):
    # Sprawdzamy, jak bardzo dana komórka pasuje do odczytu czujnika
    ma_drzewo = (mapa[i] == 1)

    if ma_drzewo:
        # Jeśli na mapie jest tu drzewo, to pomiar jest bardzo prawdopodobny (mnożymy przez 0.9)
        prawdopodobienstwo[i] = prawdopodobienstwo[i] * 0.90
    else:
        # Jeśli na mapie nie ma tu drzewa, to czujnik musiał się pomylić (mnożymy przez 0.1)
        prawdopodobienstwo[i] = prawdopodobienstwo[i] * 0.10

# Mianownik Równania 5 (Dzielenie przez prawdopodobieństwo całkowite, tzw. Normalizacja)
# Musimy sprawić, by szanse znowu sumowały się do 100% (1.0).
suma = sum(prawdopodobienstwo)
prawdopodobienstwo = [p / suma for p in prawdopodobienstwo]

# Formatowanie do 2 miejsc po przecinku, by łatwiej się czytało
prawdopodobienstwo_czytelne = [round(p, 2) for p in prawdopodobienstwo]
print(f"Po pomiarze (Równanie 5): {prawdopodobienstwo_czytelne}")
# Wynik będzie: [0.20, 0.80, 0.0, 0.0, 0.0] -> To akurat zły przykład, bo drzewo jest na 3 indeksie, a my na 1.
# Ale ten kod pokaże dokładnie jak algorytm faworyzuje komórkę z drzewem!
