# =============================================
# Лабораторная работа №1. Работа с Numpy и Pandas
# =============================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Задание 1
# ------------------------------------------------------------------
print("\n=== Задание 1 ===")
matrix_ones = np.ones((5, 5))
arr_1d = np.random.random(20)
arr_2d = arr_1d.reshape(4, 5)

print("Матрица 5x5 из единиц:\n", matrix_ones)
print("\nОдномерный массив (20 эл.):\n", arr_1d)
print("\nМатрица 4x5:\n", arr_2d)

# ------------------------------------------------------------------
# Задание 2
# ------------------------------------------------------------------
print("\n=== Задание 2 ===")
n, m, k = 3, 4, 0.5
np.random.seed(42)
arr_nm = np.random.random((n, m))

print("Случайный массив {}x{}:\n".format(n, m), arr_nm)
print("\nЭлементы > {}:\n".format(k), arr_nm[arr_nm > k])
print("\nСтатистики:")
print("Среднее:", np.mean(arr_nm))
print("Медиана:", np.median(arr_nm))
print("Ст. откл.:", np.std(arr_nm))
print("Дисперсия:", np.var(arr_nm))
print("Минимум:", np.min(arr_nm))
print("Максимум:", np.max(arr_nm))

# ------------------------------------------------------------------
# Задание 3
# ------------------------------------------------------------------
print("\n=== Задание 3 ===")
np.random.seed(0)
v1 = np.random.random(5)
v2 = np.random.random(5)

cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
euclid_dist = np.linalg.norm(v1 - v2)

print("Вектор 1:", v1)
print("Вектор 2:", v2)
print("Косинус угла:", cos_angle)
print("Евклидово расстояние:", euclid_dist)

# ------------------------------------------------------------------
# Задание 4
# ------------------------------------------------------------------
print("\n=== Задание 4 ===")
np.random.seed(1)
df = pd.DataFrame(np.random.random((7, 4)), columns=["A", "B", "C", "D"])

df = df * 3
second_row = df.iloc[1]
df = df + second_row
df["E"] = 1.0

print("Датафрейм после преобразований:\n", df)
print("\nСтатистики по столбцам:\n", df.describe())

# ------------------------------------------------------------------
# Задание 5 — ЧТЕНИЕ sp500.csv (с запятой как разделителем!)
# ------------------------------------------------------------------
print("\n=== Задание 5 ===")

# Загружаем весь файл для проверки
df_sp500_full = pd.read_csv("sp500.csv")
print("Первые 10 строк (все столбцы):\n", df_sp500_full.head(10))

# Теперь нужные столбцы
usecols = [
    "Symbol",
    "Name",
    "Sector",
    "Price",
    "Book Value",
    "52 week low",
    "52 week high",
    "Market Cap",
]
df_sp500 = pd.read_csv("sp500.csv", usecols=usecols, index_col="Symbol")

print("\nЗагружено с нужными столбцами:")
print(df_sp500.head())

print("\nСтолбец 'Name':\n", df_sp500["Name"])

# Проверяем, есть ли NFLX
if "NFLX" in df_sp500.index:
    print("\nСтрока с индексом 'NFLX':\n", df_sp500.loc["NFLX"])
else:
    print("\nИндекс 'NFLX' отсутствует. Показан пример (первый индекс):")
    print(df_sp500.iloc[0])

# Строка с номером 238 (если есть)
if len(df_sp500) > 238:
    print("\nСтрока с номером 238:\n", df_sp500.iloc[238])
else:
    print(f"\nВсего строк: {len(df_sp500)}. Показана строка с индексом 3:")
    print(df_sp500.iloc[3])

# ------------------------------------------------------------------
# Задание 6
# ------------------------------------------------------------------
print("\n=== Задание 6 ===")

# Убедимся, что данные загружены (повторно или используем df_sp500)
df_sp500 = pd.read_csv("sp500.csv", usecols=usecols, index_col="Symbol")

# 1. Строки 100–120
if len(df_sp500) >= 121:
    print("Строки 100–120:\n", df_sp500.iloc[100:121])
else:
    print("Недостаточно строк. Показаны все:")
    print(df_sp500)

# 2. Копия без Book Value
df_copy = df_sp500.copy()
df_copy = df_copy.drop(columns=["Book Value"])
print("\nКопия без 'Book Value':\n", df_copy.head())

# 3. 52 week low < 80
low_filter = df_sp500[df_sp500["52 week low"] < 80]
print("\nГде '52 week low' < 80:\n", low_filter.head())

# 4. Financials или Energy и 52 week low > 50
sector_filter = df_sp500[
    ((df_sp500["Sector"] == "Financials") | (df_sp500["Sector"] == "Energy"))
    & (df_sp500["52 week low"] > 50)
]
print("\nFinancials/Energy и 52 week low > 50:\n", sector_filter)

# 5. То же с .query
sector_query = df_sp500.query(
    "Sector in ['Financials', 'Energy'] and `52 week low` > 50"
)
print("\nТо же через .query():\n", sector_query)

# ------------------------------------------------------------------
# Задание 7 — ЧТЕНИЕ tips.csv
# ------------------------------------------------------------------
print("\n=== Задание 7 ===")

df_tips = pd.read_csv("tips.csv")
print("Статистики по столбцу 'tip':\n", df_tips["tip"].describe())

# Дополнительно (по указанию из методички)
print("\nДополнительные статистики:")
print("Сумма:", df_tips["tip"].sum())
print("Медиана:", df_tips["tip"].median())
print("Мода:", df_tips["tip"].mode().tolist())

# ------------------------------------------------------------------
# Задание 8 (дополнительное)
# ------------------------------------------------------------------
print("\n=== Задание 8 ===")

# Задача: сгенерировать 500 случайных значений роста (в см) в диапазоне 150–200,
# разбить на интервалы (150–160, 160–170, ..., 190–200) и посчитать количество в каждом.

np.random.seed(42)
heights = np.random.uniform(150, 200, size=500)

bins = np.arange(150, 201, 10)  # [150, 160, ..., 200]
labels = ["150-160", "160-170", "170-180", "180-190", "190-200"]

height_groups = pd.cut(heights, bins=bins, labels=labels, include_lowest=True)
counts = height_groups.value_counts().sort_index()

print("Распределение роста по интервалам:")
print(counts)
