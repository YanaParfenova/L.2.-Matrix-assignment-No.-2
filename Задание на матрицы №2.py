"""
С клавиатуры вводится два числа K и N. Квадратная матрица А(N,N), состоящая
из 4-х равных по размерам подматриц, B,C,D,E заполняется случайным образом
целыми числами в интервале [-10,10]. Для отладки использовать не случайное
заполнение, а целенаправленное (ввод из файла и генератором). Вид матрицы А: 
В	Е
С	D
На основе матрицы А формируется матрица  F. По матрице F необходимо вывести
не менее 3 разных графика. Программа должна использовать функции библиотек
numpy  и matplotlib

17. Формируется матрица F следующим образом: скопировать в нее А и  если в Е
количество нулей в нечетных столбцах меньше, чем сумма чисел в нечетных
строках, то поменять местами В и Е симметрично, иначе С и Е поменять местами
несимметрично. При этом матрица А не меняется. После чего если определитель
матрицы А больше суммы диагональных элементов матрицы F,то вычисляется
выражение:A^-1*A^T – K * F^-1, иначе вычисляется выражение (A^-1 +G-F^-1)*K,
где G-нижняя треугольная матрица, полученная из А. Выводятся по мере
формирования А, F и все матричные операции последовательно.
"""

import numpy as np
import matplotlib.pyplot as plt

# Функция для создания матрицы случайно или из файла
def generate_matrix(n, use_file, filename=None):
    if use_file == "да":
        A = np.loadtxt(filename, dtype=int)
        return A
       
    else:
         sub_matrix_size = n // 2
         A = np.concatenate((np.concatenate((np.random.randint(-10, 11, size=(sub_matrix_size, sub_matrix_size)),
                                              np.random.randint(-10, 11, size=(sub_matrix_size, sub_matrix_size))), axis=1),
                            np.concatenate((np.random.randint(-10, 11, size=(sub_matrix_size, sub_matrix_size)),
                                              np.random.randint(-10, 11, size=(sub_matrix_size, sub_matrix_size))), axis=1)), axis=0)
         return A

# Функция, формирующая матрицу F
def transform_matrix(A, K, n):
    sub_matrix_size = n // 2
    B = A[:sub_matrix_size, :sub_matrix_size]
    C = A[sub_matrix_size:, :sub_matrix_size]
    D = A[sub_matrix_size:, sub_matrix_size:]
    E = A[:sub_matrix_size, sub_matrix_size:]

    zeros_in_odd_cols_E = np.sum(E[:, 1::2] == 0)
    sum_of_odd_rows_E = np.sum(E[1::2, :])

    F = np.copy(A)  # Создаем копию A

    # Симметричная замена B и E
    if zeros_in_odd_cols_E < sum_of_odd_rows_E:
        print(f"\nКоличество нулей в нечетных столбцах: {zeros_in_odd_cols_E}\nМЕНЬШЕ суммы чисел в нечетных строках: {sum_of_odd_rows_E},\nсимметрично меняем B и E:")
        F[:sub_matrix_size, :sub_matrix_size] = np.copy(E)
        F[:sub_matrix_size, sub_matrix_size:] = np.copy(B)
    # Несимметричная замена C и E
    else:
        print(f"\nКоличество нулей в нечетных столбцах : {zeros_in_odd_cols_E}\nБОЛЬШЕ суммы чисел в нечетных строках: {sum_of_odd_rows_E},\nнесимметрично меняем C и E:")
        F[sub_matrix_size:, :sub_matrix_size] = np.copy(E)
        F[:sub_matrix_size, sub_matrix_size:] = np.copy(C)  
    print("Матрица F:\n", F)

    # Определитель матрицы A
    det_A = np.linalg.det(A)
    # Сумма диагональных элементов матрицы F
    sum_diag_F = np.trace(F)
    
    if det_A > sum_diag_F:
        # Вычисление выражения A^-1*A^T – K * F^-1 
        result = np.linalg.inv(A) @ A.T - K * np.linalg.inv(F) 
        print(f"\nОпределитель матрицы A: {det_A}\nБОЛЬШЕ суммы диагональных элементов матрциы F: {sum_diag_F},\nРезультат вычисления выражения A^-1*A^T – K * F^-1\n:", result)
    else:
        # Нижняя треугольная матрица
        G = np.tril(A) 
        # Вычисление выражения (A^-1 +G-F^-1)*K
        result = (np.linalg.inv(A) + G - np.linalg.inv(F)) * K
        print(f"\nОпределитель матрицы A: {det_A}\nМЕНЬШЕ суммы диагональных элементов матрциы F: {sum_diag_F},\nРезультат вычисления выражения (A^-1 +G-F^-1)*K:\n", result)
    return F, result

# По матрице F выводятся 3 разных графика
def plot_matrix(matrix):

    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.title("Тепловая карта матрицы F")
    plt.xlabel("Индекс столбца")
    plt.ylabel("Индекс строки")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.hist(matrix.flatten(), bins=20)
    plt.title("Гистограмма элементов матрицы F")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.show()

    plt.figure()
    plt.plot(np.diag(matrix))
    plt.title("Диагональные элементы матрицы F")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.show()


# Основной код
K = int(input("Введите значение K: "))

use_file = input("Использовать файл для заполнения матрицы? (да/нет): ").strip().lower()
if use_file == 'да':
    filename = "matrix.txt"
    N = 4
    A = generate_matrix(N, use_file, filename)
   
else:
    N = int(input("Введите размер матрицы N (четное число): "))
    A = generate_matrix(N, use_file)

print("Матрица A:\n", A)

F, result_matrix = transform_matrix(A, K, N)

plot_matrix(result_matrix)

