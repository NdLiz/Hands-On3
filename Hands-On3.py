class Regresion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(x)
        self.media_x = self.calcular_media(x)
        self.media_y = self.calcular_media(y)

    def calcular_media(self, datos):
        suma = sum(datos)
        return suma / len(datos)

    def calcular_coeficientes(self, matriz):
        # Calcula la inversa de la matriz mediante el método de eliminación 
        inversa = [[1.0 if i == j else 0.0 for j in range(len(matriz))] for i in range(len(matriz))]
        for i in range(len(matriz)):
            pivot = matriz[i][i]
            for j in range(len(matriz)):
                matriz[i][j] /= pivot
                inversa[i][j] /= pivot
            for k in range(len(matriz)):
                if k != i:
                    factor = matriz[k][i]
                    for j in range(len(matriz)):
                        matriz[k][j] -= factor * matriz[i][j]
                        inversa[k][j] -= factor * inversa[i][j]

        # Calcula el producto de la inversa por el vector 'y'
        producto2 = [[sum(self.x[k] ** i * self.y[k] for k in range(self.n))] for i in range(len(matriz))]

        # Calcula los coeficientes de la regresión
        coeficientes = [[sum(producto2[i][0] * inversa[i][k] for i in range(len(matriz)))] for k in range(len(matriz))]
        return coeficientes

    def calcular_coef_correlacion_determinacion(self, coeficientes):
        # Calcula las sumas y productos necesarios para el coeficiente de correlación y determinación
        suma_productos = 0.0
        suma_x_cuadrado = 0.0
        suma_y_cuadrado = 0.0

        for i in range(self.n):
            predicho = sum(coeficientes[j][0] * self.x[i] ** j for j in range(len(coeficientes)))
            suma_productos += (self.x[i] - self.media_x) * (predicho - self.media_y)
            suma_x_cuadrado += (self.x[i] - self.media_x) ** 2
            suma_y_cuadrado += (self.y[i] - self.media_y) ** 2

        coef_correlacion = suma_productos / (suma_x_cuadrado * suma_y_cuadrado) ** 0.5
        coef_determinacion = coef_correlacion ** 2
        return coef_correlacion, coef_determinacion


class RegresionLineal(Regresion):
    def __init__(self, x, y):
        super().__init__(x, y)

    def calcular_coeficientes(self):
        matriz = [[self.n, sum(self.x)], [sum(self.x), sum(x_i ** 2 for x_i in self.x)]]
        return super().calcular_coeficientes(matriz)

    def imprimir_resultados(self, coeficientes):
        print("\nRegresión Lineal: y =", coeficientes[0][0], "+", coeficientes[1][0], "* Batch size")
        coef_correlacion, coef_determinacion = super().calcular_coef_correlacion_determinacion(coeficientes)
        print("Coeficiente de Correlación:", coef_correlacion)
        print("Coeficiente de Determinación:", coef_determinacion)


class RegresionCuadratica(Regresion):
    def __init__(self, x, y):
        super().__init__(x, y)

    def calcular_coeficientes(self):
        matriz = [[self.n, sum(self.x), sum(x_i ** 2 for x_i in self.x)],
                  [sum(self.x), sum(x_i ** 2 for x_i in self.x), sum(x_i ** 3 for x_i in self.x)],
                  [sum(x_i ** 2 for x_i in self.x), sum(x_i ** 3 for x_i in self.x), sum(x_i ** 4 for x_i in self.x)]]
        return super().calcular_coeficientes(matriz)

    def imprimir_resultados(self, coeficientes):
        print("\nRegresión Cuadrática: y =", coeficientes[0][0], "+", coeficientes[1][0], "* Batch size +",
              coeficientes[2][0], "* Batch size^2")
        coef_correlacion, coef_determinacion = super().calcular_coef_correlacion_determinacion(coeficientes)
        print("Coeficiente de Correlación:", coef_correlacion)
        print("Coeficiente de Determinación:", coef_determinacion)


class RegresionCubica(Regresion):
    def __init__(self, x, y):
        super().__init__(x, y)

    def calcular_coeficientes(self):
        matriz = [[self.n, sum(self.x), sum(x_i ** 2 for x_i in self.x), sum(x_i ** 3 for x_i in self.x)],
                  [sum(self.x), sum(x_i ** 2 for x_i in self.x), sum(x_i ** 3 for x_i in self.x), sum(x_i ** 4 for x_i in self.x)],
                  [sum(x_i ** 2 for x_i in self.x), sum(x_i ** 3 for x_i in self.x), sum(x_i ** 4 for x_i in self.x), sum(x_i ** 5 for x_i in self.x)],
                  [sum(x_i ** 3 for x_i in self.x), sum(x_i ** 4 for x_i in self.x), sum(x_i ** 5 for x_i in self.x), sum(x_i ** 6 for x_i in self.x)]]
        return super().calcular_coeficientes(matriz)

    def imprimir_resultados(self, coeficientes):
        print("\nRegresión Cúbica: y =", coeficientes[0][0], "+", coeficientes[1][0], "* Batch size +",
              coeficientes[2][0], "* Batch size^2 +", coeficientes[3][0], "* Batch size^3")
        coef_correlacion, coef_determinacion = super().calcular_coef_correlacion_determinacion(coeficientes)
        print("Coeficiente de Correlación:", coef_correlacion)
        print("Coeficiente de Determinación:", coef_determinacion)


def main():
    """ Función principal que ejecuta las regresiones y muestra los resultados. """

    batchSize = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]
    machineEfficiency = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77 ,96 ,87 ,89 ,60 ,63 ,95 ,61 ,55, 56, 94, 93]

    # Ejecuta las funciones de regresión con los datos proporcionados
    regresion_lineal = RegresionLineal(batchSize, machineEfficiency)
    regresion_lineal.imprimir_resultados(regresion_lineal.calcular_coeficientes())

    regresion_cuadratica = RegresionCuadratica(batchSize, machineEfficiency)
    regresion_cuadratica.imprimir_resultados(regresion_cuadratica.calcular_coeficientes())

    regresion_cubica = RegresionCubica(batchSize, machineEfficiency)
    regresion_cubica.imprimir_resultados(regresion_cubica.calcular_coeficientes())


if __name__ == "__main__":
    # Llama a la función principal
    main()
