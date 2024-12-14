from PIL import Image
import matplotlib.pyplot as plt

import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform

# ... (código para cargar y redimensionar la imagen) ...
imagen = Image.open('Perro2.jpeg')
imagen_pixelada = imagen.resize((30, 30), resample=Image.NEAREST)
imagen_array = np.array(imagen_pixelada)

#plt.imshow(imagen_redimensionada)
#plt.show()

# ... (código para encontrar coordenadas verdes) ...
verde_bajo = np.array([0, 50, 0])    # Verde más oscuro
verde_alto = np.array([50, 150, 50])  # Verde oscuro/medio

mascara = cv2.inRange(imagen_array, verde_bajo, verde_alto)
coordenadas_verdes = np.where(mascara == 255)
umbral_distancia = 5  # Ajusta este valor según la distancia deseada

# ... (código para eliminar puntos cercanos) ...
distancias = pdist(np.column_stack(coordenadas_verdes))
matriz_distancias = squareform(distancias)

puntos_a_eliminar = set()
for i in range(len(coordenadas_verdes[0])):
    if i not in puntos_a_eliminar:
        for j in range(i + 1, len(coordenadas_verdes[0])):
            if matriz_distancias[i, j] < umbral_distancia:
                puntos_a_eliminar.add(j)

coordenadas_verdes_filtradas = tuple(np.delete(coord, list(puntos_a_eliminar)) for coord in coordenadas_verdes)

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 3 subplots

# Primera figura: Imagen original
axes[0].imshow(imagen_array)
axes[0].set_title('Imagen Original')

# Segunda figura: Todos los puntos verdes
axes[1].imshow(imagen_array)
axes[1].scatter(coordenadas_verdes[1], coordenadas_verdes[0], color='red', marker='o', s=10)
axes[1].set_title('Puntos Verdes')

# Tercera figura: Puntos verdes sin vecinos cercanos
axes[2].imshow(imagen_array)
axes[2].scatter(coordenadas_verdes_filtradas[1], coordenadas_verdes_filtradas[0], color='red', marker='o', s=10)
axes[2].set_title('Puntos Verdes Filtrados')
plt.show()

# Obtener las coordenadas de los tres puntos
try:
    x1, y1 = coordenadas_verdes_filtradas[1][0], coordenadas_verdes_filtradas[0][0]
    x2, y2 = coordenadas_verdes_filtradas[1][1], coordenadas_verdes_filtradas[0][1]
    x3, y3 = coordenadas_verdes_filtradas[1][2], coordenadas_verdes_filtradas[0][2]
    # Calcular el área del triángulo
    area = 0.5 * abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
    # Verificar si el área es mayor que cero
    if area > 0:
        print("Los tres puntos forman un triángulo.")
    else:
        print("Los tres puntos son colineales y no forman un triángulo.")
except IndexError:
    num_puntos = len(coordenadas_verdes_filtradas[0])
    print(f"Se encontraron {num_puntos} puntos verdes. Se necesitan al menos 3 para formar un triángulo.")
