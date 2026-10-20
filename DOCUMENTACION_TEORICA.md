# Documentación Teórica Magistral: Fundamentos de Clasificación y Métricas en Inteligencia Artificial

**Asignatura:** Introducción a la Inteligencia Artificial  
**Taller para la Nota III**  
**Docente / Científico de Datos:** Notas Pedagógicas Preparadas para **Kevin**  
**Versión del Entorno:** Conda (`intro_ia`) o Python Nativo (`venv`) | Python 3.11 | NumPy 2.x | Pandas 2.x | SciPy 1.x  

---

## Tabla de Contenido
1. [Introducción y Objetivos Pedagógicos](#1-introducción-y-objetivos-pedagógicos)
2. [Punto 1: Clasificación Lineal y Geometría de Hiperplanos](#2-punto-1-clasificación-lineal-y-geometría-de-hiperplanos)
   - 2.1. Definición Formal del Hiperplano de Decisión
   - 2.2. Parámetros del Taller: Vector de Pesos $\boldsymbol{w}$ y Sesgo $b$
   - 2.3. Función Discriminante y Regla de Decisión Binaria
   - 2.4. Distancia Ortogonal al Hiperplano (Margen Geométrico)
   - 2.5. Conexión con Modelos Clásicos (Perceptrón, SVM, Regresión Logística)
3. [Punto 2: Clasificación Bayesiana y Gaussianas Multivariadas (QDA)](#3-punto-2-clasificación-bayesiana-y-gaussianas-multivariadas-qda)
   - 3.1. Teoría de Decisión de Bayes y Minimización del Riesgo
   - 3.2. Modelo de Densidad Gaussiana Multivariada $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$
   - 3.3. Geometría de las Covarianzas: Varianzas, Correlaciones y Elipsoides
   - 3.4. Análisis Matemático de las Clases del Taller (Determinantes e Inversas)
   - 3.5. Distancia de Mahalanobis vs. Distancia Euclídea
   - 3.6. Derivación de la Función Discriminante Cuadrática (QDA)
   - 3.7. Naturaleza Cónica de las Fronteras de Decisión
4. [Punto 3: Búsqueda de Vecinos Más Cercanos y Métricas Espaciales](#4-punto-3-búsqueda-de-vecinos-más-cercanos-y-métricas-espaciales)
   - 4.1. Métrica Euclídea Formal y Espacio $\mathcal{L}_2$
   - 4.2. Formulación Vectorizada y Descomposición Polinomial
   - 4.3. Complejidad Algorítmica y Estrategias Top-$k$
5. [Nota del Taller: Ingesta de Datos desde Archivos CSV](#5-nota-del-taller-ingesta-de-datos-desde-archivos-csv)
   - 5.1. Módulo Estándar `csv` (Enfoque Nativo por Flujo)
   - 5.2. Módulo `numpy` (`np.loadtxt` y `np.genfromtxt`)
   - 5.3. Biblioteca `pandas` (`pd.read_csv`)
   - 5.4. Matriz Comparativa de Rendimiento y Casos de Uso
6. [Guía de Configuración y Ejecución](#6-guía-de-configuración-y-ejecución)
7. [Arquitectura de Parametrización y Desacoplamiento (Cero Datos Quemados)](#7-arquitectura-de-parametrización-y-desacoplamiento-cero-datos-quemados)

---

## 1. Introducción y Objetivos Pedagógicos

Estimado Kevin, como docente en Ciencia de Datos con una década formando ingenieros e investigadores, abordo este taller no como una mera colección de ejercicios aislados de programación, sino como un **recorrido pedagógico por los tres pilares estructurales del reconocimiento de patrones y el aprendizaje automático supervisado**:

1. **La perspectiva geométrica determinista:** Separación lineal de espacios vectoriales continuos mediante hiperplanos afines.
2. **La perspectiva probabilística generativa:** Modelado de incertidumbre y dispersión poblacional mediante densidades normales multivariadas bajo el marco óptimo de Bayes.
3. **La perspectiva métrica basada en instancias:** Agrupamiento y correspondencia no paramétrica a través de distancias normadas en espacios afines.
4. **La ingeniería de datos aplicada:** Lectura, saneamiento y transformación de persistencias tabulares `.csv`.

---

## 2. Punto 1: Clasificación Lineal y Geometría de Hiperplanos

### 2.1. Definición Formal del Hiperplano de Decisión

Consideremos un espacio euclidiano de características $d$-dimensional $\mathcal{X} = \mathbb{R}^d$, donde cada patrón o muestra se representa por un vector columna:

$$
\boldsymbol{x} = \begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_d
\end{pmatrix} \in \mathbb{R}^d
$$

Un **hiperplano afín** $H$ en $\mathbb{R}^d$ es un subespacio afín de dimensión $d - 1$ definido por el lugar geométrico de todos los puntos $\boldsymbol{x}$ que satisfacen la ecuación lineal:

$$
H = \{\boldsymbol{x} \in \mathbb{R}^d \mid \langle \boldsymbol{w}, \boldsymbol{x} \rangle + b = 0\}
$$

donde:
- $\boldsymbol{w} \in \mathbb{R}^d \setminus \{\boldsymbol{0}\}$ es el **vector de pesos** (*weight vector*). Geométricamente, $\boldsymbol{w}$ es el **vector ortogonal (normal)** a la superficie de decisión y apunta hacia el semi-espacio positivo.
- $b \in \mathbb{R}$ es el **sesgo** (*bias* o intercepto). Determina la traslación del hiperplano respecto al origen de coordenadas $\boldsymbol{0}$. Si $b = 0$, el hiperplano pasa obligatoriamente por el origen y constituye un subespacio vectorial canónico; si $b \neq 0$, es una variedad afín trasladada.
- $\langle \boldsymbol{w}, \boldsymbol{x} \rangle$ representa el **producto interno canónico** (producto punto o escalar):

$$
\langle \boldsymbol{w}, \boldsymbol{x} \rangle = \sum_{i=1}^d w_i x_i = \boldsymbol{w}^T \boldsymbol{x}
$$

---

### 2.2. Parámetros del Taller: Vector de Pesos $\boldsymbol{w}$ y Sesgo $b$

En el problema propuesto en el taller, el espacio de trabajo posee dimensión $d = 5$:

$$
\boldsymbol{w} = \begin{pmatrix}
\frac{1}{3} \\
1 \\
-3 \\
-2 \\
\frac{5}{4}
\end{pmatrix}
\approx
\begin{pmatrix}
0.3333 \\
1.0000 \\
-3.0000 \\
-2.0000 \\
1.2500
\end{pmatrix},
\qquad b = \frac{2}{3} \approx 0.6667
$$

#### Análisis de la Magnitud y Relevancia de las Características:
La norma euclidiana del vector $\boldsymbol{w}$ se calcula analíticamente como:

$$
\|\boldsymbol{w}\|_2 = \sqrt{\left(\frac{1}{3}\right)^2 + 1^2 + (-3)^2 + (-2)^2 + \left(\frac{5}{4}\right)^2} = \sqrt{\frac{1}{9} + 1 + 9 + 4 + \frac{25}{16}}
$$

$$
\|\boldsymbol{w}\|_2 = \sqrt{14 + \frac{1}{9} + \frac{25}{16}} = \sqrt{\frac{2016 + 16 + 225}{144}} = \sqrt{\frac{2257}{144}} = \frac{\sqrt{2257}}{12} \approx 3.95897
$$

Observa, Kevin, las sensibilidades relativas de cada variable:
- Las componentes $w_3 = -3$ y $w_4 = -2$ poseen las mayores magnitudes absolutas ($|w_3| = 3$, $|w_4| = 2$). Esto implica que pequeñas variaciones en las características $x_3$ y $x_4$ causan el mayor desplazamiento respecto al hiperplano.
- Como sus coeficientes son negativos, valores altos en $x_3$ y $x_4$ empujan violentamente a la muestra hacia el semi-espacio negativo ($\text{Clase } -1$).

---

### 2.3. Función Discriminante y Regla de Decisión Binaria

Definimos la función discriminante lineal continua $f: \mathbb{R}^5 \to \mathbb{R}$:

$$
f(\boldsymbol{x}) = \boldsymbol{w}^T \boldsymbol{x} + b = \frac{1}{3}x_1 + x_2 - 3x_3 - 2x_4 + \frac{5}{4}x_5 + \frac{2}{3}
$$

La regla de decisión binaria $\hat{y}: \mathbb{R}^5 \to \{+1, -1\}$ particiona el espacio en dos regiones disjuntas:

$$
\hat{y}(\boldsymbol{x}) = \text{sign}(f(\boldsymbol{x})) = \begin{cases}
+1, & \text{si } f(\boldsymbol{x}) \ge 0 \\
-1, & \text{si } f(\boldsymbol{x}) < 0
\end{cases}
$$

---

### 2.4. Distancia Ortogonal al Hiperplano (Margen Geométrico)

Un concepto indispensable en Machine Learning moderno es la **distancia euclidiana con signo** $r(\boldsymbol{x})$ desde un punto cualquiera $\boldsymbol{x}$ hasta la superficie $H$.

Tomemos un punto cualquiera $\boldsymbol{x}_p$ perteneciente al hiperplano (es decir, $\boldsymbol{w}^T \boldsymbol{x}_p + b = 0$). Descomponemos $\boldsymbol{x}$ en su proyección ortogonal $\boldsymbol{x}_p$ más una componente a lo largo del vector unitario normal:

$$
\boldsymbol{x} = \boldsymbol{x}_p + r(\boldsymbol{x}) \frac{\boldsymbol{w}}{\|\boldsymbol{w}\|_2}
$$

Multiplicando escalarmente por $\boldsymbol{w}^T$ y sumando $b$:

$$
\boldsymbol{w}^T \boldsymbol{x} + b = \left( \boldsymbol{w}^T \boldsymbol{x}_p + b \right) + r(\boldsymbol{x}) \frac{\boldsymbol{w}^T \boldsymbol{w}}{\|\boldsymbol{w}\|_2}
$$

Como $\boldsymbol{w}^T \boldsymbol{x}_p + b = 0$, se simplifica a:

$$
f(\boldsymbol{x}) = r(\boldsymbol{x}) \frac{\|\boldsymbol{w}\|_2^2}{\|\boldsymbol{w}\|_2} = r(\boldsymbol{x}) \|\boldsymbol{w}\|_2
$$

Despejando la distancia con signo $r(\boldsymbol{x})$:

$$
r(\boldsymbol{x}) = \frac{f(\boldsymbol{x})}{\|\boldsymbol{w}\|_2} = \frac{\langle \boldsymbol{w}, \boldsymbol{x} \rangle + b}{\|\boldsymbol{w}\|_2}
$$

- Si $r(\boldsymbol{x}) > 0$, la muestra reside en el semi-espacio hacia el cual apunta $\boldsymbol{w}$.
- La distancia absoluta $|r(\boldsymbol{x})|$ mide qué tan confiable es la clasificación: muestras con $|r(\boldsymbol{x})| \gg 0$ están profundamente inmersas en su clase, mientras que muestras con $|r(\boldsymbol{x})| \approx 0$ están en la zona crítica de incertidumbre.

---

### 2.5. Conexión con Modelos Clásicos

1. **Perceptrón de Rosenblatt (1958):** Actualiza el vector $\boldsymbol{w} \leftarrow \boldsymbol{w} + \eta y_i \boldsymbol{x}_i$ y $b \leftarrow b + \eta y_i$ únicamente cuando una muestra es mal clasificada.
2. **Máquinas de Vectores de Soporte (SVM - Vapnik, 1995):** Buscan el hiperplano óptimo que maximiza el margen geométrico mínimo $\frac{2}{\|\boldsymbol{w}\|_2}$, resolviendo un problema cuadrático convexo.
3. **Regresión Logística:** Mapea el score lineal a una probabilidad condicional a través de la sigmoide:

$$
P(Y = +1 \mid \boldsymbol{x}) = \frac{1}{1 + \exp(-(\boldsymbol{w}^T \boldsymbol{x} + b))}
$$

---

## 3. Punto 2: Clasificación Bayesiana y Gaussianas Multivariadas (QDA)

### 3.1. Teoría de Decisión de Bayes y Minimización del Riesgo

En el paradigma estadístico generativo, modelamos el mecanismo probabilístico subyacente que genera los datos.

Bajo la regla de Bayes, la probabilidad posterior de que una muestra observada $\boldsymbol{x}$ pertenezca a la clase $C_k$ es:

$$
P(C_k \mid \boldsymbol{x}) = \frac{p(\boldsymbol{x} \mid C_k) P(C_k)}{p(\boldsymbol{x})} = \frac{p(\boldsymbol{x} \mid C_k) P(C_k)}{\sum_{j=1}^K p(\boldsymbol{x} \mid C_j) P(C_j)}
$$

donde:
- $P(C_k)$ es la **probabilidad a priori** (*prior*) de la clase $C_k$.
- $p(\boldsymbol{x} \mid C_k)$ es la **función de verosimilitud** (*likelihood*).
- $p(\boldsymbol{x}) = \sum_{j=1}^K p(\boldsymbol{x} \mid C_j) P(C_j)$ es la **evidencia** (factor de normalización).
- $P(C_k \mid \boldsymbol{x})$ es la **probabilidad a posteriori**, base fundamental de la decisión.

Para minimizar la probabilidad total de error de clasificación (Riesgo de Bayes con costo 0-1), asignamos la muestra a la clase con mayor probabilidad posterior:

$$
\hat{k} = \arg\max_{k \in \{1, \dots, K\}} P(C_k \mid \boldsymbol{x}) \iff \arg\max_{k \in \{1, \dots, K\}} p(\boldsymbol{x} \mid C_k) P(C_k)
$$

---

### 3.2. Modelo de Densidad Gaussiana Multivariada $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

En $d$ dimensiones, una variable aleatoria continua $\boldsymbol{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ tiene como función de densidad de probabilidad (FDP):

$$
p(\boldsymbol{x}) = \frac{1}{(2\pi)^{d/2} \det(\boldsymbol{\Sigma})^{1/2}} \exp\left( -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right)
$$

donde:
- $\boldsymbol{\mu} \in \mathbb{R}^d$ es el **vector de medias** (centro de gravedad de la distribución).
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ es la **matriz de covarianza**, simétrica y definida positiva ($\boldsymbol{\Sigma} \succ 0$).

---

### 3.3. Geometría de las Covarianzas: Varianzas, Correlaciones y Elipsoides

En dos dimensiones ($d = 2$), la matriz de covarianza tiene la forma:

$$
\boldsymbol{\Sigma} = \begin{pmatrix}
\sigma_1^2 & \sigma_{12} \\
\sigma_{21} & \sigma_2^2
\end{pmatrix}
= \begin{pmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{pmatrix}
$$

Los lugares geométricos de densidad constante vienen dados por:

$$
(\boldsymbol{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) = c^2
$$

Esta ecuación describe un **elipsoide** en $\mathbb{R}^d$:
1. Los ejes principales de la elipse están alineados con los **autovectores** $\boldsymbol{v}_1, \boldsymbol{v}_2$ de $\boldsymbol{\Sigma}$.
2. Las longitudes de los semiejes son proporcionales a las raíces cuadradas de los **autovalores** $\sqrt{\lambda_1}, \sqrt{\lambda_2}$.
3. El signo de la covarianza cruzada $\sigma_{12}$ determina la orientación:
   - $\sigma_{12} > 0$: Elipse orientada con pendiente positiva.
   - $\sigma_{12} < 0$: Elipse orientada con pendiente negativa.
   - $\sigma_{12} = 0$: Ejes de la elipse perfectamente paralelos a los ejes canónicos.

---

### 3.4. Análisis Matemático de las Clases del Taller

Para el cálculo analítico de los determinantes e inversas de matrices de covarianza de dimensión $2 \times 2$, empleamos la definición general del álgebra matricial:

Dada una matriz general invertible de orden 2:

$$
\boldsymbol{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}
$$

Su determinante y su matriz inversa vienen dados analíticamente por:

$$
\det(\boldsymbol{A}) = ad - bc
$$

$$
\boldsymbol{A}^{-1} = \frac{1}{\det(\boldsymbol{A})} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

A continuación, aplicamos este procedimiento paso a paso a cada una de las 3 clases del taller:

---

#### **Clase 1:**

**Parámetros dados:**

$$
\boldsymbol{\mu}_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \qquad \boldsymbol{\Sigma}_1 = \begin{pmatrix} 1.0 & 0.5 \\ 0.5 & 1.0 \end{pmatrix}
$$

**1. Determinante:**

$$
\det(\boldsymbol{\Sigma}_1) = (1.0)(1.0) - (0.5)(0.5) = 1.0 - 0.25 = 0.75 = \frac{3}{4} > 0
$$

**2. Matriz Inversa:**

Aplicando la fórmula con $a=1.0$, $b=0.5$, $c=0.5$, $d=1.0$:

$$
\boldsymbol{\Sigma}_1^{-1} = \frac{1}{\det(\boldsymbol{\Sigma}_1)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

$$
\boldsymbol{\Sigma}_1^{-1} = \frac{1}{0.75} \begin{pmatrix} 1.0 & -0.5 \\ -0.5 & 1.0 \end{pmatrix} = \begin{pmatrix} \frac{4}{3} & -\frac{2}{3} \\ -\frac{2}{3} & \frac{4}{3} \end{pmatrix} \approx \begin{pmatrix} 1.3333 & -0.6667 \\ -0.6667 & 1.3333 \end{pmatrix}
$$

**3. Interpretación física y geométrica:**

Centrada en $(0, 0)$. Como $\sigma_{12} = 0.5 > 0$, las dos variables exhiben covarianza positiva; la nube de puntos se expande a lo largo de la diagonal principal ($x_2 \approx x_1$).

---

#### **Clase 2:**

**Parámetros dados:**

$$
\boldsymbol{\mu}_2 = \begin{pmatrix} 4 \\ 4 \end{pmatrix}, \qquad \boldsymbol{\Sigma}_2 = \begin{pmatrix} 1.0 & -0.3 \\ -0.3 & 1.0 \end{pmatrix}
$$

**1. Determinante:**

$$
\det(\boldsymbol{\Sigma}_2) = (1.0)(1.0) - (-0.3)(-0.3) = 1.0 - 0.09 = 0.91 = \frac{91}{100} > 0
$$

**2. Matriz Inversa:**

Aplicando la fórmula con $a=1.0$, $b=-0.3$, $c=-0.3$, $d=1.0$:

$$
\boldsymbol{\Sigma}_2^{-1} = \frac{1}{\det(\boldsymbol{\Sigma}_2)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

$$
\boldsymbol{\Sigma}_2^{-1} = \frac{1}{0.91} \begin{pmatrix} 1.0 & 0.3 \\ 0.3 & 1.0 \end{pmatrix} \approx \begin{pmatrix} 1.0989 & 0.3297 \\ 0.3297 & 1.0989 \end{pmatrix}
$$

**3. Interpretación física y geométrica:**

Centrada en $(4, 4)$. Como $\sigma_{12} = -0.3 < 0$, las variables tienen correlación negativa moderada; la elipse se inclina con pendiente negativa (a lo largo de la diagonal secundaria).

---

#### **Clase 3:**

**Parámetros dados:**

$$
\boldsymbol{\mu}_3 = \begin{pmatrix} -4 \\ 3 \end{pmatrix}, \qquad \boldsymbol{\Sigma}_3 = \begin{pmatrix} 0.5 & 0.0 \\ 0.0 & 2.0 \end{pmatrix}
$$

**1. Determinante:**

$$
\det(\boldsymbol{\Sigma}_3) = (0.5)(2.0) - (0.0)(0.0) = 1.0
$$

**2. Matriz Inversa:**

Dado que $\boldsymbol{\Sigma}_3$ es una matriz diagonal ($b = c = 0$), su inversa es simplemente la inversa de cada elemento de la diagonal:

$$
\boldsymbol{\Sigma}_3^{-1} = \begin{pmatrix} \frac{1}{\sigma_1^2} & 0 \\ 0 & \frac{1}{\sigma_2^2} \end{pmatrix} = \begin{pmatrix} \frac{1}{0.5} & 0.0 \\ 0.0 & \frac{1}{2.0} \end{pmatrix} = \begin{pmatrix} 2.0 & 0.0 \\ 0.0 & 0.5 \end{pmatrix}
$$

**3. Interpretación física y geométrica:**

Centrada en $(-4, 3)$. Al ser diagonal ($\sigma_{12} = 0$), las variables son estadísticamente independientes. La varianza vertical $\sigma_2^2 = 2.0$ es cuatro veces la varianza horizontal $\sigma_1^2 = 0.5$, produciendo una elipse elongada verticalmente sobre el eje $x_2$.

---

### 3.5. Distancia de Mahalanobis vs. Distancia Euclídea

La distancia euclidiana tradicional $d_E(\boldsymbol{x}, \boldsymbol{\mu}) = \sqrt{(\boldsymbol{x} - \boldsymbol{\mu})^T (\boldsymbol{x} - \boldsymbol{\mu})}$ asume esfericidad perfecta.

La **distancia de Mahalanobis** $D_{M, k}(\boldsymbol{x})$ corrige este sesgo al ponderar por la dispersión y correlación de cada clase:

$$
D_{M, k}(\boldsymbol{x}) = \sqrt{(\boldsymbol{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_k)}
$$

Geométricamente, la distancia de Mahalanobis:
1. Centra los datos restando $\boldsymbol{\mu}_k$.
2. Rota los ejes al sistema de autovectores de la covarianza (descorrelación).
3. Escala cada eje por el inverso de la desviación estándar $\frac{1}{\sqrt{\lambda_i}}$ (*whitening*).

---

### 3.6. Derivación de la Función Discriminante Cuadrática (QDA)

Tomando el logaritmo natural monótono creciente de la regla de Bayes:

$$
g_k(\boldsymbol{x}) = \ln\left( p(\boldsymbol{x} \mid C_k) P(C_k) \right) = \ln p(\boldsymbol{x} \mid C_k) + \ln P(C_k)
$$

Sustituyendo la densidad gaussiana multivariada:

$$
g_k(\boldsymbol{x}) = \ln\left( \frac{1}{(2\pi)^{d/2} \det(\boldsymbol{\Sigma}_k)^{1/2}} \exp\left(-\frac{1}{2} D_{M, k}^2(\boldsymbol{x})\right) \right) + \ln P(C_k)
$$

$$
g_k(\boldsymbol{x}) = -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_k) - \frac{1}{2} \ln \det(\boldsymbol{\Sigma}_k) - \frac{d}{2}\ln(2\pi) + \ln P(C_k)
$$

Eliminando la constante común $-\frac{d}{2}\ln(2\pi)$, obtenemos la función canónica de **QDA**:

$$
g_k(\boldsymbol{x}) = -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_k) - \frac{1}{2} \ln \det(\boldsymbol{\Sigma}_k) + \ln P(C_k)
$$

---

### 3.7. Naturaleza Cónica de las Fronteras de Decisión

La frontera de decisión entre dos clases $C_i$ y $C_j$ ocurre donde sus funciones discriminantes se igualan:

$$
g_i(\boldsymbol{x}) = g_j(\boldsymbol{x}) \iff g_i(\boldsymbol{x}) - g_j(\boldsymbol{x}) = 0
$$

Desarrollando los términos cuadráticos y agrupando potencias de $\boldsymbol{x}$:

$$
\boldsymbol{x}^T \left( \boldsymbol{\Sigma}_j^{-1} - \boldsymbol{\Sigma}_i^{-1} \right) \boldsymbol{x} + 2 \left( \boldsymbol{\Sigma}_i^{-1} \boldsymbol{\mu}_i - \boldsymbol{\Sigma}_j^{-1} \boldsymbol{\mu}_j \right)^T \boldsymbol{x} + C_{ij} = 0
$$

- **Caso LDA (Linear Discriminant Analysis):** Si todas las covarianzas fuesen idénticas ($\boldsymbol{\Sigma}_i = \boldsymbol{\Sigma}_j$), el término cuadrático $\boldsymbol{x}^T (\boldsymbol{\Sigma}_j^{-1} - \boldsymbol{\Sigma}_i^{-1}) \boldsymbol{x}$ se anularía, generando fronteras rectas.
- **Caso QDA (Nuestro Taller):** Dado que $\boldsymbol{\Sigma}_1 \neq \boldsymbol{\Sigma}_2 \neq \boldsymbol{\Sigma}_3$, la matriz cuadrática $(\boldsymbol{\Sigma}_j^{-1} - \boldsymbol{\Sigma}_i^{-1}) \neq \boldsymbol{0}$. Por consiguiente, **las superficies de separación son curvas cuadráticas o cónicas en 2D (arcos hiperbólicos, parábolas o elipses)**.

---

## 4. Punto 3: Búsqueda de Vecinos Más Cercanos y Métricas Espaciales

### 4.1. Métrica Euclídea Formal y Espacio $\mathcal{L}_2$

Dado un espacio vectorial métrico $\mathbb{R}^d$, una **función métrica** $d: \mathbb{R}^d \times \mathbb{R}^d \to [0, \infty)$ satisface:
1. **No negatividad:** $d(\boldsymbol{p}, \boldsymbol{q}) \ge 0$
2. **Identidad de indiscernibles:** $d(\boldsymbol{p}, \boldsymbol{q}) = 0 \iff \boldsymbol{p} = \boldsymbol{q}$
3. **Simetría:** $d(\boldsymbol{p}, \boldsymbol{q}) = d(\boldsymbol{q}, \boldsymbol{p})$
4. **Desigualdad triangular:** $d(\boldsymbol{p}, \boldsymbol{z}) \le d(\boldsymbol{p}, \boldsymbol{q}) + d(\boldsymbol{q}, \boldsymbol{z})$

La **distancia euclídea** es la norma inducida por el producto interno estándar:

$$
d_2(\boldsymbol{p}, \boldsymbol{q}) = \|\boldsymbol{p} - \boldsymbol{q}\|_2 = \sqrt{\sum_{k=1}^d (p_k - q_k)^2}
$$

---

### 4.2. Formulación Vectorizada y Descomposición Polinomial

Para un lote de consultas $\boldsymbol{Q} \in \mathbb{R}^{M \times d}$ y un conjunto de referencia $\boldsymbol{X} \in \mathbb{R}^{N \times d}$, implementamos dos estrategias vectorizadas:

#### Estrategia A: Broadcasting Tensorial en Memoria
Expandiendo los ejes mediante dimensiones ficticias:

$$
\boldsymbol{Q}_{M \times 1 \times d} - \boldsymbol{X}_{1 \times N \times d} \to \boldsymbol{\Delta}_{M \times N \times d}
$$

$$
\boldsymbol{D}_{M \times N} = \sqrt{\sum_{k=1}^d \boldsymbol{\Delta}_{i, j, k}^2}
$$

#### Estrategia B: Descomposición Algebraica Rápida
Desarrollando el binomio cuadrado:

$$
\|\boldsymbol{q}_i - \boldsymbol{x}_j\|^2 = \|\boldsymbol{q}_i\|^2 + \|\boldsymbol{x}_j\|^2 - 2 \langle \boldsymbol{q}_i, \boldsymbol{x}_j \rangle
$$

En álgebra matricial compacta:

$$
\boldsymbol{D}^2 = \text{diag}(\boldsymbol{Q} \boldsymbol{Q}^T) \boldsymbol{1}_N^T + \boldsymbol{1}_M \text{diag}(\boldsymbol{X} \boldsymbol{X}^T)^T - 2 \boldsymbol{Q} \boldsymbol{X}^T
$$

---

### 4.3. Complejidad Algorítmica y Estrategias Top-$k$

Para encontrar los $k$ vecinos más cercanos a partir de la fila de distancias de longitud $N$:
- **Ordenamiento completo (`np.argsort`):** Complejidad temporal $O(N \log N)$.
- **Partición lineal (`np.argpartition`):** Basada en *Introselect*, localiza los $k$ menores elementos en tiempo promedio $O(N)$, ordenando únicamente los $k$ seleccionados en $O(k \log k)$.

---

## 5. Nota del Taller: Ingesta de Datos desde Archivos CSV

La nota al final del enunciado del taller solicita: *"Estudiar la forma como se pueden cargar datos a Python a partir de archivos .csv"*.

En el módulo educativo [src/utils_csv.py](file:///Users/kevingiraldocastaneda/Proyectos/intro.IA/src/utils_csv.py) implementamos y documentamos las tres aproximaciones que todo científico de datos debe dominar:

### 5.1. Módulo Estándar `csv` (Enfoque Nativo por Flujo)
```python
import csv

with open("datos.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",")
    encabezado = next(reader)
    datos = [[float(val) for val in fila] for fila in reader if fila]
```

### 5.2. Módulo `numpy` (`np.loadtxt` y `np.genfromtxt`)
```python
import numpy as np

matriz = np.genfromtxt("datos.csv", delimiter=",", skip_header=1, dtype=np.float64)
```

### 5.3. Biblioteca `pandas` (`pd.read_csv`)
```python
import pandas as pd

df = pd.read_csv("datos.csv", delimiter=",")
X = df[["x_1", "x_2"]].to_numpy(dtype=np.float64)
```

### 5.4. Matriz Comparativa de Rendimiento y Casos de Uso

| Característica | Módulo Nativo `csv` | NumPy `genfromtxt` | Pandas `pd.read_csv` |
| :--- | :--- | :--- | :--- |
| **Dependencias** | Ninguna (Built-in) | Requiere NumPy | Requiere Pandas |
| **Velocidad de Carga** | Baja (Loops en Python) | Alta (C-Speed) | Muy Alta (C/SIMD multihilo) |
| **Consumo de Memoria** | Muy bajo (iteradores) | Bajo (ndarray contiguo) | Moderado (metadatos de índices) |
| **Inferencia de Tipos** | Manual (`float(x)`) | Semi-manual (`dtype`) | Automática |
| **Manejo de NaN/Nulos** | Manual por excepción | Parámetro `filling_values` | Métodos robustos integrados |
| **Caso de Uso Ideal** | Lambdas, scripts sin dependencias | Matrices algebraicas numéricas puras | Pipelines de Ciencia de Datos y ML |

---

## 6. Guía de Configuración y Ejecución

Tienes dos alternativas completamente funcionales:

### Opción A: Con Conda
```bash
conda activate intro_ia
python main.py
```

### Opción B: Con Python Nativo `venv` (Sin Conda)
```bash
source .venv/bin/activate       # En macOS / Linux
.venv\Scripts\Activate.ps1      # En Windows
python main.py
```

### Modo Interactivo de Inferencia (Predicción)
```bash
python main.py --interactivo
```

### Validación con Pruebas Unitarias
```bash
pytest tests/test_taller.py -v
```

---

## 7. Arquitectura de Parametrización y Desacoplamiento (Cero Datos Quemados)

Para cumplir con los más exigentes estándares de ingeniería de software en Ciencia de Datos, el proyecto elimina **cualquier valor quemado (*hardcoded data*)** en el núcleo de cálculo:

1. **Clasificador Lineal Generalizado ($d \ge 1$):**
   - El vector de pesos $\boldsymbol{w}$ y el sesgo/beta $b$ admiten cualquier dimensión $d$.
   - La visualización en 2D selecciona automáticamente las dos variables con mayor magnitud absoluta de peso $|w_i|$ y colapsa analíticamente las $d - 2$ dimensiones restantes a través de sus valores promedio $\sum_{j \neq i_1, i_2} w_j \bar{x}_j$, trazando la curva de corte exacta.
2. **Clasificador Bayesiano Multiclase ($K \ge 2$):**
   - Soporta cualquier cantidad arbitraria de clases $C_1, \dots, C_K$.
   - Los límites del gráfico, la paleta de colores (`tab10`/`viridis`), las isolíneas de densidad elíptica y las curvas de decisión entre pares $g_i(\boldsymbol{x}) = g_j(\boldsymbol{x})$ se generan de forma completamente dinámica.
3. **Persistencia y Control Vía `config.json`:**
   - La estructura de dataclasses en [config.py](file:///Users/kevingiraldocastaneda/Proyectos/intro.IA/config.py) permite serializar y deserializar configuraciones completas de experimentos, habilitando pruebas A/B y benchmarking sin alterar el código fuente.
