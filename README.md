# Taller para la Nota III - Introducción a la Inteligencia Artificial

Repositorio desarrollado para **Kevin** con soluciones pedagógicas, rigurosas y verificadas matemáticamente para el taller de la asignatura.

> **¡Totalmente Parametrizado (Cero Datos Quemados)!**  
> Todos los hiperparámetros (pesos $\boldsymbol{w}$, sesgos/betas $b$, número de clases gaussianas $K$, vectores de medias $\boldsymbol{\mu}$, matrices de covarianza $\boldsymbol{\Sigma}$, priors, dimensiones $d$ y número de vecinos $k$) están desacoplados en el módulo [config.py](file:///Users/kevingiraldocastaneda/Proyectos/intro.IA/config.py) y en el archivo serializable [config.json](file:///Users/kevingiraldocastaneda/Proyectos/intro.IA/config.json). Puedes inyectar nuevos experimentos y clases sin modificar una sola línea de código lógico.

---

## Estructura del Proyecto

```text
intro.IA/
├── environment.yml                     # Especificación del entorno Conda (intro_ia)
├── config.py                           # Módulo central de configuración y dataclasses
├── config.json                         # Configuración desacoplada en formato JSON (editable)
├── README.md                           # Guía rápida de usuario y comandos
├── DOCUMENTACION_TEORICA.md            # Documento magistral con explicaciones teóricas y matemáticas
├── main.py                             # Script orquestador principal 100% parametrizado
├── docs/
│   └── taller.png                      # Enunciado original del taller
├── data/                               # Datasets generados y archivos CSV de prueba
│   ├── punto1_muestras.csv             # 25 muestras en R^5 para clasificación lineal
│   ├── punto2_muestras.csv             # 45 muestras en R^2 para clasificación bayesiana
│   ├── punto3_referencia.csv           # 30 muestras en R^2 como base de conocimiento
│   └── punto3_consultas.csv            # 5 muestras de consulta (queries)
├── src/                                # Código fuente modular y comentado
│   ├── __init__.py
│   ├── utils_csv.py                    # Estudio de carga CSV (nativo, numpy, pandas)
│   ├── punto1_clasificador_lineal.py   # Punto 1: Hiperplano f(x) = <w, x> + b (Parametrizado)
│   ├── punto2_clasificador_gaussiano.py# Punto 2: Clasificador Bayesiano Multivariado (K clases QDA)
│   └── punto3_distancia_euclidea.py    # Punto 3: Búsqueda k-NN con distancia euclídea
├── tests/
│   └── test_taller.py                  # Pruebas unitarias automatizadas con pytest (14 tests)
└── outputs/                            # Resultados generados (reportes CSV y gráficas PNG)
    ├── punto1_resultados.csv
    ├── punto1_clasificacion.png
    ├── punto2_resultados.csv
    ├── punto2_bayes_regiones.png
    ├── punto3_resultados.csv
    └── punto3_vecinos_cercanos.png
```

---

## Gestión de Ambientes

Tienes dos opciones totalmente soportadas para ejecutar este proyecto:

### Opción A: Con Conda (Recomendado si ya usas Anaconda/Miniconda)
```bash
conda activate intro_ia
```
*(Para recrearlo desde cero: `conda env create -f environment.yml`)*

### Opción B: Con Python Nativo `venv` (Sin Conda)
```bash
# 1. Crear entorno virtual
python3 -m venv .venv

# 2. Activar entorno
source .venv/bin/activate       # En macOS / Linux
.venv\Scripts\Activate.ps1      # En Windows PowerShell

# 3. Instalar librerías
pip install -r requirements.txt
```
👉 Para una guía detallada paso a paso sobre el uso de `venv` en cualquier sistema operativo, consulta: [**GUIA_CONFIGURACION_SIN_CONDA.md**](file:///Users/kevingiraldocastaneda/Proyectos/intro.IA/GUIA_CONFIGURACION_SIN_CONDA.md)

---

## Cómo Ejecutar el Proyecto

### 1. Ejecución Principal Parametrizada
Por defecto, `main.py` lee automáticamente el archivo `config.json`:
```bash
python main.py
```

### 2. Ejecutar con una Configuración Personalizada (Nuevas Clases, Pesos o Betas)
Para correr un experimento con tus propios pesos, betas o nuevas clases gaussianas:
```bash
python main.py --config mi_experimento.json
```

Si deseas regenerar el archivo `config.json` con los valores por defecto del taller:
```bash
python main.py --guardar_config_defecto
```

### 3. Ejecutar Cada Punto de Manera Individual con Argumentos CLI

- **Punto 1 (Clasificador Lineal):**
  Admite cualquier dimensión $d$, vector de pesos $\boldsymbol{w}$ y sesgo $\beta$ vía argumentos:
  ```bash
  python src/punto1_clasificador_lineal.py --w "0.5,-1.2,3.0" --b 1.5 --csv data/punto1_muestras.csv
  ```

- **Punto 2 (Clasificador Bayesiano Gaussiano QDA):**
  Admite $K \ge 2$ clases arbitrarias y cualquier matriz de covarianza definida positiva:
  ```bash
  python src/punto2_clasificador_gaussiano.py --config config.json
  ```

- **Punto 3 (Búsqueda con Distancia Euclídea):**
  Admite cualquier valor de $k$, características e identificadores:
  ```bash
  python src/punto3_distancia_euclidea.py --ref_csv data/punto3_referencia.csv --query_csv data/punto3_consultas.csv -k 3
  ```

---

## Modo Inferencia: Predicción para Nuevos Ejemplos

Puedes clasificar cualquier muestra arbitraria que te den, ya sea por línea de comandos o mediante el menú interactivo:

### A. Modo Interactivo por Consola
Abre un menú que te solicitará las coordenadas de la muestra y te entregará el reporte matemático completo:
```bash
python main.py --interactivo
```

### B. Predicción Rápida por Línea de Comandos (CLI)

1. **Predecir muestra con Clasificador Lineal (Punto 1):**
   ```bash
   python main.py --predecir_p1 "1.0, 2.0, -1.0, 0.5, 3.0"
   ```
2. **Predecir muestra con Clasificador Bayesiano Gaussiano QDA (Punto 2):**
   ```bash
   python main.py --predecir_p2 "4.2, 3.9"
   ```
3. **Buscar vecinos más cercanos de una muestra (Punto 3):**
   ```bash
   python main.py --predecir_p3 "0.0, 3.0" -k 3
   ```

---

## Verificación de Calidad y Pruebas Unitarias

El proyecto cuenta con 17 pruebas unitarias que validan:
1. Exactitud de las fórmulas del taller ($\boldsymbol{w}, b, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$).
2. Funcionamiento con dimensiones arbitrarias ($d=3$) y pesos/betas personalizados.
3. Clasificación Bayesiana generalizada para $K=4$ clases gaussianas.
4. Serialización y deserialización de configuraciones JSON.
5. Inferencia y predicción de muestras individuales para los 3 modelos.

```bash
pytest tests/test_taller.py -v
```
*Resultado: 17 passed in 1.02s.*


---

## Documentación Teórica Detallada

Para una explicación exhaustiva con derivaciones matemáticas, deducción del Análisis Discriminante Cuadrático (QDA), geometría de elipsoides de covarianza y análisis de complejidad computacional, consulta el archivo:
👉 [DOCUMENTACION_TEORICA.md](file:///Users/kevingiraldocastaneda/Proyectos/intro.IA/DOCUMENTACION_TEORICA.md)
