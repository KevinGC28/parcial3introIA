"""
=============================================================================
MÓDULO PEDAGÓGICO: CARGA Y MANIPULACIÓN DE ARCHIVOS CSV EN PYTHON
=============================================================================
Autor: Creado para Kevin - Introducción a la Inteligencia Artificial
Propósito: Estudiar a fondo las distintas metodologías para importar datos
           desde archivos de valores separados por comas (.csv) a estructuras
           de Python (listas nativas, arrays de NumPy y DataFrames de Pandas).
=============================================================================
"""

import csv
from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Enfoque Nativo: Módulo estándar `csv` de Python
# ---------------------------------------------------------------------------
def cargar_csv_nativo(
    filepath: Union[str, Path], 
    tiene_encabezado: bool = True,
    delimitador: str = ","
) -> Tuple[List[str], List[List[float]]]:
    """
    Carga un archivo CSV utilizando únicamente la biblioteca estándar `csv`.
    
    ¿Por qué estudiarlo?
    - Es independiente de bibliotecas externas (útil en microservicios ligeros o lambdas).
    - Permite comprender el procesamiento secuencial línea por línea (ideal para archivos
      masivos que no caben en memoria RAM mediante iteradores).
      
    Parámetros:
    -----------
    filepath : str o Path
        Ruta del archivo .csv a leer.
    tiene_encabezado : bool, opcional (default=True)
        Indica si la primera fila corresponde a los nombres de las columnas.
    delimitador : str, opcional (default=',')
        Carácter que separa las columnas (comas, punto y coma, tabuladores, etc.).
        
    Retorna:
    --------
    encabezados : List[str]
        Lista con los nombres de las columnas.
    datos : List[List[float]]
        Matriz numérica como lista de listas de flotantes.
    """
    ruta = Path(filepath)
    if not ruta.is_file():
        raise FileNotFoundError(f"El archivo {filepath} no existe.")

    encabezados: List[str] = []
    datos: List[List[float]] = []

    with open(ruta, mode="r", encoding="utf-8") as archivo:
        lector = csv.reader(archivo, delimiter=delimitador)
        
        # Procesar encabezado si existe
        if tiene_encabezado:
            encabezados = next(lector, [])
            
        # Procesar cada fila convirtiendo las cadenas a números flotantes
        for num_linea, fila in enumerate(lector, start=2 if tiene_encabezado else 1):
            # Omitir líneas vacías
            if not fila or all(c.strip() == "" for c in fila):
                continue
            try:
                fila_numerica = [float(valor.strip()) for valor in fila]
                datos.append(fila_numerica)
            except ValueError as error:
                raise ValueError(
                    f"Error de conversión en la línea {num_linea}: no se pudo "
                    f"convertir el elemento a float. Detalle: {error}"
                )

    return encabezados, datos


# ---------------------------------------------------------------------------
# 2. Enfoque Numérico Rápido: NumPy (`np.loadtxt` y `np.genfromtxt`)
# ---------------------------------------------------------------------------
def cargar_csv_numpy(
    filepath: Union[str, Path],
    delimitador: str = ",",
    omitir_filas: int = 1
) -> np.ndarray:
    """
    Carga una matriz numérica desde un archivo CSV utilizando NumPy en C.
    
    ¿Por qué estudiarlo?
    - Es extremadamente rápido para datos puramente numéricos.
    - Asigna directamente un bloque contiguo de memoria (ndarrays), óptimo para álgebra lineal.
    
    Parámetros:
    -----------
    filepath : str o Path
        Ruta del archivo CSV.
    delimitador : str, opcional (default=',')
        Carácter separador de columnas.
    omitir_filas : int, opcional (default=1)
        Número de filas iniciales a omitir (normalmente 1 por los encabezados).
        
    Retorna:
    --------
    np.ndarray : Matriz bidimensional de tipo float64.
    """
    ruta = Path(filepath)
    if not ruta.is_file():
        raise FileNotFoundError(f"El archivo {filepath} no existe.")

    # genfromtxt es más robusto que loadtxt ante posibles valores faltantes o formatos mixtos
    matriz = np.genfromtxt(ruta, delimiter=delimitador, skip_header=omitir_filas, dtype=np.float64)
    return matriz


# ---------------------------------------------------------------------------
# 3. Enfoque Estándar en Ciencia de Datos: Pandas (`pd.read_csv`)
# ---------------------------------------------------------------------------
def cargar_csv_pandas(
    filepath: Union[str, Path],
    delimitador: str = ",",
    columnas_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Carga un archivo CSV utilizando Pandas, la herramienta estándar en Data Science.
    
    ¿Por qué estudiarlo?
    - Infiere automáticamente los tipos de datos (dtypes).
    - Permite indexación sofisticada, manipulación de nombres de columnas y limpieza de nulos.
    - Facilita la interoperabilidad con NumPy mediante `.to_numpy()`.
    
    Parámetros:
    -----------
    filepath : str o Path
        Ruta del archivo CSV.
    delimitador : str, opcional (default=',')
        Separador de columnas.
    columnas_features : List[str], opcional
        Lista explícita de nombres de columnas que contienen las características numéricas.
        
    Retorna:
    --------
    df : pd.DataFrame
        DataFrame completo con metadatos.
    X : np.ndarray
        Matriz de características numéricas lista para cálculos algebraicos.
    """
    ruta = Path(filepath)
    if not ruta.is_file():
        raise FileNotFoundError(f"El archivo {filepath} no existe.")

    df = pd.read_csv(ruta, delimiter=delimitador)
    
    if columnas_features is not None:
        X = df[columnas_features].to_numpy(dtype=np.float64)
    else:
        # Si no se especifican columnas, seleccionar automáticamente las columnas numéricas
        X = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)

    return df, X


# ---------------------------------------------------------------------------
# Generador de Datasets Sintéticos para Demostración y Validación
# ---------------------------------------------------------------------------
def generar_datasets_taller(data_dir: Union[str, Path] = "data") -> None:
    """
    Genera archivos .csv de prueba bien estructurados para los 3 ejercicios del taller.
    """
    dir_path = Path(data_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed=42)

    # 1. Dataset para Punto 1: 5 dimensiones (w tiene 5 componentes)
    # Generamos 20 muestras en R^5
    n_muestras_p1 = 25
    X_p1 = rng.uniform(low=-3.0, high=3.0, size=(n_muestras_p1, 5))
    df_p1 = pd.DataFrame(X_p1, columns=[f"x_{i+1}" for i in range(5)])
    df_p1.to_csv(dir_path / "punto1_muestras.csv", index=False)

    # 2. Dataset para Punto 2: 2 dimensiones (Clases gaussianas)
    # Generamos muestras desde las distribuciones gaussianas reales para probar
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    mu2 = np.array([4.0, 4.0])
    sigma2 = np.array([[1.0, -0.3], [-0.3, 1.0]])
    
    mu3 = np.array([-4.0, 3.0])
    sigma3 = np.array([[0.5, 0.0], [0.0, 2.0]])
    
    n_por_clase = 15
    X_c1 = rng.multivariate_normal(mu1, sigma1, size=n_por_clase)
    X_c2 = rng.multivariate_normal(mu2, sigma2, size=n_por_clase)
    X_c3 = rng.multivariate_normal(mu3, sigma3, size=n_por_clase)
    
    X_p2 = np.vstack([X_c1, X_c2, X_c3])
    etiquetas_reales = np.array([1]*n_por_clase + [2]*n_por_clase + [3]*n_por_clase)
    
    df_p2 = pd.DataFrame({
        "x_1": X_p2[:, 0],
        "x_2": X_p2[:, 1],
        "clase_origen": etiquetas_reales
    })
    df_p2.to_csv(dir_path / "punto2_muestras.csv", index=False)

    # 3. Datasets para Punto 3: Referencias y Consultas
    # Conjunto de referencia (base de conocimiento): 30 muestras en R^2
    X_ref = rng.uniform(low=-10.0, high=10.0, size=(30, 2))
    df_ref = pd.DataFrame(X_ref, columns=["x_1", "x_2"])
    df_ref.insert(0, "id_muestra", [f"REF_{i+1:02d}" for i in range(30)])
    df_ref.to_csv(dir_path / "punto3_referencia.csv", index=False)

    # Conjunto de consultas (queries): 5 muestras en R^2
    X_query = rng.uniform(low=-8.0, high=8.0, size=(5, 2))
    df_query = pd.DataFrame(X_query, columns=["x_1", "x_2"])
    df_query.insert(0, "id_consulta", [f"QRY_{i+1:02d}" for i in range(5)])
    df_query.to_csv(dir_path / "punto3_consultas.csv", index=False)


if __name__ == "__main__":
    print("Generando archivos CSV de prueba para el taller...")
    generar_datasets_taller()
    print("Archivos CSV generados exitosamente en la carpeta 'data/'.")
