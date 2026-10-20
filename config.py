"""
=============================================================================
MÓDULO DE CONFIGURACIÓN Y PARAMETRIZACIÓN GLOBAL
=============================================================================
Autor: Creado para Kevin - Introducción a la Inteligencia Artificial
Propósito: Centralizar TODOS los parámetros del taller y de análisis futuros
           en estructuras tipadas, desacopladas y serializables (JSON/Dict).
           ¡Cero datos quemados en la lógica de cálculo!
=============================================================================
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np


@dataclass
class ConfigPunto1:
    """Parámetros configurables para el Clasificador Lineal f(x) = <w, x> + b."""
    # Vector de pesos w (puede tener cualquier dimensión d >= 1)
    w: List[float] = field(default_factory=lambda: [1.0 / 3.0, 1.0, -3.0, -2.0, 5.0 / 4.0])
    # Sesgo escalar b
    b: float = 2.0 / 3.0
    # Etiquetas de clase para la partición binaria
    etiqueta_positiva: int = 1
    etiqueta_negativa: int = -1
    # Umbral de decisión (por defecto 0.0)
    umbral_decision: float = 0.0
    # Nombres de las características (None = autodetectar o usar x_1, ..., x_d)
    columnas_features: Optional[List[str]] = None
    # Índices o nombres de variables para proyectar en gráfica 2D (None = autodetectar mayores pesos)
    features_grafica_2d: Optional[List[int]] = None


@dataclass
class ConfigClaseGaussiana:
    """Parámetros configurables para una clase gaussiana C_k."""
    etiqueta: int
    nombre: str
    mu: List[float]
    sigma: List[List[float]]
    prior: float = 1.0 / 3.0


@dataclass
class ConfigPunto2:
    """Parámetros configurables para el Clasificador Bayesiano Gaussiano (QDA)."""
    # Lista arbitraria de clases (soporta K=2, 3, 4, 5... clases en cualquier dimensión d)
    clases: List[ConfigClaseGaussiana] = field(default_factory=lambda: [
        ConfigClaseGaussiana(
            etiqueta=1,
            nombre="Clase 1",
            mu=[0.0, 0.0],
            sigma=[[1.0, 0.5], [0.5, 1.0]],
            prior=1.0 / 3.0
        ),
        ConfigClaseGaussiana(
            etiqueta=2,
            nombre="Clase 2",
            mu=[4.0, 4.0],
            sigma=[[1.0, -0.3], [-0.3, 1.0]],
            prior=1.0 / 3.0
        ),
        ConfigClaseGaussiana(
            etiqueta=3,
            nombre="Clase 3",
            mu=[-4.0, 3.0],
            sigma=[[0.5, 0.0], [0.0, 2.0]],
            prior=1.0 / 3.0
        )
    ])
    # Columnas de características a utilizar (None = autodetectar numéricas)
    columnas_features: Optional[List[str]] = None
    # Resolución de la malla para graficar regiones de decisión
    resolucion_grid: int = 350
    # Rango de visualización (None = autocalcular con base en mu y covarianzas)
    rango_x: Optional[List[float]] = None
    rango_y: Optional[List[float]] = None


@dataclass
class ConfigPunto3:
    """Parámetros configurables para la Búsqueda de Muestras Más Cercanas."""
    # Número de vecinos más cercanos a encontrar
    k_vecinos: int = 2
    # Métrica de distancia ('euclidea', etc.)
    metrica: str = "euclidea"
    # Columnas a considerar (None = autodetectar)
    columnas_features: Optional[List[str]] = None
    # Columna identificadora en la referencia y consulta
    columna_id_ref: Optional[str] = "id_muestra"
    columna_id_query: Optional[str] = "id_consulta"


@dataclass
class ConfigRutas:
    """Rutas de archivos y directorios de entrada/salida."""
    data_dir: str = "data"
    output_dir: str = "outputs"
    
    # Archivos CSV de entrada
    csv_punto1: str = "data/punto1_muestras.csv"
    csv_punto2: str = "data/punto2_muestras.csv"
    csv_punto3_ref: str = "data/punto3_referencia.csv"
    csv_punto3_query: str = "data/punto3_consultas.csv"
    
    # Archivos de salida (imágenes y reportes CSV)
    salida_p1_img: str = "outputs/punto1_clasificacion.png"
    salida_p1_csv: str = "outputs/punto1_resultados.csv"
    
    salida_p2_img: str = "outputs/punto2_bayes_regiones.png"
    salida_p2_csv: str = "outputs/punto2_resultados.csv"
    
    salida_p3_img: str = "outputs/punto3_vecinos_cercanos.png"
    salida_p3_csv: str = "outputs/punto3_resultados.csv"


@dataclass
class ConfiguracionTaller:
    """Configuración maestra que agrupa todos los parámetros del proyecto."""
    punto1: ConfigPunto1 = field(default_factory=ConfigPunto1)
    punto2: ConfigPunto2 = field(default_factory=ConfigPunto2)
    punto3: ConfigPunto3 = field(default_factory=ConfigPunto3)
    rutas: ConfigRutas = field(default_factory=ConfigRutas)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a un diccionario nativo de Python."""
        return asdict(self)

    def to_json(self, filepath: Union[str, Path], indent: int = 4) -> None:
        """Guarda la configuración actual en un archivo JSON."""
        ruta = Path(filepath)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfiguracionTaller":
        """Construye una instancia de configuración desde un diccionario."""
        p1_data = data.get("punto1", {})
        p2_data = data.get("punto2", {})
        p3_data = data.get("punto3", {})
        rutas_data = data.get("rutas", {})

        # Reconstruir clases de Punto 2
        clases_p2 = []
        for c in p2_data.get("clases", []):
            clases_p2.append(ConfigClaseGaussiana(**c))
        
        p2_obj = ConfigPunto2(
            clases=clases_p2 if clases_p2 else ConfigPunto2().clases,
            columnas_features=p2_data.get("columnas_features"),
            resolucion_grid=p2_data.get("resolucion_grid", 350),
            rango_x=p2_data.get("rango_x"),
            rango_y=p2_data.get("rango_y")
        )

        return cls(
            punto1=ConfigPunto1(**p1_data) if p1_data else ConfigPunto1(),
            punto2=p2_obj,
            punto3=ConfigPunto3(**p3_data) if p3_data else ConfigPunto3(),
            rutas=ConfigRutas(**rutas_data) if rutas_data else ConfigRutas()
        )

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "ConfiguracionTaller":
        """Carga y valida la configuración desde un archivo JSON."""
        ruta = Path(filepath)
        if not ruta.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de configuración: {filepath}")
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def obtener_configuracion_defecto() -> ConfiguracionTaller:
    """Devuelve la configuración predeterminada con los parámetros solicitados en el taller."""
    return ConfiguracionTaller()
