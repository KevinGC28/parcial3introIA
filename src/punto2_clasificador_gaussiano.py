"""
=============================================================================
PUNTO 2: CLASIFICADOR BAYESIANO GAUSSIANO MULTIVARIADO (QDA)
=============================================================================
Asignatura: Introducción a la Inteligencia Artificial
Autor: Solución pedagógica para Kevin
Totalmente parametrizable: admite K clases arbitrarias (K >= 2), dimensiones
d arbitrarias, priors personalizables, y generación dinámica de fronteras
y regiones de decisión sin supuestos quemados.
=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ClaseGaussiana:
    """
    Representa una clase C_k modelada mediante una distribución Gaussiana Multivariada N(mu_k, Sigma_k).
    Completamente parametrizable en cualquier dimensión d >= 1.
    """

    def __init__(
        self,
        etiqueta: Any,
        mu: Union[List[float], np.ndarray],
        sigma: Union[List[List[float]], np.ndarray],
        prior: float = 1.0,
        nombre: Optional[str] = None
    ) -> None:
        self.etiqueta = etiqueta
        self.mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.prior = float(prior)
        self.d = len(self.mu)
        self.nombre = nombre if nombre is not None else f"Clase {etiqueta}"

        # Validaciones dimensionales y matemáticas
        if self.sigma.shape != (self.d, self.d):
            raise ValueError(
                f"La matriz de covarianza de '{self.nombre}' debe tener forma "
                f"({self.d}, {self.d}), pero se recibió {self.sigma.shape}"
            )

        # Determinante |Sigma|
        self.det_sigma = float(np.linalg.det(self.sigma))
        if self.det_sigma <= 0:
            raise ValueError(
                f"La matriz de covarianza de '{self.nombre}' no es definida positiva (det={self.det_sigma:.4e} <= 0)."
            )

        # Inversa Sigma^(-1) y logaritmo del determinante
        self.inv_sigma = np.linalg.inv(self.sigma)
        self.log_det_sigma = float(np.log(self.det_sigma))
        
        # Constante de normalización
        self.constante_norm = 1.0 / (((2.0 * np.pi) ** (self.d / 2.0)) * np.sqrt(self.det_sigma))

    def distancia_mahalanobis_cuadrada(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia de Mahalanobis al cuadrado:
            D_M^2(x) = (x - mu)^T * Sigma^(-1) * (x - mu)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            diff = X - self.mu
            return np.array([float(np.dot(diff, np.dot(self.inv_sigma, diff)))])
        else:
            diff = X - self.mu  # (N, d)
            return np.sum(np.dot(diff, self.inv_sigma) * diff, axis=1)

    def evaluar_densidad_probabilidad(self, X: np.ndarray) -> np.ndarray:
        """Evalúa la función de densidad de probabilidad continua p(x | C_k)."""
        d_m2 = self.distancia_mahalanobis_cuadrada(X)
        return self.constante_norm * np.exp(-0.5 * d_m2)

    def evaluar_funcion_discriminante(self, X: np.ndarray) -> np.ndarray:
        """
        Función discriminante logarítmica exacta (Análisis Discriminante Cuadrático):
            g_k(x) = ln p(x | C_k) + ln P(C_k)
        """
        d_m2 = self.distancia_mahalanobis_cuadrada(X)
        termino_pi = -0.5 * self.d * np.log(2.0 * np.pi)
        termino_cov = -0.5 * self.log_det_sigma
        termino_prior = np.log(self.prior) if self.prior > 0 else -1e12
        return -0.5 * d_m2 + termino_cov + termino_pi + termino_prior


class ClasificadorBayesianoGaussiano:
    """
    Clasificador Bayesiano Multiclase para K distribuciones Gaussianas Multivariadas.
    Totalmente dinámico: acepta cualquier número de clases K >= 2 y dimensión d.
    """

    def __init__(self, clases: Optional[List[ClaseGaussiana]] = None) -> None:
        """
        Inicializa el clasificador con una lista arbitraria de clases gaussianas.
        Si no se especifican clases, carga las 3 clases canónicas del taller.
        """
        if clases is None:
            self.clases = [
                ClaseGaussiana(
                    etiqueta=1,
                    nombre="Clase 1",
                    mu=[0.0, 0.0],
                    sigma=[[1.0, 0.5], [0.5, 1.0]],
                    prior=1.0 / 3.0
                ),
                ClaseGaussiana(
                    etiqueta=2,
                    nombre="Clase 2",
                    mu=[4.0, 4.0],
                    sigma=[[1.0, -0.3], [-0.3, 1.0]],
                    prior=1.0 / 3.0
                ),
                ClaseGaussiana(
                    etiqueta=3,
                    nombre="Clase 3",
                    mu=[-4.0, 3.0],
                    sigma=[[0.5, 0.0], [0.0, 2.0]],
                    prior=1.0 / 3.0
                )
            ]
        else:
            if len(clases) < 2:
                raise ValueError("Se requieren al menos 2 clases para realizar clasificación.")
            self.clases = list(clases)

        self._normalizar_priors()
        self.k_clases = len(self.clases)
        self.dimension = self.clases[0].d
        self.etiquetas = [c.etiqueta for c in self.clases]

        # Validar consistencia dimensional entre todas las clases
        for c in self.clases:
            if c.d != self.dimension:
                raise ValueError(
                    f"Inconsistencia dimensional: la clase '{c.nombre}' tiene dimensión {c.d}, "
                    f"pero la primera clase tiene dimensión {self.dimension}."
                )

    def _normalizar_priors(self) -> None:
        """Garantiza que la suma de las probabilidades a priori sea 1.0."""
        suma_priors = sum(c.prior for c in self.clases)
        if suma_priors > 0:
            for c in self.clases:
                c.prior /= suma_priors

    def agregar_clase(self, nueva_clase: ClaseGaussiana) -> None:
        """Permite parametrizar e inyectar nuevas clases dinámicamente."""
        if nueva_clase.d != self.dimension:
            raise ValueError(
                f"La nueva clase '{nueva_clase.nombre}' tiene dimensión {nueva_clase.d}, "
                f"pero el modelo requiere dimensión {self.dimension}."
            )
        self.clases.append(nueva_clase)
        self._normalizar_priors()
        self.k_clases = len(self.clases)
        self.etiquetas = [c.etiqueta for c in self.clases]

    def calcular_funciones_discriminantes(self, X: np.ndarray) -> np.ndarray:
        """Calcula la matriz g_k(x) de forma (N, K)."""
        X = np.asarray(X, dtype=np.float64)
        scores = [clase.evaluar_funcion_discriminante(X) for clase in self.clases]
        return np.column_stack(scores)

    def calcular_densidades(self, X: np.ndarray) -> np.ndarray:
        """Calcula la matriz de verosimilitudes p(x | C_k) de forma (N, K)."""
        X = np.asarray(X, dtype=np.float64)
        densidades = [clase.evaluar_densidad_probabilidad(X) for clase in self.clases]
        return np.column_stack(densidades)

    def calcular_distancias_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Calcula la matriz de distancias de Mahalanobis D_M,k(x) de forma (N, K)."""
        X = np.asarray(X, dtype=np.float64)
        distancias = [np.sqrt(clase.distancia_mahalanobis_cuadrada(X)) for clase in self.clases]
        return np.column_stack(distancias)

    def predecir_probabilidades_posteriori(self, X: np.ndarray) -> np.ndarray:
        """Calcula las probabilidades posteriores P(C_k | x) numéricamente estables."""
        G = self.calcular_funciones_discriminantes(X)
        G_max = np.max(G, axis=1, keepdims=True)
        exp_G = np.exp(G - G_max)
        return exp_G / np.sum(exp_G, axis=1, keepdims=True)

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Asigna cada muestra a la clase con mayor función discriminante k* = argmax_k g_k(x)."""
        G = self.calcular_funciones_discriminantes(X)
        indices_max = np.argmax(G, axis=1)
        return np.array([self.etiquetas[idx] for idx in indices_max])

    def clasificar_dataframe(
        self,
        df: pd.DataFrame,
        columnas_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Clasifica las muestras contenidas en un DataFrame de forma totalmente dinámica
        para cualquier número de clases K y nombres de columnas.
        """
        if columnas_features is None:
            cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(cols_num) < self.dimension:
                raise ValueError(
                    f"El DataFrame tiene {len(cols_num)} columnas numéricas, "
                    f"pero el clasificador requiere d={self.dimension} variables."
                )
            columnas_features = cols_num[:self.dimension]

        X = df[columnas_features].to_numpy(dtype=np.float64)
        G = self.calcular_funciones_discriminantes(X)
        D_M = self.calcular_distancias_mahalanobis(X)
        post = self.predecir_probabilidades_posteriori(X)
        y_pred = self.predecir(X)

        df_res = df.copy()
        for idx, clase in enumerate(self.clases):
            df_res[f"D_Mahalanobis_{clase.nombre}"] = D_M[:, idx]
            df_res[f"g_{clase.etiqueta}(x)"] = G[:, idx]
            df_res[f"P({clase.nombre}|x)"] = post[:, idx]

        df_res["clase_predicha"] = y_pred
        return df_res

    def graficar_regiones_decision(
        self,
        X_muestras: Optional[np.ndarray] = None,
        y_muestras: Optional[np.ndarray] = None,
        indices_features: Tuple[int, int] = (0, 1),
        rango_x: Optional[Tuple[float, float]] = None,
        rango_y: Optional[Tuple[float, float]] = None,
        resolucion: int = 350,
        ruta_salida: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Genera una visualización dinámica y adaptable a K CLASES CUALESQUIERA:
        - Malla densa con regiones coloreadas según la clase ganadora.
        - Trazado automático de curvas de nivel elípticas por cada clase gaussiana.
        - Fronteras de decisión exactas entre todos los pares de clases adyacentes.
        - Adaptación automática de límites coordenados y paleta de colores.
        """
        if self.dimension < 2:
            raise ValueError("La visualización en 2D requiere al menos d=2 características.")

        sns.set_theme(style="white", palette="deep")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        # 1. Determinar rangos automáticamente si no se especifican
        idx1, idx2 = indices_features
        if rango_x is None or rango_y is None:
            todos_x = [c.mu[idx1] for c in self.clases]
            todos_y = [c.mu[idx2] for c in self.clases]
            if X_muestras is not None:
                todos_x.extend(X_muestras[:, idx1].tolist())
                todos_y.extend(X_muestras[:, idx2].tolist())
            
            # Margen generoso de 3.5 sigmas
            max_std_x = max(np.sqrt(c.sigma[idx1, idx1]) for c in self.clases)
            max_std_y = max(np.sqrt(c.sigma[idx2, idx2]) for c in self.clases)

            rx = (min(todos_x) - 2.5 * max_std_x, max(todos_x) + 2.5 * max_std_x)
            ry = (min(todos_y) - 2.5 * max_std_y, max(todos_y) + 2.5 * max_std_y)
        else:
            rx, ry = rango_x, rango_y

        x_vals = np.linspace(rx[0], rx[1], resolucion)
        y_vals = np.linspace(ry[0], ry[1], resolucion)
        XX, YY = np.meshgrid(x_vals, y_vals)
        grid_2d = np.c_[XX.ravel(), YY.ravel()]

        # Si d > 2, rellenar dimensiones restantes con sus valores promedio
        if self.dimension > 2:
            grid_completo = np.zeros((len(grid_2d), self.dimension), dtype=np.float64)
            grid_completo[:, idx1] = grid_2d[:, 0]
            grid_completo[:, idx2] = grid_2d[:, 1]
            for dim in range(self.dimension):
                if dim not in (idx1, idx2):
                    prom_dim = np.mean([c.mu[dim] for c in self.clases])
                    grid_completo[:, dim] = prom_dim
        else:
            grid_completo = grid_2d

        # 2. Evaluar funciones discriminantes en toda la malla
        G_grid = self.calcular_funciones_discriminantes(grid_completo)
        indices_ganadores = np.argmax(G_grid, axis=1)  # 0, ..., K-1
        ZZ = indices_ganadores.reshape(XX.shape)

        # 3. Paleta de colores dinámica para K clases
        colores_base = plt.get_cmap("tab10" if self.k_clases <= 10 else "gist_rainbow")
        colores_lista = [colores_base(i / max(self.k_clases - 1, 1)) for i in range(self.k_clases)]

        # Regiones de decisión
        cmap_pastel = plt.matplotlib.colors.ListedColormap([
            plt.matplotlib.colors.to_rgba(c, alpha=0.25) for c in colores_lista
        ])
        ax.contourf(XX, YY, ZZ, levels=np.arange(-0.5, self.k_clases + 0.5, 1.0), cmap=cmap_pastel, zorder=1)

        # 4. Trazar fronteras de decisión cuadráticas entre pares de clases
        for i in range(self.k_clases):
            for j in range(i + 1, self.k_clases):
                diff_g = (G_grid[:, i] - G_grid[:, j]).reshape(XX.shape)
                ax.contour(XX, YY, diff_g, levels=[0], colors="#2c3e50", linewidths=1.8, linestyles="--", zorder=4)

        # 5. Dibujar curvas de nivel de densidad (elipses) y centros de cada clase
        for idx, clase in enumerate(self.clases):
            color = colores_lista[idx]
            mu_x, mu_y = clase.mu[idx1], clase.mu[idx2]

            # Centro (media)
            ax.scatter(
                mu_x, mu_y,
                s=240, color=color, marker="*", edgecolor="black", linewidth=1.5,
                label=f"Media {clase.nombre} ({mu_x:.1f}, {mu_y:.1f})", zorder=6
            )

            # Densidad en la malla
            dens = clase.evaluar_densidad_probabilidad(grid_completo).reshape(XX.shape)
            # Niveles a 1, 2 y 2.5 sigmas
            niveles_dens = [clase.constante_norm * np.exp(-0.5 * (k ** 2)) for k in [1.0, 2.0, 2.8]]
            ax.contour(XX, YY, dens, levels=sorted(niveles_dens), colors=[color], alpha=0.7, linewidths=1.2, zorder=3)

        # 6. Superponer muestras clasificadas si se proporcionan
        if X_muestras is not None and y_muestras is not None:
            for idx, clase in enumerate(self.clases):
                mascara = (y_muestras == clase.etiqueta)
                if np.any(mascara):
                    ax.scatter(
                        X_muestras[mascara, idx1], X_muestras[mascara, idx2],
                        s=60, color=colores_lista[idx], edgecolor="black", alpha=0.9,
                        label=f"Muestras {clase.nombre}", zorder=5
                    )

        ax.set_xlim(rx)
        ax.set_ylim(ry)
        ax.set_title(
            f"Clasificador Bayesiano Gaussiano Multivariado (QDA - {self.k_clases} Clases)\n"
            f"Superficies de Decisión y Curvas de Nivel de Densidad",
            fontsize=13, fontweight="bold", pad=15
        )
        ax.set_xlabel(f"Característica $x_{{{idx1+1}}}$", fontsize=11)
        ax.set_ylabel(f"Característica $x_{{{idx2+1}}}$", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

        plt.tight_layout()

        if ruta_salida:
            Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
            print(f"-> Gráfica de regiones de Bayes guardada en: {ruta_salida}")

        return fig


def main():
    """Ejecución y demostración parametrizable del Punto 2."""
    parser = argparse.ArgumentParser(description="Punto 2: Clasificador Bayesiano Gaussiano Parametrizable")
    parser.add_argument("--config", type=str, default=None, help="Ruta a archivo de configuración JSON")
    parser.add_argument("--csv", type=str, default="data/punto2_muestras.csv", help="Archivo CSV con muestras")
    parser.add_argument("--salida_img", type=str, default="outputs/punto2_bayes_regiones.png", help="Ruta de la imagen")
    parser.add_argument("--salida_csv", type=str, default="outputs/punto2_resultados.csv", help="Ruta del CSV de resultados")
    args = parser.parse_args()

    clases = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        p2_cfg = cfg.get("punto2", {})
        clases_raw = p2_cfg.get("clases", [])
        if clases_raw:
            clases = [
                ClaseGaussiana(
                    etiqueta=c["etiqueta"],
                    nombre=c.get("nombre", f"Clase {c['etiqueta']}"),
                    mu=c["mu"],
                    sigma=c["sigma"],
                    prior=c.get("prior", 1.0)
                ) for c in clases_raw
            ]

    clasificador = ClasificadorBayesianoGaussiano(clases=clases)

    print("=" * 75)
    print(f"EJECUCIÓN DEL PUNTO 2: CLASIFICADOR BAYESIANO GAUSSIANO ({clasificador.k_clases} CLASES)")
    print("=" * 75)

    for c in clasificador.clases:
        print(f"\n--- {c.nombre.upper()} (Etiqueta: {c.etiqueta}, Prior: {c.prior:.3f}) ---")
        print(f"  Media mu: {c.mu.tolist()}")
        print(f"  Matriz Covarianza Sigma:\n{c.sigma}")
        print(f"  Determinante |Sigma|: {c.det_sigma:.4f}")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        from utils_csv import generar_datasets_taller
        generar_datasets_taller()

    df = pd.read_csv(csv_path)
    df_resultado = clasificador.clasificar_dataframe(df)

    Path(args.salida_csv).parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(args.salida_csv, index=False)
    print(f"\nResultados exportados a: {args.salida_csv}")

    cols_feat = [c for c in df.columns if c.startswith("x_")]
    if len(cols_feat) < clasificador.dimension:
        cols_feat = df.select_dtypes(include=[np.number]).columns.tolist()[:clasificador.dimension]

    X_mat = df[cols_feat].to_numpy()
    clasificador.graficar_regiones_decision(
        X_muestras=X_mat,
        y_muestras=df_resultado["clase_predicha"].to_numpy(),
        ruta_salida=args.salida_img
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
