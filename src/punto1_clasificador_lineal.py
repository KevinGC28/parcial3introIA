"""
=============================================================================
PUNTO 1: CLASIFICADOR LINEAL BASADO EN HIPERPLANO DE DECISIÓN
=============================================================================
Asignatura: Introducción a la Inteligencia Artificial
Autor: Solución pedagógica para Kevin
Totalmente parametrizable: admite cualquier dimensión d >= 1, vector de
pesos w arbitrario, sesgo b (beta) arbitrario, umbrales y etiquetas.
=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import Union, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ClasificadorLineal:
    """
    Clasificador lineal binario basado en una superficie de decisión (hiperplano).
    
    Fundamento Matemático Parametrizable:
    -------------------------------------
    Para cualquier espacio d-dimensional (d >= 1):
    - w in R^d es el vector de pesos (vector normal al hiperplano).
    - b in R es el sesgo (bias / intercepto / beta).
    - <w, x> = w^T * x = sum_{i=1}^d w_i * x_i.
    - La función discriminante es f(x) = <w, x> + b.
    - Regla de decisión:
          f(x) >= umbral  --> Asignar a etiqueta_positiva (+1 por defecto)
          f(x) <  umbral  --> Asignar a etiqueta_negativa (-1 por defecto)
    """

    def __init__(
        self,
        w: Optional[Union[List[float], np.ndarray]] = None,
        b: Optional[float] = None,
        etiqueta_positiva: int = 1,
        etiqueta_negativa: int = -1,
        umbral_decision: float = 0.0
    ) -> None:
        """
        Inicializa el clasificador lineal con parámetros configurables.
        
        Si no se especifican w y b, se utilizan los valores canónicos del taller:
            w = [1/3, 1, -3, -2, 5/4]
            b = 2/3
        """
        if w is None:
            self.w = np.array([1.0 / 3.0, 1.0, -3.0, -2.0, 5.0 / 4.0], dtype=np.float64)
        else:
            self.w = np.asarray(w, dtype=np.float64).flatten()

        if b is None:
            self.b = float(2.0 / 3.0)
        else:
            self.b = float(b)

        self.dimension = len(self.w)
        if self.dimension < 1:
            raise ValueError("El vector de pesos w debe tener al menos una dimensión (d >= 1).")

        self.etiqueta_positiva = etiqueta_positiva
        self.etiqueta_negativa = etiqueta_negativa
        self.umbral_decision = float(umbral_decision)
        
        # Norma euclídea del vector de pesos: ||w||_2
        self.norma_w = float(np.linalg.norm(self.w))
        if self.norma_w == 0.0:
            raise ValueError("El vector de pesos w no puede ser el vector nulo (||w|| = 0).")

    def producto_interno(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula el producto interno <w, x> para una muestra o lote de muestras.
        Vectorizado para N muestras en O(N * d).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            if X.shape[0] != self.dimension:
                raise ValueError(
                    f"Dimensión inconsistente: el modelo espera d={self.dimension} variables, "
                    f"pero la muestra tiene {X.shape[0]}."
                )
            return np.array([np.dot(self.w, X)])
        elif X.ndim == 2:
            if X.shape[1] != self.dimension:
                raise ValueError(
                    f"Dimensión inconsistente: el modelo espera d={self.dimension} columnas, "
                    f"pero la matriz de datos tiene {X.shape[1]}."
                )
            return np.dot(X, self.w)
        else:
            raise ValueError("El arreglo de entrada debe ser 1D o 2D.")

    def evaluar_funcion_discriminante(self, X: np.ndarray) -> np.ndarray:
        """Evalúa f(x) = <w, x> + b para cada muestra."""
        return self.producto_interno(X) + self.b

    def distancia_con_signo_al_hiperplano(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia ortogonal con signo desde cada punto x al hiperplano:
            r(x) = (f(x) - umbral) / ||w||_2
        """
        return (self.evaluar_funcion_discriminante(X) - self.umbral_decision) / self.norma_w

    def predecir_probabilidad_logistica(self, X: np.ndarray) -> np.ndarray:
        """Calcula la probabilidad logística sigmoide P(Y = +1 | x)."""
        f_x = self.evaluar_funcion_discriminante(X) - self.umbral_decision
        f_x_clip = np.clip(f_x, -500, 500)
        return 1.0 / (1.0 + np.exp(-f_x_clip))

    def predecir(self, X: np.ndarray, umbral: Optional[float] = None) -> np.ndarray:
        """
        Aplica la regla de decisión parametrizable:
            y_pred = etiqueta_positiva si f(x) >= umbral
            y_pred = etiqueta_negativa si f(x) <  umbral
        """
        umbral_actual = self.umbral_decision if umbral is None else float(umbral)
        f_x = self.evaluar_funcion_discriminante(X)
        return np.where(f_x >= umbral_actual, self.etiqueta_positiva, self.etiqueta_negativa)

    def clasificar_dataframe(
        self,
        df: pd.DataFrame,
        columnas_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Clasifica las muestras contenidas en un DataFrame de forma totalmente dinámica.
        """
        if columnas_features is None:
            cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(cols_num) < self.dimension:
                raise ValueError(
                    f"El DataFrame contiene {len(cols_num)} columnas numéricas, "
                    f"pero el clasificador requiere d={self.dimension} características."
                )
            columnas_features = cols_num[:self.dimension]
        else:
            if len(columnas_features) != self.dimension:
                raise ValueError(
                    f"Se especificaron {len(columnas_features)} columnas, pero el modelo "
                    f"está configurado para d={self.dimension} variables de peso w."
                )

        X = df[columnas_features].to_numpy(dtype=np.float64)
        
        dot_prod = self.producto_interno(X)
        f_x = self.evaluar_funcion_discriminante(X)
        distancias = self.distancia_con_signo_al_hiperplano(X)
        probabilidades = self.predecir_probabilidad_logistica(X)
        predicciones = self.predecir(X)

        df_resultado = df.copy()
        df_resultado["<w, x>"] = dot_prod
        df_resultado["f(x)"] = f_x
        df_resultado["distancia_hiperplano"] = distancias
        df_resultado["P(Clase_Positiva|x)"] = probabilidades
        df_resultado["clase_predicha"] = predicciones
        
        return df_resultado

    def graficar_clasificacion(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        f_x: np.ndarray,
        indices_features: Optional[Tuple[int, int]] = None,
        nombres_features: Optional[List[str]] = None,
        ruta_salida: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Genera una visualización dinámica y adaptable a CUALQUIER dimensión d >= 1:
        - Panel 1: Espectro de puntuaciones f(x) con frontera de decisión y regiones sombreadas.
        - Panel 2: Proyección en 2D sobre 2 características seleccionadas o auto-detectadas
                   por mayor magnitud de pesos |w_i|, calculando la traza analítica del hiperplano.
        """
        sns.set_theme(style="whitegrid", palette="deep")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # ---------------- Panel 1: Espectro de Scores f(x) ----------------
        indices = np.arange(len(f_x))
        color_pos = "#2ecc71"
        color_neg = "#e74c3c"
        colores = np.where(y_pred == self.etiqueta_positiva, color_pos, color_neg)

        axes[0].axhline(
            self.umbral_decision, color="black", linestyle="--", linewidth=1.8,
            label=f"Frontera: $f(x)={self.umbral_decision:.2f}$"
        )
        axes[0].scatter(indices, f_x, c=colores, s=80, edgecolors="k", alpha=0.85, zorder=3)
        axes[0].vlines(indices, self.umbral_decision, f_x, colors=colores, alpha=0.4, linewidth=1.5)

        y_min = float(np.min(f_x)) - 1.0
        y_max = float(np.max(f_x)) + 1.0
        axes[0].axhspan(
            self.umbral_decision, max(y_max, self.umbral_decision + 1),
            facecolor=color_pos, alpha=0.08,
            label=f"Región Clase {self.etiqueta_positiva:+d} ($f(x) \\geq {self.umbral_decision:.2f}$)"
        )
        axes[0].axhspan(
            min(y_min, self.umbral_decision - 1), self.umbral_decision,
            facecolor=color_neg, alpha=0.08,
            label=f"Región Clase {self.etiqueta_negativa:+d} ($f(x) < {self.umbral_decision:.2f}$)"
        )

        axes[0].set_title(
            f"Puntuación Discriminante $f(x) = \\langle w, x \\rangle + {self.b:.2f}$",
            fontsize=12, fontweight="bold"
        )
        axes[0].set_xlabel("Índice de la Muestra", fontsize=11)
        axes[0].set_ylabel("Valor de $f(x)$", fontsize=11)
        axes[0].legend(loc="upper left", frameon=True)
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # ---------------- Panel 2: Proyección 2D Dinámica ----------------
        if self.dimension == 1:
            # Caso 1D: dispersión a lo largo del eje x_1
            x1 = X[:, 0]
            axes[1].scatter(x1, np.zeros_like(x1), c=colores, s=90, edgecolors="black", zorder=3)
            # Punto de corte: w1 * x1 + b = umbral => x1 = (umbral - b) / w1
            x_corte = (self.umbral_decision - self.b) / self.w[0]
            axes[1].axvline(x_corte, color="#8e44ad", linestyle="--", linewidth=2.5, label=f"Corte $x_1={x_corte:.2f}$")
            axes[1].set_xlabel(nombres_features[0] if nombres_features else "Característica $x_1$", fontsize=11)
            axes[1].set_title("Espacio 1D de Decisión", fontsize=12, fontweight="bold")
            axes[1].legend(loc="upper right")
        else:
            # Caso d >= 2: Seleccionar las 2 características
            if indices_features is not None:
                idx1, idx2 = indices_features
            else:
                # Seleccionar automáticamente las 2 características con mayor magnitud de peso |w|
                indices_ordenados = np.argsort(np.abs(self.w))
                idx1, idx2 = int(indices_ordenados[-2]), int(indices_ordenados[-1])

            x_feat1 = X[:, idx1]
            x_feat2 = X[:, idx2]

            nom1 = nombres_features[idx1] if nombres_features and idx1 < len(nombres_features) else f"$x_{{{idx1+1}}}$"
            nom2 = nombres_features[idx2] if nombres_features and idx2 < len(nombres_features) else f"$x_{{{idx2+1}}}$"

            axes[1].scatter(
                x_feat1, x_feat2,
                c=y_pred, cmap="coolwarm", s=90, edgecolors="black", alpha=0.9, zorder=3
            )

            # Calcular el efecto promedio del resto de las dimensiones en el corte
            indices_resto = [i for i in range(self.dimension) if i not in (idx1, idx2)]
            if len(indices_resto) > 0:
                x_resto_prom = np.mean(X[:, indices_resto], axis=0)
                c_resto = float(np.sum(self.w[indices_resto] * x_resto_prom)) + self.b - self.umbral_decision
            else:
                c_resto = self.b - self.umbral_decision

            # Trazar recta w_idx1 * x1 + w_idx2 * x2 + c_resto = 0
            w1, w2 = self.w[idx1], self.w[idx2]
            grid_x1 = np.linspace(np.min(x_feat1) - 0.5, np.max(x_feat1) + 0.5, 200)

            if np.abs(w2) > 1e-9:
                grid_x2 = -(w1 * grid_x1 + c_resto) / w2
                axes[1].plot(grid_x1, grid_x2, color="#8e44ad", linewidth=2.5, linestyle="-",
                             label=f"Traza de frontera ({nom1} vs {nom2})")
            elif np.abs(w1) > 1e-9:
                x_corte = -c_resto / w1
                axes[1].axvline(x_corte, color="#8e44ad", linewidth=2.5, linestyle="-",
                                label=f"Corte vertical {nom1}={x_corte:.2f}")

            axes[1].set_xlim(np.min(x_feat1) - 0.5, np.max(x_feat1) + 0.5)
            axes[1].set_ylim(np.min(x_feat2) - 0.5, np.max(x_feat2) + 0.5)
            axes[1].set_title(
                f"Espacio de Características Proyectado ({nom1} vs {nom2})",
                fontsize=12, fontweight="bold"
            )
            axes[1].set_xlabel(f"{nom1} (peso $w={w1:.3f}$)", fontsize=11)
            axes[1].set_ylabel(f"{nom2} (peso $w={w2:.3f}$)", fontsize=11)
            axes[1].legend(loc="upper right", frameon=True)
            axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.suptitle(
            f"Clasificador Lineal Parametrizable ($d={self.dimension}$ dimensiones)",
            fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        if ruta_salida:
            Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
            print(f"-> Gráfica guardada exitosamente en: {ruta_salida}")

        return fig


def main():
    """Ejecución de prueba y demostración dinámica del Punto 1."""
    parser = argparse.ArgumentParser(description="Punto 1: Clasificador Lineal Totalmente Parametrizable")
    parser.add_argument("--config", type=str, default=None, help="Ruta a archivo de configuración JSON")
    parser.add_argument("--w", type=str, default=None, help="Vector de pesos w separado por comas (ej: '0.5,-1.2,3.0')")
    parser.add_argument("--b", type=float, default=None, help="Sesgo b (beta)")
    parser.add_argument("--csv", type=str, default="data/punto1_muestras.csv", help="Ruta al archivo CSV de muestras")
    parser.add_argument("--salida_img", type=str, default="outputs/punto1_clasificacion.png", help="Ruta de la gráfica")
    parser.add_argument("--salida_csv", type=str, default="outputs/punto1_resultados.csv", help="Ruta del reporte CSV")
    args = parser.parse_args()

    w_vals = None
    b_val = args.b

    # Cargar desde config.json si se proporciona
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        p1_cfg = cfg.get("punto1", {})
        w_vals = p1_cfg.get("w")
        b_val = p1_cfg.get("b", b_val)
    elif args.w:
        w_vals = [float(v.strip()) for v in args.w.split(",")]

    clasificador = ClasificadorLineal(w=w_vals, b=b_val)
    print("=" * 70)
    print("EJECUCIÓN DEL PUNTO 1: CLASIFICADOR LINEAL PARAMETRIZABLE")
    print("=" * 70)
    print(f"Dimensión d:    {clasificador.dimension}")
    print(f"Vector de pesos: {clasificador.w}")
    print(f"Sesgo b:        {clasificador.b:.4f}")
    print(f"Norma ||w||_2:  {clasificador.norma_w:.4f}\n")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        from utils_csv import generar_datasets_taller
        generar_datasets_taller()

    df = pd.read_csv(csv_path)
    df_resultado = clasificador.clasificar_dataframe(df)
    
    Path(args.salida_csv).parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(args.salida_csv, index=False)
    print(f"Resultados guardados en: {args.salida_csv}")

    cols_feat = [c for c in df.columns if c.startswith("x_")]
    if len(cols_feat) < clasificador.dimension:
        cols_feat = df.select_dtypes(include=[np.number]).columns.tolist()[:clasificador.dimension]
    
    X_mat = df[cols_feat].to_numpy()
    clasificador.graficar_clasificacion(
        X=X_mat,
        y_pred=df_resultado["clase_predicha"].to_numpy(),
        f_x=df_resultado["f(x)"].to_numpy(),
        nombres_features=cols_feat,
        ruta_salida=args.salida_img
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
