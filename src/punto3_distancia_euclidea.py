"""
=============================================================================
PUNTO 3: BÚSQUEDA DE MUESTRAS MÁS CERCANAS MEDIANTE DISTANCIA EUCLÍDEA
=============================================================================
Asignatura: Introducción a la Inteligencia Artificial
Autor: Solución pedagógica para Kevin
Totalmente parametrizable: admite k vecinos arbitrarios, dimensiones d
arbitrarias, identificadores dinámicos y proyección visual adaptable.
=============================================================================
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class BuscadorVecinosEuclideos:
    """
    Motor vectorial para la búsqueda de las muestras más cercanas (k-Nearest Neighbors)
    utilizando la métrica de Distancia Euclídea L2.
    Totalmente parametrizable en cualquier dimensión d >= 1.
    """

    def __init__(self, X_referencia: np.ndarray, ids_referencia: Optional[List[str]] = None) -> None:
        self.X_ref = np.asarray(X_referencia, dtype=np.float64)
        if self.X_ref.ndim != 2:
            raise ValueError(f"X_referencia debe ser una matriz bidimensional (N, d). Recibido ndim={self.X_ref.ndim}")

        self.n_muestras_ref, self.d = self.X_ref.shape

        if ids_referencia is not None:
            if len(ids_referencia) != self.n_muestras_ref:
                raise ValueError("La longitud de ids_referencia no coincide con el número de muestras.")
            self.ids_ref = [str(i) for i in ids_referencia]
        else:
            self.ids_ref = [f"Ref_{i+1:02d}" for i in range(self.n_muestras_ref)]

    def calcular_matriz_distancias(self, Q: np.ndarray, metodo: str = "broadcasting") -> np.ndarray:
        """Calcula la matriz completa de distancias D de dimensión (M, N)."""
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
            
        if Q.shape[1] != self.d:
            raise ValueError(f"Dimensión inconsistente: Q tiene {Q.shape[1]} variables pero la referencia tiene {self.d}")

        if metodo == "broadcasting":
            diferencias = Q[:, np.newaxis, :] - self.X_ref[np.newaxis, :, :]
            return np.linalg.norm(diferencias, axis=2)
        elif metodo == "algebraico":
            q_norm_sq = np.sum(Q ** 2, axis=1, keepdims=True)
            x_norm_sq = np.sum(self.X_ref ** 2, axis=1, keepdims=True).T
            cross_term = 2.0 * np.dot(Q, self.X_ref.T)
            dist_sq = q_norm_sq + x_norm_sq - cross_term
            dist_sq = np.maximum(dist_sq, 0.0)
            return np.sqrt(dist_sq)
        else:
            raise ValueError(f"Método '{metodo}' no reconocido. Use 'broadcasting' o 'algebraico'.")

    def encontrar_mas_cercanos(
        self,
        Q: np.ndarray,
        k: int = 1,
        ids_consultas: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Para cada muestra en Q, encuentra las k muestras más cercanas de la referencia.
        """
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)

        m_consultas = Q.shape[0]
        k_efectivo = min(max(1, int(k)), self.n_muestras_ref)

        if ids_consultas is None:
            ids_consultas = [f"Qry_{i+1:02d}" for i in range(m_consultas)]
        else:
            ids_consultas = [str(i) for i in ids_consultas]

        D = self.calcular_matriz_distancias(Q)  # (M, N)
        resultados = []

        for i in range(m_consultas):
            distancias_i = D[i, :]
            indices_ordenados = np.argsort(distancias_i)[:k_efectivo]
            
            info_vecinos = []
            for rango, idx_ref in enumerate(indices_ordenados, start=1):
                info_vecinos.append({
                    "rango": rango,
                    "indice_ref": int(idx_ref),
                    "id_ref": self.ids_ref[idx_ref],
                    "coord_ref": self.X_ref[idx_ref].tolist(),
                    "distancia_euclidea": float(distancias_i[idx_ref])
                })

            resultados.append({
                "indice_consulta": i,
                "id_consulta": ids_consultas[i],
                "coord_consulta": Q[i].tolist(),
                "vecinos": info_vecinos
            })

        return resultados

    def generar_reporte_dataframe(
        self,
        Q: np.ndarray,
        k: int = 1,
        ids_consultas: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Genera un DataFrame tabular plano apto para exportar a CSV."""
        resultados = self.encontrar_mas_cercanos(Q, k=k, ids_consultas=ids_consultas)
        filas = []

        for res in resultados:
            id_q = res["id_consulta"]
            coord_q = res["coord_consulta"]
            for v in res["vecinos"]:
                fila = {
                    "id_consulta": id_q,
                    "coord_consulta": str(coord_q),
                    "rango_cercania": v["rango"],
                    "id_referencia_mas_cercana": v["id_ref"],
                    "coord_referencia": str(v["coord_ref"]),
                    "distancia_euclidea": round(v["distancia_euclidea"], 6)
                }
                filas.append(fila)

        return pd.DataFrame(filas)

    def graficar_vecinos_2d(
        self,
        Q: np.ndarray,
        k: int = 1,
        ids_consultas: Optional[List[str]] = None,
        indices_features: Tuple[int, int] = (0, 1),
        nombres_features: Optional[List[str]] = None,
        ruta_salida: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Visualización gráfica adaptable:
        - Si d=2: representación exacta en el plano euclídeo.
        - Si d>2: proyección en los 2 ejes especificados (default: primeras dos características).
        """
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)

        resultados = self.encontrar_mas_cercanos(Q, k=k, ids_consultas=ids_consultas)

        idx1, idx2 = indices_features
        nom1 = nombres_features[idx1] if nombres_features and idx1 < len(nombres_features) else f"$x_{{{idx1+1}}}$"
        nom2 = nombres_features[idx2] if nombres_features and idx2 < len(nombres_features) else f"$x_{{{idx2+1}}}$"

        sns.set_theme(style="whitegrid", palette="muted")
        fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)

        # 1. Graficar muestras de referencia
        ax.scatter(
            self.X_ref[:, idx1], self.X_ref[:, idx2],
            c="#3498db", s=90, edgecolors="black", alpha=0.8,
            label=f"Muestras de Referencia ($N={self.n_muestras_ref}$)", zorder=3
        )
        for i, (x_coord, y_coord) in enumerate(zip(self.X_ref[:, idx1], self.X_ref[:, idx2])):
            ax.annotate(
                self.ids_ref[i], (x_coord, y_coord),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7.5, color="#2c3e50", alpha=0.85
            )

        # 2. Graficar muestras de consulta
        ax.scatter(
            Q[:, idx1], Q[:, idx2],
            c="#e67e22", s=180, marker="D", edgecolors="black", linewidths=1.5,
            label=f"Muestras de Consulta ($M={Q.shape[0]}$)", zorder=5
        )

        # 3. Dibujar enlaces hacia los vecinos
        colores_flechas = ["#e74c3c", "#9b59b6", "#16a085", "#f39c12", "#27ae60", "#2980b9"]

        for idx_q, res in enumerate(resultados):
            coord_q = np.array(res["coord_consulta"])
            id_q = res["id_consulta"]
            color_enlace = colores_flechas[idx_q % len(colores_flechas)]

            ax.annotate(
                f"{id_q}", (coord_q[idx1], coord_q[idx2]),
                textcoords="offset points", xytext=(-8, 10),
                fontsize=9.5, fontweight="bold", color="#d35400"
            )

            for v in res["vecinos"]:
                coord_ref = np.array(v["coord_ref"])
                dist = v["distancia_euclidea"]
                rango = v["rango"]

                ax.annotate(
                    "", xy=(coord_ref[idx1], coord_ref[idx2]), xytext=(coord_q[idx1], coord_q[idx2]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color_enlace,
                        lw=1.8 if rango == 1 else 1.2,
                        linestyle="-" if rango == 1 else ":",
                        alpha=0.9 if rango == 1 else 0.6
                    ),
                    zorder=4
                )

                punto_medio = 0.5 * (coord_q[[idx1, idx2]] + coord_ref[[idx1, idx2]])
                ax.text(
                    punto_medio[0], punto_medio[1],
                    f"d={dist:.2f}",
                    fontsize=8, fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color_enlace, alpha=0.9),
                    zorder=6
                )

        subtitulo = f" (Proyección {nom1} vs {nom2})" if self.d > 2 else ""
        ax.set_title(
            f"Búsqueda de Muestras Más Cercanas con Distancia Euclídea ($k={k}$){subtitulo}",
            fontsize=14, fontweight="bold", pad=15
        )
        ax.set_xlabel(nom1, fontsize=12)
        ax.set_ylabel(nom2, fontsize=12)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()

        if ruta_salida:
            Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
            print(f"-> Gráfica de vecinos euclídeos guardada en: {ruta_salida}")

        return fig


def main():
    """Ejecución y demostración parametrizable del Punto 3."""
    parser = argparse.ArgumentParser(description="Punto 3: Búsqueda de Muestras Más Cercanas Parametrizable")
    parser.add_argument("--config", type=str, default=None, help="Ruta a archivo de configuración JSON")
    parser.add_argument("--ref_csv", type=str, default="data/punto3_referencia.csv", help="CSV de referencia")
    parser.add_argument("--query_csv", type=str, default="data/punto3_consultas.csv", help="CSV de consulta")
    parser.add_argument("-k", "--top_k", type=int, default=None, help="Número de vecinos k")
    parser.add_argument("--salida_img", type=str, default="outputs/punto3_vecinos_cercanos.png", help="Ruta imagen")
    parser.add_argument("--salida_csv", type=str, default="outputs/punto3_resultados.csv", help="Ruta reporte CSV")
    args = parser.parse_args()

    k_val = args.top_k
    ref_path = Path(args.ref_csv)
    query_path = Path(args.query_csv)
    col_id_ref = "id_muestra"
    col_id_qry = "id_consulta"

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        p3_cfg = cfg.get("punto3", {})
        if k_val is None:
            k_val = p3_cfg.get("k_vecinos", 2)
        col_id_ref = p3_cfg.get("columna_id_ref", col_id_ref)
        col_id_qry = p3_cfg.get("columna_id_query", col_id_qry)

    if k_val is None:
        k_val = 2

    if not ref_path.exists() or not query_path.exists():
        from utils_csv import generar_datasets_taller
        generar_datasets_taller()

    df_ref = pd.read_csv(ref_path)
    df_query = pd.read_csv(query_path)

    # Autodetectar columnas de características numéricas
    cols_coord_ref = [c for c in df_ref.columns if c != col_id_ref and np.issubdtype(df_ref[c].dtype, np.number)]
    cols_coord_qry = [c for c in df_query.columns if c != col_id_qry and np.issubdtype(df_query[c].dtype, np.number)]
    
    # Usar la intersección ordenada
    cols_comunes = [c for c in cols_coord_ref if c in cols_coord_qry]
    if not cols_comunes:
        cols_comunes = cols_coord_ref

    X_ref = df_ref[cols_comunes].to_numpy()
    ids_ref = df_ref[col_id_ref].tolist() if col_id_ref in df_ref.columns else None

    X_query = df_query[cols_comunes].to_numpy()
    ids_query = df_query[col_id_qry].tolist() if col_id_qry in df_query.columns else None

    print("=" * 75)
    print("EJECUCIÓN DEL PUNTO 3: BÚSQUEDA MEDIANTE DISTANCIA EUCLÍDEA PARAMETRIZADA")
    print("=" * 75)
    print(f"Base de Referencia: {len(X_ref)} muestras de dimensión {X_ref.shape[1]}")
    print(f"Lote de Consultas:  {len(X_query)} muestras de dimensión {X_query.shape[1]}")
    print(f"Parámetro k vecinos: {k_val}")
    print(f"Características consideradas: {cols_comunes}")

    buscador = BuscadorVecinosEuclideos(X_referencia=X_ref, ids_referencia=ids_ref)
    df_resultados = buscador.generar_reporte_dataframe(Q=X_query, k=k_val, ids_consultas=ids_query)

    Path(args.salida_csv).parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(args.salida_csv, index=False)
    print(f"Reporte detallado guardado en: {args.salida_csv}")

    buscador.graficar_vecinos_2d(
        Q=X_query,
        k=k_val,
        ids_consultas=ids_query,
        nombres_features=cols_comunes,
        ruta_salida=args.salida_img
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
