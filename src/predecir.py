"""
=============================================================================
MÓDULO DE INFERENCIA Y PREDICCIÓN DE MUESTRAS INDIVIDUALES / LOTES
=============================================================================
Autor: Creado para Kevin - Introducción a la Inteligencia Artificial
Propósito: Proveer un motor de predicción desacoplado que permita evaluar
           cualquier muestra nueva proporcionada por el usuario en:
           1. Clasificador Lineal (Punto 1)
           2. Clasificador Bayesiano Gaussiano QDA (Punto 2)
           3. Búsqueda de Muestras Más Cercanas con Distancia Euclídea (Punto 3)
=============================================================================
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd

from config import ConfiguracionTaller, obtener_configuracion_defecto
from punto1_clasificador_lineal import ClasificadorLineal
from punto2_clasificador_gaussiano import ClasificadorBayesianoGaussiano, ClaseGaussiana
from punto3_distancia_euclidea import BuscadorVecinosEuclideos
from utils_csv import cargar_csv_pandas


class MotorInferencia:
    """
    Orquesta la inferencia y predicción para nuevos ejemplos en los 3 modelos del taller.
    """

    def __init__(self, config: Optional[ConfiguracionTaller] = None) -> None:
        self.config = config if config is not None else obtener_configuracion_defecto()
        
        # 1. Instanciar Clasificador Lineal (Punto 1)
        p1 = self.config.punto1
        self.clf_lineal = ClasificadorLineal(
            w=p1.w,
            b=p1.b,
            etiqueta_positiva=p1.etiqueta_positiva,
            etiqueta_negativa=p1.etiqueta_negativa,
            umbral_decision=p1.umbral_decision
        )

        # 2. Instanciar Clasificador Bayesiano Gaussiano (Punto 2)
        p2 = self.config.punto2
        clases = [
            ClaseGaussiana(
                etiqueta=c.etiqueta,
                nombre=c.nombre,
                mu=c.mu,
                sigma=c.sigma,
                prior=c.prior
            )
            for c in p2.clases
        ]
        self.clf_gaussiano = ClasificadorBayesianoGaussiano(clases=clases)

        # 3. Inicializar Buscador de Vecinos Euclídeos (Punto 3)
        self._inicializar_buscador_vecinos()

    def _inicializar_buscador_vecinos(self) -> None:
        """Carga la base de referencia configurada para la búsqueda euclídea."""
        p3 = self.config.punto3
        ruta_ref = Path(self.config.rutas.csv_punto3_ref)
        
        if ruta_ref.is_file():
            df_ref = pd.read_csv(ruta_ref)
            id_col = p3.columna_id_ref
            
            if p3.columnas_features:
                cols_feat = p3.columnas_features
            else:
                cols_feat = [c for c in df_ref.columns if c != id_col and np.issubdtype(df_ref[c].dtype, np.number)]
            
            X_ref = df_ref[cols_feat].to_numpy()
            ids_ref = df_ref[id_col].tolist() if id_col in df_ref.columns else None
            self.buscador_vecinos = BuscadorVecinosEuclideos(X_referencia=X_ref, ids_referencia=ids_ref)
            self.columnas_vecinos = cols_feat
        else:
            self.buscador_vecinos = None
            self.columnas_vecinos = []

    # -----------------------------------------------------------------------
    # Inferencia Punto 1: Clasificador Lineal
    # -----------------------------------------------------------------------
    def predecir_muestra_lineal(self, x: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Clasifica una muestra individual usando f(x) = <w, x> + b.
        """
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        if len(x_arr) != self.clf_lineal.dimension:
            raise ValueError(
                f"Dimensión incorrecta para el Clasificador Lineal: se esperaban {self.clf_lineal.dimension} "
                f"características, pero se recibieron {len(x_arr)}."
            )

        dot_prod = float(self.clf_lineal.producto_interno(x_arr)[0])
        f_val = float(self.clf_lineal.evaluar_funcion_discriminante(x_arr)[0])
        distancia = float(self.clf_lineal.distancia_con_signo_al_hiperplano(x_arr)[0])
        prob_pos = float(self.clf_lineal.predecir_probabilidad_logistica(x_arr)[0])
        pred = int(self.clf_lineal.predecir(x_arr)[0])

        return {
            "muestra": x_arr.tolist(),
            "dimension": self.clf_lineal.dimension,
            "vector_w": self.clf_lineal.w.tolist(),
            "sesgo_b": self.clf_lineal.b,
            "producto_interno": dot_prod,
            "f_x": f_val,
            "umbral": self.clf_lineal.umbral_decision,
            "distancia_ortogonal": distancia,
            "distancia_absoluta": abs(distancia),
            "probabilidad_positiva": prob_pos,
            "probabilidad_negativa": 1.0 - prob_pos,
            "clase_predicha": pred,
            "nombre_clase": f"Clase Positiva (+{self.clf_lineal.etiqueta_positiva})" if pred == self.clf_lineal.etiqueta_positiva else f"Clase Negativa ({self.clf_lineal.etiqueta_negativa})"
        }

    # -----------------------------------------------------------------------
    # Inferencia Punto 2: Clasificador Bayesiano Gaussiano (QDA)
    # -----------------------------------------------------------------------
    def predecir_muestra_gaussiana(self, x: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Clasifica una muestra individual calculando densidades, distancias de Mahalanobis
        y probabilidades posteriores de Bayes P(C_k | x).
        """
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        if len(x_arr) != self.clf_gaussiano.dimension:
            raise ValueError(
                f"Dimensión incorrecta para el Clasificador Bayesiano: se esperaban {self.clf_gaussiano.dimension} "
                f"características, pero se recibieron {len(x_arr)}."
            )

        G = self.clf_gaussiano.calcular_funciones_discriminantes(x_arr)[0]
        densidades = self.clf_gaussiano.calcular_densidades(x_arr)[0]
        dist_mahalanobis = self.clf_gaussiano.calcular_distancias_mahalanobis(x_arr)[0]
        posteriores = self.clf_gaussiano.predecir_probabilidades_posteriori(x_arr)[0]
        pred_label = self.clf_gaussiano.predecir(x_arr)[0]

        detalle_clases = []
        for idx, clase in enumerate(self.clf_gaussiano.clases):
            detalle_clases.append({
                "etiqueta": clase.etiqueta,
                "nombre": clase.nombre,
                "media_mu": clase.mu.tolist(),
                "prior": clase.prior,
                "distancia_mahalanobis": float(dist_mahalanobis[idx]),
                "densidad_p_x_dado_c": float(densidades[idx]),
                "discriminante_g_k": float(G[idx]),
                "probabilidad_posterior": float(posteriores[idx])
            })

        idx_ganador = int(np.argmax(posteriores))
        clase_ganadora = self.clf_gaussiano.clases[idx_ganador]

        return {
            "muestra": x_arr.tolist(),
            "dimension": self.clf_gaussiano.dimension,
            "clase_predicha": pred_label,
            "nombre_clase_predicha": clase_ganadora.nombre,
            "confianza_posterior": float(posteriores[idx_ganador]),
            "detalle_clases": detalle_clases
        }

    # -----------------------------------------------------------------------
    # Inferencia Punto 3: Muestras Más Cercanas con Distancia Euclídea
    # -----------------------------------------------------------------------
    def encontrar_vecinos_muestra(
        self,
        q: Union[List[float], np.ndarray],
        k: Optional[int] = None,
        id_consulta: str = "Ejemplo_Usuario"
    ) -> Dict[str, Any]:
        """
        Encuentra las k muestras más cercanas de la base de conocimiento para una muestra dada.
        """
        if self.buscador_vecinos is None:
            self._inicializar_buscador_vecinos()
            if self.buscador_vecinos is None:
                raise RuntimeError(
                    f"No se pudo cargar la base de referencia desde '{self.config.rutas.csv_punto3_ref}'."
                )

        q_arr = np.asarray(q, dtype=np.float64).flatten()
        k_efectivo = k if k is not None else self.config.punto3.k_vecinos

        res = self.buscador_vecinos.encontrar_mas_cercanos(
            Q=q_arr,
            k=k_efectivo,
            ids_consultas=[id_consulta]
        )[0]

        return {
            "id_consulta": id_consulta,
            "muestra_consulta": q_arr.tolist(),
            "k_vecinos_solicitados": k_efectivo,
            "vecinos_cercanos": res["vecinos"]
        }

    # -----------------------------------------------------------------------
    # Generadores de Reportes Formateados para Consola / UI
    # -----------------------------------------------------------------------
    @staticmethod
    def formatear_reporte_lineal(res: Dict[str, Any]) -> str:
        lineas = [
            "---------------------------------------------------------------------------",
            "PREDICCIÓN CON CLASIFICADOR LINEAL [f(x) = <w, x> + b]",
            "---------------------------------------------------------------------------",
            f"Muestra evaluada (x):       {res['muestra']}",
            f"Dimensión d:               {res['dimension']}",
            f"Vector de pesos (w):       {res['vector_w']}",
            f"Sesgo / Beta (b):          {res['sesgo_b']:.4f}",
            f"Producto Escalar <w, x>:   {res['producto_interno']:.4f}",
            f"Valor Discriminante f(x):  {res['f_x']:.4f}  (Umbral de decisión: {res['umbral']:.2f})",
            f"Distancia al hiperplano:   {res['distancia_ortogonal']:+.4f} (|r| = {res['distancia_absoluta']:.4f})",
            f"Probabilidad sigmoide:     P(Clase=+1|x) = {res['probabilidad_positiva'] * 100:.2f}%",
            "---------------------------------------------------------------------------",
            f"VEREDICTO:                 {res['nombre_clase']} (Etiqueta: {res['clase_predicha']:+d})",
            "---------------------------------------------------------------------------"
        ]
        return "\n".join(lineas)

    @staticmethod
    def formatear_reporte_gaussiano(res: Dict[str, Any]) -> str:
        lineas = [
            "---------------------------------------------------------------------------",
            "PREDICCIÓN CON CLASIFICADOR BAYESIANO GAUSSIANO (QDA)",
            "---------------------------------------------------------------------------",
            f"Muestra evaluada (x):       {res['muestra']}",
            f"Dimensión d:               {res['dimension']}",
            "Evaluación por Clase:"
        ]
        for c in res["detalle_clases"]:
            lineas.append(
                f"  [{c['nombre']}] D_Mahalanobis: {c['distancia_mahalanobis']:.4f} | "
                f"Score g(x): {c['discriminante_g_k']:+.3f} | "
                f"Probabilidad Posterior P(C|x): {c['probabilidad_posterior'] * 100:6.2f}%"
            )
        lineas.extend([
            "---------------------------------------------------------------------------",
            f"VEREDICTO DE BAYES:        {res['nombre_clase_predicha']} (Etiqueta: {res['clase_predicha']})",
            f"Nivel de Confianza:        {res['confianza_posterior'] * 100:.2f}%",
            "---------------------------------------------------------------------------"
        ])
        return "\n".join(lineas)

    @staticmethod
    def formatear_reporte_vecinos(res: Dict[str, Any]) -> str:
        lineas = [
            "---------------------------------------------------------------------------",
            f"BÚSQUEDA DE MUESTRAS MÁS CERCANAS (DISTANCIA EUCLÍDEA, k={res['k_vecinos_solicitados']})",
            "---------------------------------------------------------------------------",
            f"Consulta ({res['id_consulta']}): {res['muestra_consulta']}",
            "Vecinos más cercanos encontrados:"
        ]
        for v in res["vecinos_cercanos"]:
            lineas.append(
                f"  Rango {v['rango']}: ID '{v['id_ref']}' | "
                f"Coordenadas: {v['coord_ref']} | "
                f"Distancia Euclídea d = {v['distancia_euclidea']:.4f}"
            )
        lineas.append("---------------------------------------------------------------------------")
        return "\n".join(lineas)
