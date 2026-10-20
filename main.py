"""
=============================================================================
TALLER COMPLETO: INTRODUCCIÓN A LA INTELIGENCIA ARTIFICIAL (NOTA III)
=============================================================================
Profesor / Científico de Datos: Solución magistral para Kevin
Arquitectura Totalmente Parametrizada:
  - CERO DATOS QUEMADOS: Pesos (w), sesgos/betas (b), número de clases (K),
    medias (mu), covarianzas (Sigma), priors, dimensiones y métricas son
    100% configurables vía diccionario, dataclass o archivo JSON externo.
  - MODO INFERENCIA Y PREDICCIÓN: Permite clasificar y predecir nuevas
    muestras individuales por línea de comandos o modo interactivo.
=============================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Asegurar importación de módulos locales
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

from config import (
    ConfiguracionTaller,
    ConfigPunto1,
    ConfigPunto2,
    ConfigPunto3,
    ConfigRutas,
    ConfigClaseGaussiana,
    obtener_configuracion_defecto
)
from utils_csv import generar_datasets_taller, cargar_csv_pandas, cargar_csv_nativo, cargar_csv_numpy
from punto1_clasificador_lineal import ClasificadorLineal
from punto2_clasificador_gaussiano import ClasificadorBayesianoGaussiano, ClaseGaussiana
from punto3_distancia_euclidea import BuscadorVecinosEuclideos
from predecir import MotorInferencia


def banner(titulo: str):
    ancho = 80
    print("\n" + "=" * ancho)
    print(f" {titulo.center(ancho - 2)} ")
    print("=" * ancho + "\n")


def parsear_muestra_str(texto: str) -> List[float]:
    """Convierte una cadena como '1.0, 2.5, -3.2' o '[1, 2]' a lista de floats."""
    limpio = texto.strip().lstrip("[").rstrip("]").replace(";", ",")
    return [float(val.strip()) for val in limpio.split(",") if val.strip()]


def ejecutar_modo_interactivo(motor: MotorInferencia):
    """Interfaz interactiva por consola para ingresar muestras y predecir."""
    banner("MODO INTERACTIVO DE PREDICCIÓN (INFERENCIA)")
    print("Estimado Kevin, bienvenido al entorno de pruebas de inferencia.")
    
    while True:
        print("\nSelecciona el modelo que deseas probar:")
        print("  1. Predecir con Clasificador Lineal (Punto 1)")
        print("  2. Predecir con Clasificador Bayesiano Gaussiano QDA (Punto 2)")
        print("  3. Buscar Vecinos Más Cercanos con Distancia Euclídea (Punto 3)")
        print("  4. Salir")
        
        opcion = input("\nIngresa tu opción (1-4): ").strip()
        if opcion == "1":
            dim = motor.clf_lineal.dimension
            print(f"\n-> Ingrese las {dim} características de la muestra separadas por comas.")
            print(f"   (Ejemplo para d={dim}: {', '.join(['1.0'] * dim)})")
            entrada = input("Coordenadas x: ")
            try:
                muestra = parsear_muestra_str(entrada)
                res = motor.predecir_muestra_lineal(muestra)
                print("\n" + motor.formatear_reporte_lineal(res))
            except Exception as e:
                print(f"[Error en predicción lineal]: {e}")

        elif opcion == "2":
            dim = motor.clf_gaussiano.dimension
            print(f"\n-> Ingrese las {dim} características de la muestra separadas por comas.")
            print(f"   (Ejemplo para d={dim}: {', '.join(['0.5'] * dim)})")
            entrada = input("Coordenadas x: ")
            try:
                muestra = parsear_muestra_str(entrada)
                res = motor.predecir_muestra_gaussiana(muestra)
                print("\n" + motor.formatear_reporte_gaussiano(res))
            except Exception as e:
                print(f"[Error en predicción gaussiana]: {e}")

        elif opcion == "3":
            dim = motor.buscador_vecinos.d if motor.buscador_vecinos else 2
            print(f"\n-> Ingrese las {dim} características de la consulta separadas por comas.")
            entrada = input("Coordenadas q: ")
            k_input = input(f"Número de vecinos k (presione enter para {motor.config.punto3.k_vecinos}): ").strip()
            k_val = int(k_input) if k_input else motor.config.punto3.k_vecinos
            try:
                muestra = parsear_muestra_str(entrada)
                res = motor.encontrar_vecinos_muestra(muestra, k=k_val)
                print("\n" + motor.formatear_reporte_vecinos(res))
            except Exception as e:
                print(f"[Error en búsqueda euclídea]: {e}")

        elif opcion == "4":
            print("\nFinalizando sesión interactiva. ¡Hasta pronto, Kevin!")
            break
        else:
            print("Opción no válida. Por favor elija un número del 1 al 4.")


def ejecutar_taller_parametrizado(cfg: ConfiguracionTaller):
    """
    Ejecuta el pipeline completo utilizando ÚNICAMENTE los parámetros
    definidos en el objeto de configuración 'cfg'.
    """
    banner("TALLER PARA LA NOTA III - INTRODUCCIÓN A LA IA (PARAMETRIZADO)")
    print("Estimado Kevin, ejecutando pipeline con configuración desacoplada (cero datos quemados).\n")

    # -----------------------------------------------------------------------
    # PASO 0: Ingesta de Datos y Verificación de Datasets
    # -----------------------------------------------------------------------
    banner("PASO 0: VERIFICACIÓN Y ESTUDIO DE CARGA DE DATOS CSV")
    
    ruta_p1_csv = Path(cfg.rutas.csv_punto1)
    if not ruta_p1_csv.exists():
        print(f"Generando datasets sintéticos en: {cfg.rutas.data_dir}...")
        generar_datasets_taller(cfg.rutas.data_dir)

    print("Comparación didáctica de los 3 métodos de carga de datos:")
    _, datos_nativo = cargar_csv_nativo(cfg.rutas.csv_punto1)
    print(f"  1. Módulo estándar 'csv':  {len(datos_nativo)} muestras leídas como listas nativas.")
    matriz_np = cargar_csv_numpy(cfg.rutas.csv_punto1)
    print(f"  2. Módulo 'numpy':        Array ndarray de dimensiones {matriz_np.shape}.")
    df_p1, X_p1 = cargar_csv_pandas(cfg.rutas.csv_punto1, columnas_features=cfg.punto1.columnas_features)
    print(f"  3. Biblioteca 'pandas':   DataFrame con {len(df_p1)} filas y columnas {list(df_p1.columns)}.")

    # -----------------------------------------------------------------------
    # PASO 1: Punto 1 - Clasificador Lineal f(x) = <w, x> + b (Parametrizado)
    # -----------------------------------------------------------------------
    banner("PASO 1: CLASIFICADOR LINEAL PARAMETRIZABLE f(x) = <w, x> + b")
    p1_params = cfg.punto1
    p1_clf = ClasificadorLineal(
        w=p1_params.w,
        b=p1_params.b,
        etiqueta_positiva=p1_params.etiqueta_positiva,
        etiqueta_negativa=p1_params.etiqueta_negativa,
        umbral_decision=p1_params.umbral_decision
    )

    print(f"Dimensión del espacio d:     {p1_clf.dimension}")
    print(f"Vector de pesos normal w:    {p1_clf.w.tolist()}")
    print(f"Sesgo escalar b (beta):      {p1_clf.b:.4f}")
    print(f"Norma euclídea ||w||_2:      {p1_clf.norma_w:.4f}")
    print(f"Umbral de decisión:          {p1_clf.umbral_decision:.2f}")

    df_p1_res = p1_clf.clasificar_dataframe(df_p1, columnas_features=p1_params.columnas_features)
    
    Path(cfg.rutas.salida_p1_csv).parent.mkdir(parents=True, exist_ok=True)
    df_p1_res.to_csv(cfg.rutas.salida_p1_csv, index=False)
    print(f"Reporte de clasificación exportado a: {cfg.rutas.salida_p1_csv}")

    cols_feat_p1 = p1_params.columnas_features or [c for c in df_p1.columns if c.startswith("x_")]
    if len(cols_feat_p1) < p1_clf.dimension:
        cols_feat_p1 = df_p1.select_dtypes(include=["number"]).columns.tolist()[:p1_clf.dimension]

    p1_clf.graficar_clasificacion(
        X=df_p1[cols_feat_p1].to_numpy(),
        y_pred=df_p1_res["clase_predicha"].to_numpy(),
        f_x=df_p1_res["f(x)"].to_numpy(),
        nombres_features=cols_feat_p1,
        ruta_salida=cfg.rutas.salida_p1_img
    )

    # -----------------------------------------------------------------------
    # PASO 2: Punto 2 - Clasificador Bayesiano Gaussiano QDA (Parametrizado)
    # -----------------------------------------------------------------------
    banner("PASO 2: CLASIFICADOR BAYESIANO GAUSSIANO MULTIVARIADO (QDA)")
    p2_params = cfg.punto2
    
    clases_gaussianas = [
        ClaseGaussiana(
            etiqueta=c.etiqueta,
            nombre=c.nombre,
            mu=c.mu,
            sigma=c.sigma,
            prior=c.prior
        )
        for c in p2_params.clases
    ]
    p2_clf = ClasificadorBayesianoGaussiano(clases=clases_gaussianas)

    print(f"Número de clases configuradas K: {p2_clf.k_clases}")
    print(f"Dimensión del espacio d:         {p2_clf.dimension}")
    for c in p2_clf.clases:
        print(f"  - [{c.nombre}] Media mu: {c.mu.tolist()} | Prior P(C): {c.prior:.3f} | Det|Sigma|: {c.det_sigma:.4f}")

    df_p2, X_p2 = cargar_csv_pandas(cfg.rutas.csv_punto2, columnas_features=p2_params.columnas_features)
    df_p2_res = p2_clf.clasificar_dataframe(df_p2, columnas_features=p2_params.columnas_features)
    
    Path(cfg.rutas.salida_p2_csv).parent.mkdir(parents=True, exist_ok=True)
    df_p2_res.to_csv(cfg.rutas.salida_p2_csv, index=False)
    print(f"\nReporte bayesiano exportado a: {cfg.rutas.salida_p2_csv}")

    p2_clf.graficar_regiones_decision(
        X_muestras=X_p2,
        y_muestras=df_p2_res["clase_predicha"].to_numpy(),
        rango_x=tuple(p2_params.rango_x) if p2_params.rango_x else None,
        rango_y=tuple(p2_params.rango_y) if p2_params.rango_y else None,
        resolucion=p2_params.resolucion_grid,
        ruta_salida=cfg.rutas.salida_p2_img
    )

    # -----------------------------------------------------------------------
    # PASO 3: Punto 3 - Búsqueda de Muestras Más Cercanas (Parametrizado)
    # -----------------------------------------------------------------------
    banner("PASO 3: BÚSQUEDA DE VECINOS MÁS CERCANOS (DISTANCIA EUCLÍDEA)")
    p3_params = cfg.punto3
    
    df_ref = cargar_csv_pandas(cfg.rutas.csv_punto3_ref)[0]
    df_qry = cargar_csv_pandas(cfg.rutas.csv_punto3_query)[0]

    id_col_ref = p3_params.columna_id_ref
    id_col_qry = p3_params.columna_id_query

    cols_ref = p3_params.columnas_features or [c for c in df_ref.columns if c != id_col_ref and df_ref[c].dtype in ("float64", "int64")]
    cols_qry = p3_params.columnas_features or [c for c in df_qry.columns if c != id_col_qry and df_qry[c].dtype in ("float64", "int64")]
    cols_comunes = [c for c in cols_ref if c in cols_qry]

    X_ref = df_ref[cols_comunes].to_numpy()
    ids_ref = df_ref[id_col_ref].tolist() if id_col_ref in df_ref.columns else None

    X_qry = df_qry[cols_comunes].to_numpy()
    ids_qry = df_qry[id_col_qry].tolist() if id_col_qry in df_qry.columns else None

    print(f"Muestras de Referencia N:     {len(X_ref)} (Dimensión d={X_ref.shape[1]})")
    print(f"Muestras de Consulta M:       {len(X_qry)}")
    print(f"Parámetro k vecinos:          {p3_params.k_vecinos}")
    print(f"Características consideradas: {cols_comunes}")

    buscador = BuscadorVecinosEuclideos(X_referencia=X_ref, ids_referencia=ids_ref)
    df_p3_res = buscador.generar_reporte_dataframe(
        Q=X_qry,
        k=p3_params.k_vecinos,
        ids_consultas=ids_qry
    )

    Path(cfg.rutas.salida_p3_csv).parent.mkdir(parents=True, exist_ok=True)
    df_p3_res.to_csv(cfg.rutas.salida_p3_csv, index=False)
    print(f"Reporte de vecinos exportado a: {cfg.rutas.salida_p3_csv}")

    buscador.graficar_vecinos_2d(
        Q=X_qry,
        k=p3_params.k_vecinos,
        ids_consultas=ids_qry,
        nombres_features=cols_comunes,
        ruta_salida=cfg.rutas.salida_p3_img
    )

    banner("PIPELINE EJECUTADO CON ÉXITO")
    print("Kevin, todos los algoritmos se ejecutaron sin valores quemados.")
    print("Archivos generados en la carpeta de salida configurada:")
    for f in Path(cfg.rutas.output_dir).glob("*"):
        print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Orquestador del Taller de IA - Ejecución Parametrizada e Inferencia"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Ruta al archivo JSON de configuración con todos los hiperparámetros"
    )
    parser.add_argument(
        "--guardar_config_defecto",
        action="store_true",
        help="Exporta la configuración canónica del taller a config.json y termina"
    )
    # Argumentos de Inferencia / Predicción
    parser.add_argument(
        "--predecir_p1",
        type=str,
        default=None,
        help="Predice una muestra individual en el Punto 1 (ej: '1.0, 2.0, -1.0, 0.5, 3.0')"
    )
    parser.add_argument(
        "--predecir_p2",
        type=str,
        default=None,
        help="Predice una muestra individual en el Punto 2 (ej: '2.5, 1.8')"
    )
    parser.add_argument(
        "--predecir_p3",
        type=str,
        default=None,
        help="Encuentra los vecinos más cercanos de una muestra en el Punto 3 (ej: '-1.0, 4.0')"
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=None,
        help="Número de vecinos k para la búsqueda euclídea"
    )
    parser.add_argument(
        "--interactivo",
        action="store_true",
        help="Inicia el menú interactivo por consola para ingresar muestras y predecir"
    )
    args = parser.parse_args()

    # 1. Regenerar plantilla config.json si se solicita
    if args.guardar_config_defecto:
        cfg_defecto = obtener_configuracion_defecto()
        cfg_defecto.to_json(args.config)
        print(f"Plantilla de configuración exportada exitosamente a: {args.config}")
        return

    # 2. Cargar configuración
    ruta_cfg = Path(args.config)
    if ruta_cfg.is_file():
        config = ConfiguracionTaller.from_json(ruta_cfg)
    else:
        config = obtener_configuracion_defecto()
        config.to_json("config.json")

    # 3. Verificar si el usuario solicitó inferencia / predicción
    motor = MotorInferencia(config=config)

    # Modo interactivo
    if args.interactivo:
        ejecutar_modo_interactivo(motor)
        return

    # Predicción Punto 1 por CLI
    if args.predecir_p1 is not None:
        muestra = parsear_muestra_str(args.predecir_p1)
        res = motor.predecir_muestra_lineal(muestra)
        print("\n" + motor.formatear_reporte_lineal(res))
        return

    # Predicción Punto 2 por CLI
    if args.predecir_p2 is not None:
        muestra = parsear_muestra_str(args.predecir_p2)
        res = motor.predecir_muestra_gaussiana(muestra)
        print("\n" + motor.formatear_reporte_gaussiano(res))
        return

    # Predicción Punto 3 por CLI
    if args.predecir_p3 is not None:
        muestra = parsear_muestra_str(args.predecir_p3)
        res = motor.encontrar_vecinos_muestra(muestra, k=args.top_k)
        print("\n" + motor.formatear_reporte_vecinos(res))
        return

    # 4. Si no se solicitó una predicción puntual, ejecutar el pipeline completo
    print(f"-> Utilizando configuración de hiperparámetros desde: {ruta_cfg}")
    ejecutar_taller_parametrizado(config)


if __name__ == "__main__":
    main()
