"""
=============================================================================
SUITE DE PRUEBAS UNITARIAS Y VALIDACIÓN MATEMÁTICA
=============================================================================
Profesor / Científico de Datos: Validación rigurosa para Kevin
Ejecución: pytest tests/test_taller.py
Incluye pruebas con parámetros del taller y con hiperparámetros arbitrarios
(nuevas clases K=4, nuevas dimensiones d=3, pesos y betas personalizados).
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Agregar directorios al PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from config import ConfiguracionTaller, ConfigClaseGaussiana, obtener_configuracion_defecto
from punto1_clasificador_lineal import ClasificadorLineal
from punto2_clasificador_gaussiano import ClasificadorBayesianoGaussiano, ClaseGaussiana
from punto3_distancia_euclidea import BuscadorVecinosEuclideos
from utils_csv import cargar_csv_nativo, cargar_csv_numpy, cargar_csv_pandas, generar_datasets_taller


# ---------------------------------------------------------------------------
# PRUEBAS DEL PUNTO 1: CLASIFICADOR LINEAL
# ---------------------------------------------------------------------------
class TestPunto1ClasificadorLineal:

    @pytest.fixture
    def clf(self):
        return ClasificadorLineal()

    def test_parametros_exactos_enunciado(self, clf):
        """Verifica que w y b coincidan exactamente con el enunciado del taller."""
        w_esperado = np.array([1.0 / 3.0, 1.0, -3.0, -2.0, 5.0 / 4.0])
        b_esperado = 2.0 / 3.0

        np.testing.assert_allclose(clf.w, w_esperado, atol=1e-12)
        assert pytest.approx(clf.b, 1e-12) == b_esperado

    def test_evaluacion_analitica_origen(self, clf):
        """En el origen x = 0, f(0) = b = 2/3 > 0, clase esperada = +1."""
        x_origen = np.zeros(5)
        f_val = clf.evaluar_funcion_discriminante(x_origen)[0]
        y_val = clf.predecir(x_origen)[0]

        assert pytest.approx(f_val, 1e-12) == (2.0 / 3.0)
        assert y_val == 1

    def test_evaluacion_vectores_canonicos(self, clf):
        """Prueba f(e_3) donde e_3 = (0, 0, 1, 0, 0): f(e_3) = -3 + 2/3 = -7/3 < 0."""
        e3 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        f_val = clf.evaluar_funcion_discriminante(e3)[0]
        y_val = clf.predecir(e3)[0]

        assert pytest.approx(f_val, 1e-12) == (-7.0 / 3.0)
        assert y_val == -1

    def test_norma_y_distancia_hiperplano(self, clf):
        """Verifica el cálculo de la norma euclídea ||w||_2 y la distancia ortogonal con signo."""
        norma_calculada = np.sqrt((1/3)**2 + 1**2 + (-3)**2 + (-2)**2 + (5/4)**2)
        assert pytest.approx(clf.norma_w, 1e-12) == norma_calculada

        x_origen = np.zeros(5)
        r_calculado = clf.distancia_con_signo_al_hiperplano(x_origen)[0]
        assert pytest.approx(r_calculado, 1e-12) == (2.0 / 3.0) / norma_calculada

    def test_parametrizacion_arbitraria_pesos_y_beta(self):
        """Verifica que acepte dimensiones arbitrarias (d=3), pesos w y beta b personalizados."""
        w_custom = [2.0, -1.0, 0.5]
        b_custom = -1.5
        clf_custom = ClasificadorLineal(
            w=w_custom,
            b=b_custom,
            etiqueta_positiva=10,
            etiqueta_negativa=-10,
            umbral_decision=0.5
        )

        assert clf_custom.dimension == 3
        assert clf_custom.b == -1.5
        
        # Muestra de prueba: x = [1, 0, 0] => f(x) = 2(1) - 1.5 = 0.5. Como f(x) >= umbral(0.5) => +10
        muestra = np.array([1.0, 0.0, 0.0])
        assert pytest.approx(clf_custom.evaluar_funcion_discriminante(muestra)[0], 1e-12) == 0.5
        assert clf_custom.predecir(muestra)[0] == 10

        # Muestra de prueba: x = [0, 1, 0] => f(x) = -1 - 1.5 = -2.5 < 0.5 => -10
        muestra2 = np.array([0.0, 1.0, 0.0])
        assert clf_custom.predecir(muestra2)[0] == -10


# ---------------------------------------------------------------------------
# PRUEBAS DEL PUNTO 2: CLASIFICADOR BAYESIANO GAUSSIANO
# ---------------------------------------------------------------------------
class TestPunto2ClasificadorGaussiano:

    @pytest.fixture
    def clf(self):
        return ClasificadorBayesianoGaussiano()

    def test_propiedades_matrices_covarianza(self, clf):
        """Verifica determinantes e inversas analíticas de las tres clases."""
        c1, c2, c3 = clf.clases

        assert pytest.approx(c1.det_sigma, 1e-12) == 0.75
        inv_esperada_c1 = (1.0 / 0.75) * np.array([[1.0, -0.5], [-0.5, 1.0]])
        np.testing.assert_allclose(c1.inv_sigma, inv_esperada_c1, atol=1e-12)

        assert pytest.approx(c2.det_sigma, 1e-12) == 0.91

        assert pytest.approx(c3.det_sigma, 1e-12) == 1.0
        inv_esperada_c3 = np.array([[2.0, 0.0], [0.0, 0.5]])
        np.testing.assert_allclose(c3.inv_sigma, inv_esperada_c3, atol=1e-12)

    def test_distancia_mahalanobis_en_los_centros(self, clf):
        """En el centro mu_k, la distancia de Mahalanobis a la clase k debe ser idénticamente cero."""
        for clase in clf.clases:
            d_m = clase.distancia_mahalanobis_cuadrada(clase.mu)[0]
            assert pytest.approx(d_m, 1e-12) == 0.0

    def test_clasificacion_en_los_centros(self, clf):
        """Evaluar que cada vector de medias se clasifique correctamente en su propia clase."""
        for clase in clf.clases:
            pred = clf.predecir(clase.mu)[0]
            assert pred == clase.etiqueta

    def test_conservacion_probabilidades_posteriores(self, clf):
        """La suma de probabilidades posteriores P(C_k|x) debe ser estrictamente 1.0."""
        muestras_arbitrarias = np.array([
            [0.0, 0.0],
            [4.0, 4.0],
            [-4.0, 3.0],
            [1.5, -2.5],
            [-10.0, 10.0]
        ])
        post = clf.predecir_probabilidades_posteriori(muestras_arbitrarias)
        sumas = np.sum(post, axis=1)
        np.testing.assert_allclose(sumas, np.ones(len(muestras_arbitrarias)), atol=1e-12)

    def test_parametrizacion_arbitraria_k_clases_y_dimensiones(self):
        """Verifica que el clasificador admita K=4 clases y dimensión d=3 sin restricciones."""
        clases_4 = [
            ClaseGaussiana(etiqueta=1, nombre="Clase A", mu=[0, 0, 0], sigma=np.eye(3), prior=0.25),
            ClaseGaussiana(etiqueta=2, nombre="Clase B", mu=[5, 5, 5], sigma=np.eye(3), prior=0.25),
            ClaseGaussiana(etiqueta=3, nombre="Clase C", mu=[-5, 5, 0], sigma=np.eye(3), prior=0.25),
            ClaseGaussiana(etiqueta=4, nombre="Clase D", mu=[0, -5, 5], sigma=np.eye(3), prior=0.25)
        ]
        clf_4 = ClasificadorBayesianoGaussiano(clases=clases_4)
        assert clf_4.k_clases == 4
        assert clf_4.dimension == 3

        # Cada centro debe clasificarse en su propia etiqueta
        for c in clases_4:
            assert clf_4.predecir(c.mu)[0] == c.etiqueta


# ---------------------------------------------------------------------------
# PRUEBAS DEL PUNTO 3: DISTANCIA EUCLÍDEA Y BÚSQUEDA DE VECINOS
# ---------------------------------------------------------------------------
class TestPunto3DistanciaEuclidea:

    def test_distancia_pitagorica_exacta(self):
        """Prueba triángulo rectángulo 3-4-5: d((0,0), (3,4)) = 5.0."""
        X_ref = np.array([[3.0, 4.0]])
        Q = np.array([[0.0, 0.0]])

        buscador = BuscadorVecinosEuclideos(X_referencia=X_ref)
        D_broad = buscador.calcular_matriz_distancias(Q, metodo="broadcasting")
        D_alg = buscador.calcular_matriz_distancias(Q, metodo="algebraico")

        assert pytest.approx(D_broad[0, 0], 1e-12) == 5.0
        assert pytest.approx(D_alg[0, 0], 1e-12) == 5.0

    def test_encontrar_k_vecinos_ordenados(self):
        """Verifica que los vecinos se ordenen de menor a mayor distancia."""
        X_ref = np.array([
            [0.0, 3.0],  # dist = 3
            [1.0, 0.0],  # dist = 1
            [0.0, 2.0]   # dist = 2
        ])
        Q = np.array([[0.0, 0.0]])

        buscador = BuscadorVecinosEuclideos(X_referencia=X_ref, ids_referencia=["R3", "R1", "R2"])
        res = buscador.encontrar_mas_cercanos(Q, k=3)
        vecinos = res[0]["vecinos"]

        assert vecinos[0]["id_ref"] == "R1"
        assert pytest.approx(vecinos[0]["distancia_euclidea"], 1e-12) == 1.0

        assert vecinos[1]["id_ref"] == "R2"
        assert pytest.approx(vecinos[1]["distancia_euclidea"], 1e-12) == 2.0

        assert vecinos[2]["id_ref"] == "R3"
        assert pytest.approx(vecinos[2]["distancia_euclidea"], 1e-12) == 3.0


# ---------------------------------------------------------------------------
# PRUEBAS DE CARGA CSV Y SERIALIZACIÓN DE CONFIGURACIÓN
# ---------------------------------------------------------------------------
class TestConfiguracionYCSV:

    def test_consistencia_entre_metodos_de_carga(self, tmp_path):
        """Garantiza que csv nativo, numpy y pandas obtengan valores numéricos idénticos."""
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text("v1,v2,v3\n1.0,2.5,-3.2\n4.1,0.0,9.9\n")

        _, datos_nativo = cargar_csv_nativo(csv_file)
        matriz_np = cargar_csv_numpy(csv_file)
        _, X_pd = cargar_csv_pandas(csv_file)

        matriz_nativo = np.array(datos_nativo)

        np.testing.assert_allclose(matriz_nativo, matriz_np, atol=1e-12)
        np.testing.assert_allclose(matriz_nativo, X_pd, atol=1e-12)

    def test_serializacion_deserializacion_config(self, tmp_path):
        """Verifica que la configuración completa pueda guardarse y cargarse desde JSON."""
        cfg = obtener_configuracion_defecto()
        cfg.punto1.w = [1.0, 2.0, 3.0]
        cfg.punto1.b = 0.5
        
        json_file = tmp_path / "test_config.json"
        cfg.to_json(json_file)

        cfg_cargada = ConfiguracionTaller.from_json(json_file)
        assert cfg_cargada.punto1.w == [1.0, 2.0, 3.0]
        assert cfg_cargada.punto1.b == 0.5
        assert len(cfg_cargada.punto2.clases) == 3


# ---------------------------------------------------------------------------
# PRUEBAS DEL MOTOR DE INFERENCIA (PREDICCIÓN DE MUESTRAS INDIVIDUALES)
# ---------------------------------------------------------------------------
class TestMotorInferencia:

    @pytest.fixture
    def motor(self):
        from predecir import MotorInferencia
        return MotorInferencia()

    def test_inferencia_lineal_ejemplo(self, motor):
        """Verifica la predicción de una muestra individual en Punto 1."""
        # Muestra en el origen: f(0) = 2/3 > 0 => clase +1
        res = motor.predecir_muestra_lineal([0.0, 0.0, 0.0, 0.0, 0.0])
        assert res["clase_predicha"] == 1
        assert pytest.approx(res["f_x"], 1e-12) == (2.0 / 3.0)

    def test_inferencia_gaussiana_ejemplo(self, motor):
        """Verifica la predicción bayesiana para una muestra cercana a la Clase 2."""
        res = motor.predecir_muestra_gaussiana([4.0, 4.0])
        assert res["clase_predicha"] == 2
        assert res["confianza_posterior"] > 0.99

    def test_inferencia_vecinos_ejemplo(self, motor):
        """Verifica la búsqueda de vecinos para una consulta arbitraria."""
        res = motor.encontrar_vecinos_muestra([0.0, 0.0], k=2)
        assert len(res["vecinos_cercanos"]) == 2
        assert res["vecinos_cercanos"][0]["distancia_euclidea"] <= res["vecinos_cercanos"][1]["distancia_euclidea"]

