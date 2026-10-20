# Guía Práctica de Configuración en Python Nativo (Sin Conda)

**Asignatura:** Introducción a la Inteligencia Artificial  
**Destinatario:** Elaborado con estándares pedagógicos para **Kevin**  
**Herramientas:** Python estándar (`venv`) + Gestor de paquetes (`pip`)  
**Compatibilidad:** macOS, Linux y Windows  

---

## 1. Introducción y Motivación

Estimado Kevin, aunque **Conda** es muy popular en Ciencia de Datos por gestionar dependencias binarias a nivel de sistema operativo (C++, CUDA, Fortran), **Python nativo (`venv`)** es el estándar universal en desarrollo de software por las siguientes ventajas:

1. **Ligereza extrema:** No instala binarios pesados ni administradores adicionales; un entorno `venv` solo pesa unos pocos megabytes más los paquetes instalados.
2. **Inclusión por defecto:** Viene preinstalado en cualquier distribución oficial de Python 3.3 en adelante (módulo estándar `venv`), sin necesidad de instalar Anaconda ni Miniconda.
3. **Portabilidad y Despliegue en la Nube:** Es el método preferido para contenedores Docker, microservicios, AWS Lambda, Google Cloud Run y servidores de integración continua (CI/CD).

A continuación, te explico el paso a paso exacto para configurar y ejecutar todo nuestro proyecto con `venv` y `pip`.

---

## 2. Requisitos Previos

Antes de comenzar, asegúrate de tener instalado Python en tu equipo (se recomienda versión **3.10**, **3.11** o **3.12**).

Abre tu terminal y verifica la versión disponible:

### En macOS / Linux:
```bash
python3 --version
```

### En Windows (PowerShell o Símbolo del Sistema):
```powershell
python --version
```

> **Nota:** Si la versión reportada es 3.10 o superior, estás listo para continuar.

---

## 3. Paso 1: Creación del Entorno Virtual (`venv`)

El entorno virtual es una carpeta aislada donde se instalarán las bibliotecas del proyecto sin interferir con el Python de tu sistema ni con otros proyectos.

1. Abre la terminal y ubícate en la raíz del proyecto `intro.IA`:
   ```bash
   cd /Users/kevingiraldocastaneda/Proyectos/intro.IA
   ```

2. Ejecuta el comando de creación de entorno utilizando el módulo estándar `venv`. Nombraremos la carpeta `.venv`:

   - **En macOS / Linux:**
     ```bash
     python3 -m venv .venv
     ```

   - **En Windows:**
     ```powershell
     python -m venv .venv
     ```

Este comando creará una carpeta oculta llamada `.venv` que contiene copias aisladas de los ejecutables de Python y `pip`.

---

## 4. Paso 2: Activación del Entorno Virtual

Para que tu terminal utilice este entorno aislado en lugar del Python global, debes **activarlo**:

### En macOS / Linux (Zsh / Bash):
```bash
source .venv/bin/activate
```

### En Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

*(Si PowerShell muestra un error de políticas de ejecución en Windows, ejecuta una sola vez: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*.

### En Windows (CMD / Símbolo del Sistema tradicional):
```cmd
.venv\Scripts\activate.bat
```

### ¿Cómo saber si está activado?
Observarás que el prompt de tu terminal ahora tiene el prefijo `(.venv)` al inicio de la línea:
```text
(.venv) usuario@maquina intro.IA %
```

---

## 5. Paso 3: Actualización de `pip` e Instalación de Dependencias

Una vez activado el entorno, el primer paso recomendado es actualizar el instalador de paquetes `pip` a su versión más reciente:

```bash
pip install --upgrade pip
```

A continuación, instalamos todas las dependencias del proyecto utilizando el archivo [requirements.txt](file:///Users/kevingiraldocastaneda/Proyectos/intro.IA/requirements.txt):

```bash
pip install -r requirements.txt
```

Esto instalará automáticamente:
- **`numpy`**: Álgebra lineal vectorizada y operaciones matriciales.
- **`pandas`**: Ingesta y manipulación de archivos CSV.
- **`scipy`**: Rutinas científicas avanzadas.
- **`matplotlib` & `seaborn`**: Generación de gráficos y superficies de decisión.
- **`scikit-learn`**: Métricas y transformaciones auxiliares.
- **`pytest`**: Suite de validación matemática y pruebas unitarias.

---

## 6. Paso 4: Verificación de la Instalación

Para comprobar que todas las librerías se instalaron correctamente dentro del entorno virtual, ejecuta el siguiente comando de una sola línea:

```bash
python -c "import numpy, pandas, scipy, matplotlib, seaborn, pytest; print('¡Entorno venv configurado y verificado con éxito!')"
```

Si no arroja errores y ves el mensaje de éxito, tu entorno está 100% operativo.

---

## 7. Paso 5: Cómo Usar y Ejecutar el Proyecto

Con el entorno virtual activado `(.venv)`, puedes utilizar todas las funcionalidades del proyecto:

### A. Ejecutar Todo el Taller Automatizado
Ejecuta los tres puntos consecutivamente, cargando datos desde CSV, generando las clasificaciones y guardando las imágenes en `outputs/`:
```bash
python main.py
```

---

### B. Modo Interactivo de Predicción (Recomendado para Clases o Evaluaciones)
Si deseas ingresar las coordenadas de una muestra por consola y obtener el resultado de clasificación al instante:
```bash
python main.py --interactivo
```
Se desplegará el menú interactivo para evaluar muestras en Punto 1, Punto 2 o Punto 3.

---

### C. Predicción Rápida por Línea de Comandos (CLI)
Puedes probar muestras específicas directamente desde la terminal:

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

### D. Ejecución de Módulos Individuales con Parámetros Personalizados
Si necesitas cambiar pesos, betas o clases sin editar código:

- **Punto 1 (Pesos y Beta personalizados):**
  ```bash
  python src/punto1_clasificador_lineal.py --w "2.0,-1.5,0.8,1.0,-0.5" --b -0.5
  ```

- **Punto 2 (Clasificador Gaussiano desde configuración JSON):**
  ```bash
  python src/punto2_clasificador_gaussiano.py --config config.json
  ```

- **Punto 3 (Búsqueda de vecinos con k variable):**
  ```bash
  python src/punto3_distancia_euclidea.py -k 5
  ```

---

### E. Ejecutar la Suite de Pruebas Unitarias
Para verificar que todas las matemáticas y algoritmos son exactos:
```bash
pytest tests/test_taller.py -v
```
> Deberás ver las **17 pruebas unitarias aprobadas al 100%**.

---

## 8. Paso 6: Desactivación y Mantenimiento del Entorno

### Desactivar el Entorno
Cuando termines de trabajar en el proyecto, simplemente escribe:
```bash
deactivate
```
El prefijo `(.venv)` desaparecerá y regresarás al shell global de tu máquina.

### ¿Cómo eliminar el entorno si deseas reiniciar desde cero?
Dado que `venv` es puramente una carpeta local, si alguna vez deseas resetearlo por completo, solo debes borrar la carpeta `.venv`:
```bash
rm -rf .venv
```
Y repetir desde el [Paso 1](#3-paso-1-creación-del-entorno-virtual-venv).

---

## 9. Tabla Comparativa Resumen: `venv` vs. `conda`

| Acción | Con Python Nativo (`venv`) | Con Conda (`conda`) |
| :--- | :--- | :--- |
| **Crear entorno** | `python3 -m venv .venv` | `conda create -n intro_ia python=3.11` |
| **Activar entorno** | `source .venv/bin/activate` (Mac/Linux)<br>`.venv\Scripts\Activate.ps1` (Win) | `conda activate intro_ia` |
| **Instalar librerías** | `pip install -r requirements.txt` | `conda env update -f environment.yml` |
| **Desactivar** | `deactivate` | `conda deactivate` |
| **Eliminar entorno** | `rm -rf .venv` | `conda env remove -n intro_ia` |
| **Tamaño en disco** | Muy ligero (~150 MB con dependencias) | Más pesado (~1-2 GB con runtime base) |
