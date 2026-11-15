# Pronóstico del precio de Bitcoin con MLP multisalida

Proyecto Integrador de Deep Learning (sección 2.21) · Universidad del Norte · Noviembre de 2026
Autores: Anyely Sanjuan Benavides y Juan David Díaz Fernández · Profesor: Dr. Lihki Rubio

Pronosticamos el precio de cierre diario de BTCUSDT (Binance, 2018–2025) a 7, 14, 21 y 28 días con un MLP multisalida en Keras. Para cada combinación de lag (7, 14, 21, 28) y horizonte buscamos la mejor arquitectura entre 584 candidatas, validamos con las tres estrategias de `timeseries-cv` (Group K-Fold, Forward Chaining y K-Fold), diagnosticamos los residuos con el test BDS y evaluamos en un hold-out temporal de 292 días. El mejor modelo por horizonte se sirve con una API en FastAPI.

**Informe (Jupyter Book):** https://juanddf.github.io/Tarea2DeepLearning/

## Estructura

```
Tarea2DeepLearning/
├── data/btc_1d.csv                 # Serie diaria BTCUSDT (Binance)
├── notebooks/
│   ├── 1_eda.ipynb                 # Exploración de la serie
│   ├── 2_architecture_search.ipynb # Búsqueda de arquitectura (Top-5, curvas, efectos)
│   ├── 3_model_training.ipynb      # Evaluación fold a fold, estrategias, hold-out
│   ├── 4_residual_analysis.ipynb   # BDS, Ljung-Box y ACF de residuos
│   └── figs/                       # Figuras
├── src/
│   ├── train_t2.py                 # Entrenamiento reanudable con tope de 18 h (Colab, 8 CPU)
│   └── export_models.py            # Copia el mejor modelo por horizonte a app/
├── app/
│   ├── api.py                      # API de inferencia (FastAPI)
│   ├── schemas.py                  # Modelos Pydantic
│   ├── model_h{7,14,21,28}.keras   # Mejor MLP multisalida por horizonte
│   └── scalers_h{7,14,21,28}.joblib
├── results/                        # Tablas CSV (búsqueda, métricas, hold-out, BDS)
├── tests/                          # Pruebas de la API y de los modelos
├── book/                           # Páginas y estilo del Jupyter Book
├── Dockerfile
├── requirements.txt                # API
├── requirements-dev.txt            # Pruebas
├── requirements-docs.txt           # Notebooks y Jupyter Book
└── .github/workflows/ci.yml        # Integración continua
```

## Reproducir el entrenamiento

El entrenamiento se ejecutó en Google Colab Pro (CPU, 8 núcleos, alta RAM) con `src/train_t2.py`. El programa guarda cada resultado en Google Drive y se reanuda si la sesión se interrumpe. Detalles del protocolo y de los topes de tiempo: capítulo «Diseño experimental» del informe.

```python
# En Colab, con la serie en /content/drive/MyDrive/tarea2_series/data/btc_1d.csv
!pip -q install --no-deps timeseries-cv
!python src/train_t2.py              # T2_SMOKE=1 para una prueba corta
```

Los resultados crudos (`results/raw`, `results/preds`, `results/models`) quedan en Drive y no se versionan. Los notebooks leen esos resultados y escriben las tablas CSV en `results/`. Para exportar los modelos a la API:

```bash
python src/export_models.py --results /content/drive/MyDrive/tarea2_series/results --app app
```

## API

Instalación y ejecución local:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell)   |   source .venv/bin/activate (Linux/macOS)
pip install -r requirements.txt
uvicorn app.api:app --reload
```

Documentación interactiva en http://127.0.0.1:8000/docs. Endpoints:

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado y horizontes disponibles |
| GET | `/model/{horizon}` | Lag, arquitectura y máscara de dropout del modelo de ese horizonte |
| POST | `/predict` | Recibe `lags` (precios de cierre en USD, del más antiguo al más reciente) y `horizon` ∈ {7, 14, 21, 28}; devuelve `horizon` valores en USD |

Ejemplo (el número de lags debe coincidir con el lag del modelo de ese horizonte, consultable en `/model/{horizon}`):

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"lags": [87237.13, 88485.49, 87648.22, 87500.0, 87900.0, 88100.0, 88300.0], "horizon": 7}'
```

### Docker

```bash
docker build -t btc-mlp-api .
docker run -p 8000:8000 btc-mlp-api
```

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest
```

Si `app/` aún no contiene los modelos entrenados, las pruebas crean modelos pequeños con la misma interfaz. GitHub Actions ejecuta `flake8` y `pytest` en cada push y pull request.

## Jupyter Book

```bash
pip install -r requirements-docs.txt
jupyter-book build . --all
ghp-import -n -p -f _build/html
```

Los notebooks se ejecutan en Colab y se guardan con sus salidas; el Book no los ejecuta (`execute_notebooks: "off"`).

## Uso de inteligencia artificial

Usamos Claude (Anthropic) como apoyo en programación, análisis de datos y redacción. Los autores revisamos, verificamos y aprobamos todo el contenido, y asumimos la responsabilidad por él.
