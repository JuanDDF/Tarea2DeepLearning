# Pronóstico del precio de Bitcoin con MLP multisalida

**Proyecto Integrador de Deep Learning · Universidad del Norte · Noviembre de 2026**

Anyely Sanjuan Benavides y Juan David Díaz Fernández · Profesor: Dr. Lihki Rubio

```{figure} images/criptomonedas.jpg
:name: fig-criptomonedas
:width: 80%
:alt: Monedas físicas de Ethereum, Ripple y Bitcoin sobre fondo oscuro

Monedas conmemorativas de Ethereum, Ripple y Bitcoin. Imagen tomada del blog de Economía y Empresa de la Universitat Oberta de Catalunya (2021), [fuente](https://blogs.uoc.edu/economia-empresa/wp-content/uploads/sites/152/2021/11/criptomonedes.jpeg).
```

## Descripción del proyecto

Pronosticamos el precio de cierre diario del par BTCUSDT a 7, 14, 21 y 28 días con un perceptrón multicapa (MLP) multisalida, que entrega en una sola pasada los H valores del horizonte. Combinamos cuatro tamaños de ventana de entrada (lags de 7, 14, 21 y 28 días) con los cuatro horizontes, para un total de 16 configuraciones.

Para cada configuración buscamos la mejor arquitectura en un espacio de 584 candidatas (de una a tres capas ocultas, con 32, 64, 128 o 256 neuronas y Dropout(0.5) activable por capa). Seleccionamos con un criterio que combina el error de validación y el control del sobreajuste. Validamos con las tres estrategias de partición temporal de la librería `timeseries-cv` (Group K-Fold, Forward Chaining y K-Fold) y reportamos MAPE, MAE, RMSE y MSE por paso de horizonte. Diagnosticamos los residuos con el test BDS y evaluamos el modelo final en un hold-out temporal de 292 días.

El mejor modelo de cada horizonte se sirve mediante una API (FastAPI) empaquetada en Docker, con pruebas unitarias e integración continua en GitHub Actions.

## Contenido

```{tableofcontents}
```

## Uso de inteligencia artificial

Usamos Claude (Anthropic) como apoyo en programación, análisis de datos y redacción. Los autores revisamos, verificamos y aprobamos todo el contenido, y asumimos la responsabilidad por él.

## Referencias

- Binance. (2026). *Binance Spot API: Kline/Candlestick data*. https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- Broock, W. A., Scheinkman, J. A., Dechert, W. D., & LeBaron, B. (1996). A test for independence based on the correlation dimension. *Econometric Reviews, 15*(3), 197–235.
- Lopes, D. R. (2020). *timeseries-cv: Time series cross-validation* [Software]. https://github.com/DidierRLopes/timeseries-cv
