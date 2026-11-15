# Diseño experimental

## Datos y hold-out

Descargamos de la API pública de Binance las velas diarias de BTCUSDT entre el 1 de enero de 2018 y el 31 de diciembre de 2025: 2922 días sin fechas faltantes. Usamos el precio de cierre.

Reservamos como hold-out temporal el 10 % final de la serie: 292 días, del 15 de marzo al 31 de diciembre de 2025. Esto supera el mínimo de 2H días para el horizonte más largo (56 días). El hold-out no interviene en la validación cruzada, en la búsqueda de arquitectura ni en el ajuste de los escaladores; solo lo usamos para evaluar los modelos finales.

## Transformación y escalado

El precio pasó de alrededor de 3 mil a más de 100 mil dólares en el periodo. Para estabilizar la escala modelamos log(precio) y, dentro de cada fold, estandarizamos con la media y la desviación del bloque de entrenamiento. Usamos el mismo escalado para las entradas y las salidas, porque ambas son valores de la misma serie.

Las métricas de la búsqueda de arquitectura (RMSE en unidades estandarizadas) sirven para comparar candidatos dentro de una misma configuración. Las métricas de evaluación (MAPE, MAE, RMSE y MSE) se calculan en dólares, después de invertir el escalado y el logaritmo.

## Configuraciones y modelo

Combinamos lags de 7, 14, 21 y 28 días con horizontes H de 7, 14, 21 y 28 días: 16 configuraciones. El modelo es un MLP multisalida en Keras:

- Entrada de tamaño lag.
- Capas ocultas Dense con activación ReLU, cada una seguida opcionalmente de Dropout(0.5).
- Capa de salida Dense(H) lineal.
- Adam, pérdida MSE y métrica RMSE.
- 30 épocas, lotes de 32, sin mezclar el orden temporal (`shuffle=False`).

## Estrategias de validación

Usamos las tres funciones de `timeseries-cv` con `n_steps_jump = 1`. Ninguna entrega un conjunto de prueba, por lo que las métricas de validación cruzada se calculan sobre el bloque de validación de cada fold. Al leer el código fuente de la librería y medir sus salidas encontramos tres propiedades que condicionan el diseño:

- **Group K-Fold** produce siempre 5 folds. En cada uno, una de cada cinco ventanas va a validación y las demás a entrenamiento, intercaladas a lo largo de toda la serie (con lag 28 y H 28: 176 ventanas de entrenamiento y 44 de validación por fold). Las ventanas de validación quedan rodeadas de ventanas de entrenamiento cercanas en el tiempo, lo que hace esta validación optimista frente a un pronóstico real hacia adelante.
- **Forward Chaining** genera un fold por cada posición posible: 2546 folds con lag 28 y H 28. El entrenamiento crece hacia adelante y cada fold valida sobre **una sola ventana** inmediatamente posterior. Usamos 5 folds repartidos de forma uniforme en la segunda mitad de la serie.
- **K-Fold** también valida sobre una sola ventana por fold, pero entrena con ventanas anteriores y posteriores a ella; es decir, usa información del futuro respecto al bloque validado. Usamos 5 folds repartidos de forma uniforme en toda la serie. Además, con `n_steps_jump` mayor que el lag la función no termina y consume toda la memoria; por eso fijamos `n_steps_jump = 1` para todas las estrategias.

Como Forward Chaining y K-Fold validan sobre una ventana por fold, sus métricas por fold son más ruidosas, y el test BDS por fold no es calculable: un solo residuo de h = 1 no permite estimar dependencia serial.

## Búsqueda de arquitectura

El espacio de búsqueda contiene todas las combinaciones con repetición de 32, 64, 128 y 256 neuronas para 1, 2 y 3 capas ocultas (84 topologías). Cada topología se combina con todas sus máscaras de dropout (2 elevado al número de capas), para un total de 584 candidatas por configuración.

Hicimos la búsqueda con Group K-Fold como estrategia principal, por dos razones. Valida sobre decenas de ventanas por fold, lo que da una estimación menos ruidosa que una sola ventana, y entrena sobre bloques pequeños, lo que hace viable evaluar 9344 candidatos en el tiempo disponible. Para cada candidato registramos, promediando las curvas de los 5 folds:

- `best_val_rmse`: mínimo del RMSE de validación sobre las épocas.
- `last_train_rmse`: RMSE de entrenamiento en la última época.
- `gap = |last_train_rmse − best_val_rmse|` y `gap_ratio = gap / best_val_rmse`.
- Las curvas de aprendizaje.

El RMSE de entrenamiento que reporta Keras se calcula con dropout activo y promediado sobre los lotes de cada época, por lo que es una medida aproximada del error de entrenamiento.

**Regla de selección:** entre los candidatos con `gap_ratio < 0.15` elegimos el de menor `best_val_rmse`. Si ninguno cumple el filtro, elegimos el de menor `best_val_rmse` y lo señalamos. La selección usa solo validación; el RMSE en el hold-out de los cinco mejores candidatos se reporta como evidencia final.

La arquitectura ganadora de cada configuración se re-evalúa con Forward Chaining y K-Fold. Después se entrena fold a fold con las tres estrategias para obtener predicciones, métricas por paso de horizonte y residuos. Como referencias usamos `MLPRegressor` de scikit-learn y el pronóstico ingenuo (último precio observado repetido H días).

## Presupuesto de cómputo

Entrenamos en Google Colab Pro con 8 núcleos de CPU, sin GPU: para redes densas tan pequeñas, la GPU no acelera y consume más recursos. Antes de lanzar medimos los tiempos reales. Con 30 épocas, la red más costosa del espacio (tres capas de 256 neuronas con dropout) tardó 5,1 s por fold de Group K-Fold en un solo proceso. Con 8 procesos en paralelo el rendimiento total fue 4,7 veces mayor.

Fijamos un límite estricto de 18 horas de entrenamiento y ordenamos el trabajo por prioridad, de modo que un corte por tiempo dejara siempre resultados completos y utilizables:

| Fase | Contenido |
|---|---|
| F1 | Búsqueda completa en la configuración representativa (lag 14, H 14) |
| F2 | Ganadora de F1 evaluada en las 16 configuraciones con las 3 estrategias |
| F3 | Modelos finales y hold-out con la ganadora de F1 |
| F4 | Búsqueda en las otras 15 configuraciones, intercalada entre configuraciones |
| F5 | Ganadora propia de cada configuración: modelo final, Top-5 en hold-out, re-evaluación |
| F6 | Entrenamiento fold a fold con las 3 estrategias: predicciones, métricas, BDS y referencias |

El programa guarda cada resultado en Google Drive con escritura atómica y, si la sesión se interrumpe, se reanuda saltando lo ya hecho. Tiene tres topes medidos desde el arranque: F4 no inicia candidatos después de 16 h 45 min; ninguna fase inicia tareas nuevas después de 17 h 45 min; y a las 18 h el proceso se detiene. El orden de evaluación de los candidatos dentro de cada configuración es aleatorio con semilla fija, y F4 avanza de forma intercalada entre las 15 configuraciones, para que un corte deje en todas una muestra no sesgada y de tamaño similar. Cada entrenamiento usa una semilla derivada de su identificador, lo que hace reproducible la corrida.

## Ejecución real

El entrenamiento corrió 17 h 12 min entre el 14 y el 15 de noviembre de 2026 y terminó todas las fases sin errores ni entrenamientos truncados. La sesión se reanudó tres veces (dos por ajustes del programa y una para cambiar el orden de F4 a intercalado); una verificación de integridad posterior encontró 0 registros dañados, 0 duplicados y 0 resultados sin valor. El ritmo sostenido fue de 6.4 a 8.8 tareas por minuto (una tarea = un candidato con sus 5 folds), menor que el estimado con las mediciones previas. En consecuencia, F4 alcanzó el tope de 16 h 45 min con 441 o 442 de las 584 candidatas evaluadas en cada una de las 15 configuraciones (75.6 %). La configuración (14, 14) se evaluó completa. F5 y F6 terminaron en 18 minutos, una hora antes del corte.
