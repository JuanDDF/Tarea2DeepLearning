# Conclusiones

## Respuesta a la pregunta del proyecto

Un MLP multisalida con arquitectura buscada por validación cruzada no mejora el pronóstico del precio de cierre diario de BTCUSDT frente al pronóstico ingenuo, que repite el último precio observado. El resultado se repite en las 16 combinaciones de lag y horizonte, con las tres estrategias de validación (48 de 48) y en el hold-out (16 de 16). Con Group K-Fold el RMSE del MLP es entre 1.05 y 1.79 veces el del ingenuo; en el hold-out, entre 1.20 y 4.34 veces. El mejor modelo en el hold-out (lag 14, H 7) obtiene un MAPE de 3.96 %, frente a 2.81 % del ingenuo.

Este resultado era el esperable a partir del análisis exploratorio. El nivel del precio es integrado de orden 1 (ADF y KPSS concuerdan) y su autocorrelación parcial se concentra en el rezago 1: la serie se comporta como una caminata aleatoria, para la cual el último valor es la mejor predicción lineal. El MLP tiene que aprender esa persistencia a partir de entradas estandarizadas, y su aproximación es menos exacta que la regla trivial.

## Arquitectura óptima por lag y horizonte

La arquitectura ganadora cambia con la configuración: no hay una red única que domine. Con lag 7 ganan redes de una capa oculta (32 a 256 neuronas); con lags 14, 21 y 28 dominan las de dos y tres capas. Ninguna de las 16 ganadoras usa dropout. Con una tasa de 0.5, el dropout eleva la mediana del RMSE de validación de 0.159 (sin dropout) a 0.292 (tres capas con dropout) y reduce la fracción de candidatas balanceadas de 82 % a 3.5 %. Como las curvas de entrenamiento y validación sin dropout quedan superpuestas al final de las 30 épocas, el problema no presenta sobreajuste que el dropout deba corregir.

Entre las cinco mejores candidatas de cada configuración las diferencias de validación son pequeñas y su orden no se conserva en el hold-out. En (14, 14) la candidata #2 obtiene 6921 USD en el hold-out frente a 10 711 USD de la #1. La selección por validación es el procedimiento correcto, pero dentro de ese grupo tiene poca capacidad para anticipar el desempeño fuera de muestra.

## Efecto del horizonte de predicción

El error crece con el horizonte en todos los lags. Con Group K-Fold el RMSE medio pasa de 2729 USD (lag 7, H 7) a 4706 USD (lag 7, H 28), y dentro de cada horizonte crece de forma aproximadamente lineal con el paso h. Para un mismo horizonte, el error también crece con el lag: lag 7 es el mejor en validación para los cuatro horizontes. Una ventana de entrada más larga no aporta información útil, coherente con que la dependencia del nivel se concentre en el último rezago.

## Comparación de estrategias de validación

Group K-Fold da el menor RMSE de validación en 14 de las 16 configuraciones y la estimación más estable entre folds, pero también la más optimista. Al intercalar bloques de entrenamiento y validación, el modelo entrena con datos posteriores a la validación y con bloques de nivel de precio similar; en (14, 14) su RMSE de validación es 3724 USD, y el del mismo modelo en el hold-out, 10 711 USD. Forward Chaining es la única estrategia que respeta el orden temporal, pero `timeseries-cv` construye cada fold con una sola ventana de validación, y su estimación es muy variable (en (14, 14) la desviación entre folds es casi igual a la media). K-Fold entrena con datos futuros y valida sobre una ventana, y tiene los dos problemas. Group K-Fold sirve para comparar arquitecturas entre sí; el hold-out es la única estimación honesta del error futuro.

## Diagnóstico de residuos

En el hold-out, BDS y Ljung-Box rechazan la independencia de los residuos de h = 1 en las 16 configuraciones. La causa principal es un sesgo sistemático del modelo final: el residuo medio va de −25 714 a +19 735 USD y en 10 de las 16 configuraciones supera en valor absoluto a la desviación estándar de los residuos. Los residuos están además fuertemente autocorrelacionados (0.85 en el rezago 1 en (14, 14)). Como Ljung-Box ya rechaza, el rechazo de BDS no permite afirmar que quede dependencia no lineal aprovechable. En validación con Group K-Fold el promedio de p-valores BDS supera 0.05 en 12 de 16 configuraciones, pero esa prueba es poco exigente, porque los residuos de un fold no forman una serie consecutiva.

## Desempeño en el hold-out y frente a las referencias

El MLP supera a `MLPRegressor` de scikit-learn en las 16 configuraciones con Group K-Fold, con ventajas de 2 % a 18 % en catorce de ellas. La búsqueda de arquitectura aporta frente a una red por defecto, pero ninguna de las dos redes supera al ingenuo. En el hold-out el error del MLP está dominado por el desplazamiento de la predicción: el periodo contiene el máximo histórico y precios por encima de todo el tramo de entrenamiento, y la red extrapola. Las configuraciones con lag 14 y lag 21 y horizontes de 21 y 28 días superan el 19 % de MAPE.

Para la API exportamos, en cada horizonte, el modelo con menor RMSE de validación de Group K-Fold, que resulta ser el de lag 7 en los cuatro horizontes. En el hold-out lag 7 no es el mejor en ningún horizonte. Lo reportamos sin corregir la elección, porque usar el hold-out para decidir invalidaría la evaluación.

## Limitaciones

- **Cobertura de la búsqueda.** Con el límite de 18 horas, la configuración (14, 14) evaluó las 584 candidatas y las otras 15 configuraciones una muestra aleatoria de 441 o 442 (75.6 %). La ganadora de esas 15 es la mejor de la muestra, no necesariamente del espacio completo.
- **Escalado.** Modelamos log(precio) con una media y una desviación escalares por fold. Esta transformación no elimina la no estacionariedad del nivel, y en el hold-out la red recibe entradas fuera del rango de entrenamiento.
- **Estrategias de `timeseries-cv`.** Group K-Fold es optimista; Forward Chaining y K-Fold validan una ventana por fold. Usamos 5 folds por estrategia con salto 1, seleccionados de forma uniforme cuando la librería genera más.
- **BDS.** Solo es calculable por fold con Group K-Fold, y allí los residuos no son consecutivos.
- **Una sola semilla por candidata y 30 épocas fijas.** No estimamos la variabilidad entre semillas, y la diferencia entre las mejores candidatas es del orden de ese ruido.
- **Hold-out en un régimen particular.** 2025 combina el máximo histórico con la menor volatilidad de la muestra; sus métricas no se extrapolan a periodos más volátiles.

## Trabajo futuro

- Predecir el retorno logarítmico respecto al último precio, o añadir una conexión directa con la última observación, de modo que el pronóstico ingenuo sea el punto de partida y la red solo aprenda la corrección.
- Reentrenar en ventana móvil (walk-forward) para reducir la extrapolación fuera del rango de entrenamiento.
- Validar con Forward Chaining sobre bloques de varias ventanas, para obtener una estimación temporalmente correcta y estable.
- Repetir la selección con varias semillas y reportar intervalos para las métricas.
