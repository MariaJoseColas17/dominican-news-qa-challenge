# Dominican News QA Challenge

Proyecto final de ICC-344-T: fine-tuning de BETO para question answering
extractivo sobre noticias dominicanas, a partir del modelo base provisto por
la cátedra (`Lisibonny/modelo_qa_beto_squad_es_pdqa`).

## Descripción

El sistema recibe una pregunta y un contexto noticioso, y extrae la respuesta
contenida explícitamente en el texto. Se parte del modelo baseline oficial y se
comparan varias configuraciones sobre el split de `validation`, cambiando una
sola variable por experimento, para identificar cuál mejora el desempeño y —
sobre todo — **a qué se debe** esa mejora.

Los experimentos están organizados en dos rondas:

**Primera ronda** — dos variantes de hiperparámetros contra el baseline
publicado: número de épocas y weight decay.

**Segunda ronda** — se agrega un **control** y más experimentos, porque la
primera ronda tenía un problema de atribución: ambas variantes se entrenaron
*sobre* el checkpoint ya afinado, mientras que la fila "Baseline" era ese mismo
modelo sin entrenamiento adicional. Por eso las dos dieron métricas idénticas y
corrigieron el mismo único ejemplo. El control (fine-tuning continuado con los
hiperparámetros del propio baseline) aísla el efecto del entrenamiento extra y
permite saber si el hiperparámetro aportó algo por encima de él.

El notebook está organizado siguiendo los 10 puntos del apartado "Trabajo
solicitado" del enunciado, en ese orden. El bloque inicial es el notebook
introductorio de la cátedra, sin modificar. La tabla de configuraciones está en
la sección [Resultados](#resultados).

Los folds de la Variante 4 se construyen **exclusivamente dentro de `train`**:
`validation` sigue siendo el único split de comparación y selección, y `test` no
se usa en ningún momento para ajustar nada. Los cinco modelos de fold se
promedian en memoria en un único checkpoint, así que en disco no quedan cinco
copias.

## Contenido del repositorio

- `Introduccion_QA_Noticias.ipynb` — notebook ejecutado: evaluación del
  baseline, análisis del dataset y de errores, las dos rondas de experimentos,
  tabla comparativa consolidada, selección del modelo final y generación de
  `submission.csv`.
- `requirements.txt` — dependencias del proyecto.
- `05_evaluate_qa.py`, `07_validate_submission.py` — scripts oficiales de la
  cátedra, incluidos en el repo para que el notebook corra de principio a fin
  sin archivos externos.
- `MODEL_CARD.md` — copia de la model card publicada en Hugging Face.
- `submission.csv` — predicciones finales del modelo elegido sobre `test`.
- `baseline_results.csv`, `control_results.csv` y `variante1..6_results.csv` —
  predicciones y métricas por ejemplo de cada experimento sobre `validation`.
  Pesan unos pocos KB cada uno y los necesita la celda que verifica las métricas
  con `05_evaluate_qa.py`.
- `comparacion_experimentos.csv` — tabla comparativa consolidada.

El notebook produce algunas advertencias (*warnings*) de librerías durante la
ejecución; son informativas y no afectan los resultados ni la reproducibilidad.

## Resultados

Configuraciones evaluadas sobre `validation` (15 preguntas), verificadas con
`05_evaluate_qa.py`:

| Configuración | Qué cambia | EM | F1 | Δ EM vs Control |
|---|---|---:|---:|---:|
| Baseline oficial | sin entrenamiento adicional | 60.00 | 75.84 | −6.67 |
| Control | nada; 4 épocas adicionales | 66.67 | 77.70 | 0.00 |
| Variante 1 | 10 épocas adicionales | 66.67 | 77.70 | 0.00 |
| Variante 2 | `weight_decay` 0.01 → 0.1 | 66.67 | 77.70 | 0.00 |
| Variante 3 | `learning_rate` 2e-05 → 8e-05 | 73.33 | 78.40 | +6.66 |
| Variante 4 | 5 folds promediados en un modelo, `lr=8e-05` | 66.67 | 77.70 | 0.00 |
| **Variante 5** | **Variante 4 + `max_answer_len` 15 → 30** | **73.33** | **80.15** | **+6.66** |
| Variante 6 | Variante 3 + `max_answer_len` 15 → 30 | 66.67 | 74.31 | 0.00 |

**Modelo final: Variante 5** — EM 73.33% y F1 80.15%, los valores más altos en
ambas métricas. Frente al baseline son **+13.33 EM** y **+4.31 F1**; frente al
Control, que es la comparación atribuible, **+6.66 EM** y **+2.45 F1**.

> ⚠️ La Variante 5 comparte pesos con la Variante 4: lo único que las distingue
> es `max_answer_len=30`. Ese valor se guarda en el `config.json` del checkpoint,
> así que viaja con el modelo, pero `transformers` no lo aplica solo — su
> pipeline usa un default de 15. El notebook lo lee del config automáticamente;
> desde fuera hay que leerlo o pasarlo (ver
> [Modelo final publicado](#modelo-final-publicado)). Con 15 el modelo rinde
> 66.67 / 77.70.

### Hallazgos

**1. Solo el learning rate importó entre los hiperparámetros de entrenamiento.**
El Control alcanzó 66.67 / 77.70 entrenando 4 épocas más sin cambiar nada, es
decir exactamente lo mismo que las Variantes 1 y 2. **H1 (épocas) y H2 (weight
decay) quedan refutadas:** su mejora sobre el baseline no era atribuible al
hiperparámetro. Sin el Control habrían parecido confirmadas — es el hallazgo
metodológico central del trabajo.

**2. Rotar folds por sí solo perjudica.** La Variante 4 usa los hiperparámetros
de la Variante 3 y bajó de 73.33 a 66.67: perdió la ganancia del learning rate.
Cada fold ve 36 ejemplos en vez de 45, y con un dataset tan pequeño esa pérdida
pesa más que el beneficio de promediar. **H4 refutada.**

**3. Levantar el tope de longitud no es una mejora universal.** Sobre el modelo
con folds subió el EM a 73.33 y el F1 a 80.15, recuperando un ejemplo cuya
referencia mide 18 tokens y era imposible de emitir con el tope de 15. Sobre el
modelo sin folds (Variante 6) **empeoró**: eligió un span largo en una pregunta
de respuesta corta que ya acertaba. **H5 se confirma solo condicionalmente.**

**4. Los factores interactúan.** El diseño de dos factores da un patrón cruzado
en EM: sin folds 73.33 / 66.67 y con folds 66.67 / 73.33 para
`max_answer_len` 15 / 30. Ninguno ayuda solo de forma consistente, pero la
combinación da el mejor resultado. Con 15 preguntas, cada celda difiere en un
ejemplo, así que es una observación, no un efecto demostrado.

**5. Tres ejemplos fallaron en las ocho configuraciones:** una pregunta
cualitativa frente a un contexto lleno de cifras, una pregunta causal, y una
respuesta compuesta de 27 tokens que enumera tres personas. Los tres requieren
más datos de entrenamiento con esos patrones, no más ajuste de hiperparámetros.

## Cómo correrlo

1. Abrir `Introduccion_QA_Noticias.ipynb` en Google Colab.
2. Activar GPU: `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → `GPU`.
3. Ejecutar todas las celdas en orden (`Entorno de ejecución` → `Reiniciar y ejecutar todo`).

No hace falta subir nada a mano: la celda «Preparación: scripts oficiales»
descarga `05_evaluate_qa.py` y `07_validate_submission.py` de este repositorio si
no están en el directorio de trabajo. Para publicar el modelo sí se necesita un
token de Hugging Face con rol **write**.

La primera celda instala las dependencias dentro del entorno de Colab. El
`requirements.txt` de este repositorio lista el conjunto completo, incluyendo
`scikit-learn` (necesario para los folds de la Variante 4) y `huggingface_hub`
(necesario para publicar el modelo).

## Modelo final publicado

[Majo17/modelo_qa_beto_pdqa_variante1](https://huggingface.co/Majo17/modelo_qa_beto_pdqa_variante1)

La model card completa (hiperparámetros, experimentos, ablación, análisis de
errores y limitaciones) está publicada como `README.md` de ese repositorio en
Hugging Face, con copia en `MODEL_CARD.md`.

El checkpoint lleva `max_answer_len=30` en su `config.json`. Para reproducir las
métricas hay que aplicarlo, leyéndolo del propio config:

```python
from transformers import AutoConfig, pipeline

REPO = "Majo17/modelo_qa_beto_pdqa_variante1"
config = AutoConfig.from_pretrained(REPO)
qa = pipeline("question-answering", model=REPO, tokenizer=REPO,
              max_answer_len=int(config.max_answer_len))
```

## Recursos oficiales de la cátedra

- Dataset: [Lisibonny/pdqa](https://huggingface.co/datasets/Lisibonny/pdqa)
- Modelo baseline: [Lisibonny/modelo_qa_beto_squad_es_pdqa](https://huggingface.co/Lisibonny/modelo_qa_beto_squad_es_pdqa)
- Aplicación de referencia: [Lisibonny/Repartidor_Dominicano](https://huggingface.co/spaces/Lisibonny/Repartidor_Dominicano)

## Integrantes

- María José Colás — 10154787
- María Tiburcio — 10150779
