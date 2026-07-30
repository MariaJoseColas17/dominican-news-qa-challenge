---
language:
- es
pipeline_tag: question-answering
tags:
- extractive-qa
- dominican-news
- spanish
base_model: Lisibonny/modelo_qa_beto_squad_es_pdqa
datasets:
- Lisibonny/pdqa
metrics:
- exact_match
- f1
---

# Modelo QA BETO - PDQA (folds promediados + learning rate alto)

> **Nota sobre el nombre del repositorio.** Se llama
> `modelo_qa_beto_pdqa_variante1` por la primera ronda de experimentos, pero los
> pesos publicados corresponden a la **Variante 5**, que fue la configuración
> seleccionada como modelo final.

> ⚠️ **Este modelo usa `max_answer_len=30`.** El valor está guardado en el
> `config.json` de este repositorio, así que viaja con el checkpoint. Pero
> `transformers` **no lo aplica automáticamente**: su pipeline de
> question-answering usa un default de 15 y solo respeta el valor si se le pasa.
> Con 15 el modelo rinde 66.67 EM / 77.70 F1 en lugar de los 73.33 / 80.15
> reportados. Ver [Ejemplo de uso](#ejemplo-de-uso) para las dos formas de
> cargarlo correctamente.

## Descripción

Modelo desarrollado para el Dominican News Question Answering Challenge de ICC-344-T.
El sistema recibe una pregunta y un contexto noticioso, y extrae una respuesta contenida
explícitamente en el contexto.

## Integrantes

- María José Colás — 10154787
- María Tiburcio — 10150779

## Recursos oficiales

- Dataset: [Lisibonny/pdqa](https://huggingface.co/datasets/Lisibonny/pdqa)
- Baseline: [Lisibonny/modelo_qa_beto_squad_es_pdqa](https://huggingface.co/Lisibonny/modelo_qa_beto_squad_es_pdqa)
- Aplicación: [Lisibonny/Repartidor_Dominicano](https://huggingface.co/spaces/Lisibonny/Repartidor_Dominicano)

## Modelo base

- Repositorio: [Lisibonny/modelo_qa_beto_squad_es_pdqa](https://huggingface.co/Lisibonny/modelo_qa_beto_squad_es_pdqa)
- Justificación: modelo oficial provisto por la cátedra, ya afinado (BETO + SQuAD-es + PDQA). El enunciado del reto exige partir de este modelo, sin entrenar desde cero.
- Número aproximado de parámetros: ~110 millones (arquitectura BERT-base)

## Datos

| División | Filas | Uso |
|---|---:|---|
| train | 45 | Entrenamiento |
| validation | 15 | Comparación y selección |
| test | 20 | Predicciones finales |

No se publicaron ni utilizaron respuestas privadas de test.

## Entrenamiento

| Parámetro | Valor |
|---|---|
| Seed | 42 |
| Learning rate | 8e-05 |
| Batch size | 16 |
| Épocas | 4 |
| Max length | 384 |
| Doc stride | 128 |
| Weight decay | 0.01 |
| Estrategia | 5 folds sobre `train` (`KFold(5, shuffle=True, random_state=42)`), pesos promediados en un único checkpoint |
| Hardware | Google Colab, GPU NVIDIA Tesla T4 |
| Tiempo | ~30 s por fold (5 folds) |

### Parámetros de inferencia

| Parámetro | Valor |
|---|---|
| `max_answer_len` | **30** (guardado en `config.json`; el default de transformers es 15) |
| `max_seq_len` | 384 |

`max_answer_len=30` es parte de la definición de este modelo, no una sugerencia:
sin él las métricas reportadas no se reproducen. Está escrito en el `config.json`
del checkpoint para que quede versionado y visible, pero el pipeline de
`transformers` no lo lee de ahí por su cuenta — hay que pasárselo, o leerlo del
config como muestra el ejemplo de uso.

## Experimentos

Todos evaluados sobre `validation` (15 preguntas) y verificados con
`05_evaluate_qa.py`.

| Experimento | Cambio | EM | F1 |
|---|---|---:|---:|
| Baseline oficial | Sin entrenamiento adicional | 60.00 | 75.84 |
| Control | 4 épocas adicionales, hiperparámetros del baseline | 66.67 | 77.70 |
| Variante 1 | 10 épocas adicionales | 66.67 | 77.70 |
| Variante 2 | Weight decay 0.01 → 0.1 | 66.67 | 77.70 |
| Variante 3 | Learning rate 2e-05 → 8e-05 | 73.33 | 78.40 |
| Variante 4 | 5 folds promediados, lr 8e-05 | 66.67 | 77.70 |
| **Variante 5** | **Variante 4 + max_answer_len 30** | **73.33** | **80.15** |
| Variante 6 | Variante 3 + max_answer_len 30 | 66.67 | 74.31 |
| **Modelo final** | **Variante 5** | **73.33** | **80.15** |

## Ablación o comparación controlada

Todas las variantes parten del checkpoint ya afinado
`Lisibonny/modelo_qa_beto_squad_es_pdqa` y aplican **fine-tuning continuado**
sobre él. Es importante ser preciso al respecto: la fila "Baseline oficial" es
ese mismo checkpoint **sin entrenamiento adicional**, así que la comparación
directa contra el baseline mezcla dos efectos (entrenar más, y cambiar el
hiperparámetro).

Por eso se incluye un **Control**: fine-tuning continuado con exactamente los
hiperparámetros del baseline (`learning_rate=2e-05`, `epochs=4`,
`batch_size=16`, `weight_decay=0.01`, `seed=42`). El control aísla el efecto del
entrenamiento adicional, y cada variante se compara contra él cambiando una sola
variable:

- **Control:** 4 épocas adicionales, sin cambiar ningún hiperparámetro.
- **Variante 1:** 10 épocas adicionales en lugar de 4.
- **Variante 2:** `weight_decay` 0.01 → 0.1, manteniendo 4 épocas.
- **Variante 3:** `learning_rate` 2e-05 → 8e-05, manteniendo 4 épocas.
- **Variante 4:** rotación de 5 folds **dentro de `train`** con los
  hiperparámetros de la Variante 3, promediando los pesos de los cinco modelos
  en un único checkpoint. Aísla el efecto de entrenar sobre una vista única de
  45 ejemplos frente a rotar particiones.
- **Variantes 5 y 6:** `max_answer_len` 15 → 30 sobre los pesos de las Variantes
  4 y 3 respectivamente. **No reentrenan**: al reutilizar pesos idénticos, la
  única diferencia es el parámetro de decodificación, lo que hace la comparación
  perfectamente controlada. Responden a que dos referencias de `validation`
  miden 18 y 27 tokens y son inalcanzables con el tope por defecto de 15.

Ninguna variante usó `test` para entrenar, ajustar o seleccionar, y no se creó
ninguna división nueva para reportar resultados.

**Modelo final elegido: Variante 5.**

**Justificación.** Es la única configuración que obtiene el EM y el F1 más altos
a la vez. El Control demuestra que llegar a 66.67 se consigue solo entrenando 4
épocas más sin cambiar ningún hiperparámetro, así que las Variantes 1, 2 y 4 no
aportan nada por encima de él; la Variante 5 le saca +6.66 EM y +2.45 F1.

Contra la Variante 3 el EM empata (11/15 ambas) pero aciertan preguntas
distintas, y el desempate es el F1, donde la Variante 5 está 1.75 puntos arriba.

La ganancia sobre la Variante 4 es perfectamente atribuible: comparten los mismos
pesos y solo cambia `max_answer_len`. Una sola predicción cambió, y es el ejemplo
cuya referencia mide 18 tokens — imposible de emitir con el tope de 15.

**Hipótesis refutadas.** H1 (épocas) y H2 (weight decay) no superaron al Control.
H4 (folds por sí solos) empeoró respecto a la Variante 3: cada fold ve 36
ejemplos en vez de 45, y con un dataset tan pequeño esa pérdida pesa más que el
beneficio de promediar. H5 (`max_answer_len`) solo se confirma **en combinación
con los folds**: aplicada al modelo sin folds (Variante 6) empeoró el resultado.

## Resultados

- Exact Match en validation: **73.33%** (baseline: 60.00%)
- F1 en validation: **80.15%** (baseline: 75.84%)
- Mejora atribuible frente al Control: **+6.66 EM**, **+2.45 F1**
- Aciertos exactos: **11 de 15** (baseline: 9 de 15)
- Script de evaluación: `05_evaluate_qa.py`

## Ejemplo de uso

**Opción recomendada: leer el valor del propio config.** Así no hay ningún número
que recordar y el modelo queda autocontenido.

```python
from transformers import AutoConfig, pipeline

REPO = "Majo17/modelo_qa_beto_pdqa_variante1"

config = AutoConfig.from_pretrained(REPO)
opciones = {}
if getattr(config, "max_answer_len", None):
    opciones["max_answer_len"] = int(config.max_answer_len)

qa = pipeline("question-answering", model=REPO, tokenizer=REPO, **opciones)

qa(
    question="¿Quién hizo el anuncio?",
    context="La ministra informó que el programa comenzará en agosto.",
)
```

**Opción corta, pasando el valor a mano:**

```python
from transformers import pipeline

qa = pipeline(
    "question-answering",
    model="Majo17/modelo_qa_beto_pdqa_variante1",
    tokenizer="Majo17/modelo_qa_beto_pdqa_variante1",
    max_answer_len=30,   # sin esto el pipeline usa 15 y las métricas no se reproducen
)

qa(
    question="¿Quién hizo el anuncio?",
    context="La ministra informó que el programa comenzará en agosto.",
)
```

## Análisis de errores

1. **Truncamiento de frontera.** El modelo ubica la región correcta pero corta el
   span de más o de menos. Ejemplo: `11 bateadores` en vez de `a 11 bateadores`.
   Este patrón se corrige con cualquier entrenamiento adicional.
2. **Truncamiento por el tope del pipeline.** Dos referencias de `validation`
   miden 18 y 27 tokens, por encima del `max_answer_len=15` por defecto: eran
   inalcanzables por configuración, no por limitación del modelo. Subir el tope a
   30 recuperó la de 18 tokens.
3. **Sobreextensión al levantar el tope.** El efecto tiene un costo: aplicado al
   modelo sin folds, el tope de 30 hizo que eligiera un span largo en
   `¿Quién fundó Menudo?` —una pregunta de respuesta corta que ya acertaba— y
   perdió el acierto.
4. **Preguntas causales o cualitativas.** Ante `¿Cómo será la inflación en 2023?`
   el modelo extrae una cifra en vez de la descripción cualitativa esperada.
   Ninguna configuración lo corrigió.
5. **Respuestas compuestas.** Cuando la referencia enumera tres personas con sus
   cargos (27 tokens), el modelo extrae una sola entidad. Sigue fallando incluso
   con el tope levantado a 30, así que ahí el límite no era la longitud.

## Limitaciones

- La respuesta debe estar presente en el contexto.
- El modelo puede confundir entidades, fechas o cantidades similares.
- El modelo no verifica la veracidad de la noticia.
- No debe utilizarse como única fuente para decisiones de alto impacto.
- El dataset de entrenamiento es muy pequeño (45 ejemplos), lo que limita la
  capacidad del modelo para generalizar a preguntas ambiguas o con múltiples
  respuestas candidatas en el contexto.
- El `max_answer_len` del pipeline impone un tope al largo de la respuesta que
  el modelo puede devolver. Con el valor por defecto (15 tokens), dos
  referencias de `validation` de 18 y 27 tokens son inalcanzables por
  configuración, no por limitación del modelo.

## Reproducibilidad

- Repositorio de código: https://github.com/MariaJoseColas17/dominican-news-qa-challenge
- Commit: [completar con el hash del commit final: `git rev-parse --short HEAD`]
- Notebook: `Introduccion_QA_Noticias.ipynb`
- Archivo de dependencias: `requirements.txt`
- Scripts de evaluación y validación: `05_evaluate_qa.py`, `07_validate_submission.py`
- Semillas: 42 (`random`, `numpy`, `torch`, `TrainingArguments.seed` y
  `KFold(random_state=42)`)