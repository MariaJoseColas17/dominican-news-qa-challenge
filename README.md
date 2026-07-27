# Dominican News QA Challenge

Proyecto final de ICC-344-T: fine-tuning de BETO para question answering
extractivo sobre noticias dominicanas, a partir del modelo base provisto por
la cátedra (`Lisibonny/modelo_qa_beto_squad_es_pdqa`).

## Descripción

El sistema recibe una pregunta y un contexto noticioso, y extrae la respuesta
contenida explícitamente en el texto. Se parte del modelo baseline oficial y
se comparan dos variantes de fine-tuning, cada una modificando un único
hiperparámetro respecto al baseline, para identificar la configuración que
mejor mejora el desempeño sobre el split de `validation`.

## Contenido del repositorio

- `03_Introduccion_QA_Noticias.ipynb` — notebook ejecutado con: evaluación
  del baseline, análisis de errores, entrenamiento y evaluación de 2
  variantes controladas, selección del modelo final, y generación de
  `submission.csv`.
- `09_requirements.txt` — dependencias del proyecto.
- `submission.csv` — predicciones finales del modelo elegido sobre el split
  de `test`.
- `baseline_results.csv` — predicciones y métricas detalladas del baseline
  sobre `validation`.
- `variante1_results.csv` — predicciones y métricas detalladas de la
  Variante 1 (aumento de épocas).
- `variante2_results.csv` — predicciones y métricas detalladas de la
  Variante 2 (aumento de weight decay).

## Resultados

| Configuración | Exact Match | F1 |
|---|---:|---:|
| Baseline oficial | 60.00% | 75.84% |
| Variante 1 (épocas 4→10) | 66.67% | 77.70% |
| Variante 2 (weight decay 0.01→0.1) | 66.67% | 77.70% |
| **Modelo final (Variante 1)** | **66.67%** | **77.70%** |

## Cómo correrlo

1. Abrir `03_Introduccion_QA_Noticias.ipynb` en Google Colab.
2. Activar GPU: `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → `GPU`.
3. Ejecutar todas las celdas en orden (`Entorno de ejecución` → `Reiniciar y ejecutar todo`).
4. El notebook instala automáticamente las dependencias listadas en `09_requirements.txt`.

## Modelo final publicado

[Majo17/modelo_qa_beto_pdqa_variante1](https://huggingface.co/Majo17/modelo_qa_beto_pdqa_variante1)

La model card completa (hiperparámetros, análisis de errores, limitaciones)
está publicada como `README.md` de ese repositorio en Hugging Face.

## Recursos oficiales de la cátedra

- Dataset: [Lisibonny/pdqa](https://huggingface.co/datasets/Lisibonny/pdqa)
- Modelo baseline: [Lisibonny/modelo_qa_beto_squad_es_pdqa](https://huggingface.co/Lisibonny/modelo_qa_beto_squad_es_pdqa)
- Aplicación de referencia: [Lisibonny/Repartidor_Dominicano](https://huggingface.co/spaces/Lisibonny/Repartidor_Dominicano)

## Integrantes

- María José Colás — 10154787
- María Tiburcio — 10150779
