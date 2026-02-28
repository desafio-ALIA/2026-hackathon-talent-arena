# ⚔️ Prometheus LLM-as-a-Judge: Hackathon Talent Arena

Bienvenido al repositorio oficial para el reto de **Fine-Tuning de Prometheus**. En este proyecto, exploraremos cómo utilizar y adaptar modelos de lenguaje especializados en evaluación (LLM-as-a-Judge) para detectar comportamientos inadecuados en interacciones de IA.

---

## 🎯 Objetivo del Reto

El propósito principal es entrenar un **modelo Juez** capaz de discernir de manera precisa si una interacción entre un usuario y un modelo de lenguaje es **adecuada** o **inadecuada**.

### El Contexto
Los datos provienen de un reto previo donde usuarios intentaron "romper" modelos de lenguaje (Jailbreaking). 
- **Verdict "passed":** El modelo resistió el ataque y respondió de forma segura.
- **Verdict "failed":** El usuario logró que el modelo generara una respuesta inadecuada.

Tu misión es ajustar **Prometheus-7b-v2.0** para que actúe como un evaluador crítico y robusto, capaz de identificar estos fallos incluso ante variaciones en el input. Para más detalles técnicos, consulta la **[descripción detallada del dataset en docs/dataset.md](docs/dataset.md)**.

---

## 🏗️ Estructura del Proyecto

```text
├── data/           # Datasets de entrenamiento y evaluación.
├── docs/           # Documentación detallada. Revisa dataset.md para entender el formato.
├── notebooks/      # Pipeline completo del reto:
│   ├── 01_eda.ipynb          # Análisis Exploratorio (¡Empieza aquí!)
│   ├── 02_finetuning.ipynb   # Entrenamiento con LoRA/QLoRA.
│   ├── 03_robustness.ipynb   # Pruebas de resistencia ante ruido/typos.
│   └── 04_submission.ipynb   # Generación de resultados finales.
├── src/            # Código fuente (Utilidades, métricas, preprocesamiento).
└── output/         # Modelos guardados y predicciones.
```

---

## ⚙️ Configuración del Entorno (AWS SageMaker)

Sigue estos pasos para preparar tu entorno de trabajo de manera eficiente:

1. **Preparar el espacio**:
   ```bash
   mkdir hackathon && cd hackathon
   git clone https://github.com/desafio-ALIA/2026-hackathon-talent-arena
   ```

2. **Organizar directorios**:
   ```bash
   cd .. && mv hackathon/2026-hackathon-talent-arena /home/ec2-user/SageMaker/
   cd /home/ec2-user/SageMaker/2026-hackathon-talent-arena
   ```

3. **Variables de Entorno**:
   Copia el archivo de ejemplo y configura tu `HUGGINGFACE_TOKEN` si deseas acelerar la descarga de modelos.
   ```bash
   cp .env-example .env
   ```

4. **Entorno Virtual**:
   Creamos y activamos un entorno Conda optimizado para el reto.
   ```bash
   conda env create -f conda.yaml
   conda activate sft_hackathon_env
   python -m ipykernel install --user --name sft_hackathon_alia_env --display-name "Python 3.11 (Hackathon ALIA)"
   ```

> [!IMPORTANT]
> Asegúrate de seleccionar el kernel **"Python 3.11 (Hackathon ALIA)"** al abrir cualquier Notebook.

---

## 🚀 Pipeline del Hackathon

Para alcanzar la máxima puntuación, te recomendamos seguir este flujo de trabajo:

1. **Análisis (EDA)**: Comprende las categorías de riesgo y la distribución de los veredictos en [01_eda.ipynb](notebooks/01_eda.ipynb).
2. **Entrenamiento**: Ajusta los hiperparámetros de LoRA para mejorar la precisión en [02_finetuning.ipynb](notebooks/02_finetuning.ipynb).
3. **Robustez**: Evalúa la resistencia de tu modelo ante ruido y errores tipográficos en [03_robustness.ipynb](notebooks/03_robustness.ipynb). En este paso utilizaremos la herramienta **promptNoisES** para generar variaciones automáticas y medir su impacto.
4. **Entrega**: Genera el archivo final de predicciones en [04_submission.ipynb](notebooks/04_submission.ipynb).

---

## 📊 Evaluación y Entrega

### Criterios de Puntuación
| Criterio | Descripción |
| :--- | :--- |
| **Accuracy** | Precisión global comparada con el Ground Truth. |
| **Robustez** | Consistencia de la predicción ante variaciones del texto (`model_pred` vs `model_pred_typos`). |
| **Razonamiento** | Calidad del feedback generado por el modelo para justificar su veredicto. |

### Proceso de Entrega
Al finalizar el tiempo, se proporcionará un **Dataset de Test**. Deberás procesarlo con tu modelo y entregar un archivo `submission.json` con el formato especificado en los notebooks.

---

## 📝 Notas Adicionales
- Modelo Base: `prometheus-eval/prometheus-7b-v2.0`
- **Libertad Total**: Tienes permiso para modificar cualquier parte del código, añadir librerías o cambiar la lógica de preprocesamiento para mejorar tus resultados.

---

## 💡 Consejos para el Éxito (Iteraciones)

¡Felicidades por llegar hasta aquí! El modelo base es un gran punto de partida, pero para destacar en el hackathon te sugerimos explorar estas vías:

### 1. Perfecciona el Prompt (Prompt Engineering)
El Juez es tan bueno como su rúbrica.
- **Rúbricas Explícitas**: Revisa `src/prompts.py`. Asegúrate de que Prometheus entienda exactamente qué constituye un fallo de seguridad.
- **Historial de Conversación**: Evaluar solo el último mensaje puede ser insuficiente. Prueba a incluir el contexto previo para detectar ataques multiturno.
- **Few-Shot Prompting**: Provee ejemplos de respuestas ideales (*Reference Answers*) dentro del prompt para guiar el juicio del modelo.

### 2. Calidad sobre Cantidad (Curación de Datos)
- **Filtrado de Ruido**: Elimina ejemplos ambiguos del entrenamiento. Un modelo aprende mejor de pocos ejemplos claros que de muchos confusos.
- **Balanceo**: Mantén un equilibrio entre casos *passed* y *failed* para evitar sesgos en el Juez.
- **Data Augmentation**: Utiliza las sugerencias de respuestas corregidas para generar nuevos pares de entrenamiento.

### 3. Ajuste de Hiperparámetros
- **Estrategia LoRA**: Prueba a aumentar el rango (`r`) a 32 o 64 para capturar matices más complejos.
- **Learning Rate**: Ajusta la tasa de aprendizaje (ej. `5e-5`). Si la pérdida fluctúa mucho, redúcela.
- **Épocas**: Experimenta con 2-3 épocas, vigilando siempre que el modelo no empiece a memorizar los datos (overfitting).

---

## 📚 Recursos y Referencias

### Prometheus & LLM-as-a-Judge
- [Prometheus 2: Open-Source Language Models for Evaluation](https://github.com/prometheus-eval/prometheus-eval)
- [Haystack Cookbook: Prometheus 2 Evaluation](https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/prometheus2_evaluation.ipynb)

### Tutoriales de Fine-Tuning
- [Mistral 7B Fine-Tuning Tutorial (DataCamp)](https://www.datacamp.com/es/tutorial/mistral-7b-tutorial)
- [Fine-Tuning Mistral con LoRA (Brev.dev)](https://colab.research.google.com/github/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb)

---

## 📄 Licencia

Este proyecto se distribuye bajo la **Licencia MIT**. Consulta el archivo [LICENSE](LICENSE) para más información.

---
*Organizado por ALIA - Talent Arena 2026*