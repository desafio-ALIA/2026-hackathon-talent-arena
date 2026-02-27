# Exploración y Fine-Tuning de Prometheus (LLM-as-a-Judge)

Este proyecto contiene scripts y notebooks para explorar y realizar fine-tuning (LoRA) sobre el modelo Prometheus, un modelo diseñado para actuar como juez en la evaluación de otros LLMs.

## Estructura del Proyecto

- `data/`: Contiene los conjuntos de datos. `dataset.json` es el dataset inicial.
- `docs/`: Documentación adicional. Revisa `docs/dataset.md` para entender la estructura de los datos de entrada.
- `src/`: Scripts de utilidad, manejo de datos, modelos y generación de ruidos.
- `notebooks/`: 
    - `01_eda.ipynb`: Análisis exploratorio de datos. ¡Empieza por aquí!
    - `02_finetuning.ipynb`: Script de fine-tuning usando LoRA para entrenar tu modelo Juez.
    - `03_robustness.ipynb`: Script para evaluar la robustez de tu modelo frente a diferentes variaciones en los prompts introducidos por el usuario.
- `output/`: Directorio para guardar modelos entrenados y resultados.

## Instrucciones de Instalación (AWS SageMaker)

1. Abre una terminal en SageMaker.
2. Crea una carpeta llamada `hackathon` y entra en ella:
   ```bash
   mkdir hackathon && cd hackathon
   ```
3. Clona el repositorio del reto:
   ```bash
   git clone https://github.com/desafio-ALIA/2026-hackathon-talent-arena
   ```
4. Mueve la carpeta a tu directorio de SageMaker y entra:
   ```bash
   cd .. && mv hackathon/2026-hackathon-talent-arena /home/ec2-user/SageMaker/
   cd /home/ec2-user/SageMaker/2026-hackathon-talent-arena
   ```
5. Haz una copia del archivo `.env-example` a `.env`:
   ```bash
   cp .env-example .env
   ```
6. Introduce en `.env` tu clave de Hugging Face (opcional).
7. Crea el entorno de conda con las dependencias necesarias:
   ```bash
   conda env create -f conda.yaml
   ```
8. Activa el entorno virtual:
   ```bash
   conda activate sft_hackathon_env
   ```
9. Instala el kernel para poder seleccionarlo en los Jupyter Notebooks:
   ```bash
   python -m ipykernel install --user --name sft_hackathon_alia_env --display-name "Python 3.11 (Hackathon ALIA)"
   ```
10. ¡Importante! Al usar los notebooks, selecciona siempre el kernel **"Python 3.11 (Hackathon ALIA)"**.

## Desarrollo del Reto

Para maximizar tus resultados durante el hackathon, el reto consta de las siguientes fases:

1. **Preparación de Datos**: Sube los datos que se te van a compartir a la carpeta `data/`. Pon el nombre del fichero de datos proporcionado en tu archivo `.env`. *(Se ha dejado un sample inicialmente para que puedas probar)*
2. **Exploración de Datos**: Inspecciona y entiende los datos en `notebooks/01_eda.ipynb`.
3. **Fine-Tuning del Modelo**: Entrena y ajusta el modelo en `notebooks/02_finetuning.ipynb`. (Nota que al final del notebook hay sugerencias para iterar y mejorar el modelo).
4. **Evaluación de la Robustez**: Evalúa la robustez del modelo ajustado en `notebooks/03_robustness.ipynb`. 

### Terminado el reto: Evaluación y Entrega

Una vez concluido el desarrollo:

- Se os pasará un **dataset de test**, del cual deberéis devolver las predicciones del modelo en formato JSON. 
- El JSON de test que se os proporciona tiene las etiquetas `"id"` y `"user_prompt"` por observación. 
- Debéis completarlo entregando un JSON que contenga, además, las siguientes claves por cada observación:
  - `"model_pred"`
  - `"model_reason"`
  - `"model_pred_typos"`
  - `"model_reason_typos"`
- **Criterios de Evaluación**:
  - Se evaluará principalmente el **accuracy** entre el valor real y el valor predicho por el modelo (clasificando si es un texto OK o no de entrada).
  - Se valorará positivamente si `"model_pred"` y `"model_pred_typos"` coinciden, lo cual indicaría que el modelo fine-tuneado es más robusto ante errores.
- Finalmente, **ese fichero resultante se nos compartirá** y nosotros evaluaremos los resultados comparando con las etiquetas reales de cada observación.

### Importante: Modificación de los Prompts
Si decides modificar el `ABSOLUTE_PROMPT` para la evaluación final, ten en cuenta que el template debe contener como mínimo los siguientes placeholders (variables entre llaves) para que el script pueda inyectar la información en cada observación correctamente:
- `{question}`: La última interacción o pregunta del usuario en la conversación.
- `{answer}`: La respuesta generada por el agente de IA a evaluar.
- `{proposed_answer}`: La respuesta real y verificada (ground truth) de referencia.

*(Puedes flexibilizar o añadir más variables en el prompt si tu dataset las incluye, pero esos tres placeholders son los mapeos base)*

## Notas

- El modelo por defecto es `prometheus-eval/prometheus-7b-v2.0`.