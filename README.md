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

## Configuración

1.  **Entorno Virtual**:
    Asegúrate de tener Python instalado. Crea y activa un entorno virtual:
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

2.  **Instalar Dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Variables de Entorno (IMPORTANTE)**:
    Para descargar el modelo y usarlo, necesitas una cuenta de [Hugging Face](https://huggingface.co/) y generar un token de acceso (HF Token).
    - Crea una cuenta gratis si no la tienes.
    - Ve a tus Preferencias / Settings -> Access Tokens.
    - Crea un nuevo token (Read/Write) y cópialo.
    - Copia el archivo `.env-example` a `.env` y añade tu token de Hugging Face:
    ```bash
    copy .env-example .env
    ```
    Edita `.env` con tu `HF_TOKEN`.

## Pasos para el Hackathon

Para maximizar tus resultados durante el hackathon, te recomendamos seguir este flujo:

1.  **Exploración de Datos**: Abre `notebooks/01_eda.ipynb` para entender los datos con los que estás trabajando (revisa `docs/dataset.md`).
2.  **Fine-Tuning y Evaluación Base**: Abre `notebooks/02_finetuning.ipynb`. Aquí podrás correr el modelo base, ver si aprueba en sus evaluaciones y luego afinar el modelo con LoRA para ajustar sus criterios a la rúbrica de Safety que se pide.
3.  **Evaluación de Robustez**: Abre `notebooks/03_robustness.ipynb`. Corrompe los prompts introducidos (simulando errores de usuario o gramaticales) para evaluar si tu modelo fino (Fine-Tuned) sigue siendo igual de preciso a la hora de juzgar interacciones.

## Notas

- El modelo por defecto es `prometheus-eval/prometheus-7b-v2.0` (o similar).
- El script de carga del modelo en `src/model_utils.py` tiene la descarga comentada para evitar descargas masivas accidentales durante la configuración. Descoméntalo cuando estés listo.
