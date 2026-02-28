# 📊 Documentación del Dataset de Entrada

Este documento detalla la estructura y el contenido de los datasets utilizados en el hackathon. Los datos se centran en interacciones de **Seguridad y Alineación (Safety & Alignment)**, donde se pone a prueba la capacidad de los modelos para resistir comportamientos maliciosos.

---

## 📂 Formato del Dataset (JSON)

Cada registro en el dataset representa una interacción individual y contiene los siguientes campos, agrupados por su propósito:

### 1. Metadatos de la Interacción
- `iam-id` / `user_id`: Identificadores únicos del usuario.
- `timestamp`: Fecha y hora de la interacción.
- `message-id`: Identificador único del mensaje (clave para el submission).
- `category`: Información sobre el riesgo evaluado (Privacidad, Odio, etc.). Contiene:
    - `id`, `name`, `icon`, `color`.

### 2. Datos del Reto (Input/Output)
- `challenge`: Descripción de la intención maliciosa o el objetivo del usuario.
- `conversation`: Historial completo de la interacción entre el usuario y la IA.
- `question`: El mensaje específico del usuario que debe ser evaluado por el Juez.
- `last_interaction`: La respuesta generada por la IA que el Juez debe calificar.
- `verdict`: Evaluación binaria de la interacción:
    - **`passed`**: El modelo respondió de forma segura.
    - **`failed`**: El modelo generó contenido inapropiado o sucumbió al jailbreak.
- `correctedResponse`: La respuesta ideal o "Ground Truth" que el modelo debería haber dado.

### 3. Validación Adicional (LLM-as-a-Judge)
Estos campos son el resultado de un proceso de validación pre-procesado por otros modelos:

- **`validation.context`**:
    - `val_context_bool`: ¿Es la conversación pertinente al reto propuesto?
    - `val_context_reasoning`: Explicación de la relevancia.
- **`validation.goal`**:
    - `val_goal_bool`: ¿Logró el usuario su objetivo de jailbreak?
    - `val_goal_reasoning`: Justificación del éxito o fallo del ataque.
- **`validation.answer`**:
    - `val_proposed_answer_bool`: ¿Es segura la respuesta generada/corregida?
    - `val_proposed_answer_correct_gen`: Respuesta final validada por el sistema.
- **`val_stop_reason`**: Indica por qué se detuvo el flujo de validación:
    - `Context Invalid`, `Goal Not Achieved`, `Proposed Answer Invalid`.

---

## 📤 Formato de Entrega (Submission)

El script `04_submission.ipynb` generará automáticamente un archivo `submission.json`. Tu modelo debe predecir el veredicto para cada registro del dataset de test ciego.

### Estructura del Objeto de Salida
Cada entrada en el JSON de entrega debe incluir:

| Campo | Tipo | Descripción |
| :--- | :--- | :--- |
| `message-id` | String | Identificador único del caso. |
| `po_m_pred` | Int (`0`\|`1`) | Predicción de tu modelo (Original). `1` = passed, `0` = failed. |
| `po_m_reason` | String | Explicación/Razonamiento de tu modelo para `po_m_pred`. |
| `pt_m_pred` | Int (`0`\|`1`) | Predicción ante variaciones de ruido (Typos/Robustez). |
| `pt_m_reason` | String | Razonamiento para la variante con ruido. |
| `pg_m_pred` | Int (`0`\|`1`) | Predicción ante variaciones gramaticales. |
| `pg_m_reason` | String | Razonamiento para la variante gramatical. |

---

> [!TIP]
> Puedes explorar estos datos de forma interactiva en el notebook `01_eda.ipynb` para entender mejor las distribuciones y los casos de borde.