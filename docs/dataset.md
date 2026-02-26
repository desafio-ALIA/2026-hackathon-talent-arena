# Documentación del Dataset de Entrada

El dataset proporcionado para los retos del hackathon contiene información detallada sobre interacciones entre usuarios y un modelo de lenguaje, centradas específicamente en escenarios de **seguridad, sesgos y alineación** (Safety & Alignment).

## Formato del Dataset

El dataset se entrega en formato JSON y contiene los siguientes campos principales por registro:

- `iam-id` / `user_id`: Identificadores únicos del usuario que realiza la petición.
- `timestamp`: Fecha y hora de la interacción.
- `message-id`: Identificador único del mensaje.
- `category`: Categoría del riesgo evaluado (ej. Privacidad, Odio, Sesgo de género). Incluye id, icono, nombre y color asociado.
- `challenge`: Descripción de la intención maliciosa o el desafío que el usuario intenta que el modelo incumpla.
- `verdict`: Evaluación previa de la respuesta del modelo (`passed` o `failed`).
  - **passed**: El modelo superó el reto de forma segura.
  - **failed**: El modelo cayó en la trampa (jailbreak) o generó una respuesta insegura.
- `proposed_answer`: Una respuesta sugerida o de referencia. Si el veredicto es `passed`, este campo puede estar vacío, ya que la respuesta generada por el modelo es válida. Si es `failed`, suele contener la respuesta ideal (segura).
- `validation`: Metadatos sobre la validación del contexto.
- `raw`: Un diccionario anidado que contiene toda la conversación en bruto. Destaca especialmente el campo `messages`, que es una lista de diccionarios representando los turnos de la conversación entre el `user` y el `assistant`.

## Preparación para Evaluación (Prometheus)

Para las tareas del hackathon (como el uso de Prometheus LLM-as-a-Judge o el Fine-Tuning), el pipeline provisto extraerá:
1. El último turno de intercambio válido entre el usuario y el asistente de la lista `messages`.
2. Convertirá el `verdict` original (`passed`/`failed`) a formato binario (`1`/`0`).

¡Asegúrate de explorar el notebook `01_eda.ipynb` para ver datos reales en acción!
