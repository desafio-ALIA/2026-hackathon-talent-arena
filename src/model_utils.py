import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from peft import PeftModel


def get_model_and_tokenizer(model_name="prometheus-eval/prometheus-7b-v2.0" ):
    """
    Carga el modelo Prometheus y su tokenizador asociado desde Hugging Face.
    
    Esta función es esencial para el hackathon ya que inicializa el evaluador LLM-as-a-Judge.
    Recuerda configurar tu token de Hugging Face de antemano.
    
    Args:
        model_name (str): La versión específica del modelo de Prometheus a cargar.
        
    Returns:
        model, tokenizer: Tupla con el modelo y el tokenizador listos para realizar inferencias.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables.")
        
    print(f"Loading model: {model_name}...")
    
    # Placeholder for actual loading logic to prevent large downloads during setup
    # In a real run, you would uncomment the following:
    tokenizer = None
    model = None
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
         model_name,
         token=hf_token,
         device_map="auto"
    )
    # Mover fuera para que sea eficiente
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    
    return model, tokenizer



    
def split_model_reason_result(sample):
    """
    Post-procesa la salida del modelo para separar la explicación de la puntuación.
    
    Busca la etiqueta '[RESULT]' para dividir el texto. Si no la encuentra, 
    asume que todo el texto es el razonamiento y devuelve un resultado nulo.

    Args:
        sample (dict): Ejemplo que contiene 'model_output'.

    Returns:
        dict: Diccionario con las claves 'reason' (explicación) y 'result' (puntuación limpia).
    """
    output = sample.get("model_output", "")
    
    if "[RESULT]" in output:
        # Dividimos por la última aparición del tag para evitar errores
        parts = output.rsplit("[RESULT]", 1)
        reason = parts[0].strip()
        result_raw = parts[1].strip()
        
        # Limpieza mediante regex para capturar solo el dígito (evita puntos finales, etc.)
        score_match = re.search(r'(\d+)', result_raw)
        result = score_match.group(1) if score_match else result_raw
    else:
        reason = output.strip()
        result = None
    
    return {
        "reason": reason,
        "model_pred": result
    }



def model_predict_batched(model, tokenizer, batch, input_col = "user_content"):
    """
    Realiza inferencia en lotes (batch) utilizando el modelo cargado en GPU.
    
    Gestiona automáticamente el padding a la izquierda, desactiva el cálculo de gradientes
    para ahorrar memoria VRAM y extrae únicamente la respuesta generada por el modelo.

    Args:
        batch (dict): Un lote del dataset que contiene una lista bajo la clave input_col.

    Returns:
        dict: Diccionario con 'model_output', que contiene la lista de críticas generadas.
    """
    # Detección dinámica del dispositivo del modelo
    model_device = next(model.parameters()).device
    
    # Preparación de mensajes para la plantilla de chat
    messages_list = [[{"role": "user", "content": p}] for p in batch[input_col]]

    # Configuración de padding segura
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    # Tokenización masiva
    inputs = tokenizer.apply_chat_template(
        messages_list,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True
    ).to(model_device)

    # Inferencia optimizada
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1000, 
            do_sample=True,
            temperature=0.1, # Temperatura baja para mayor consistencia en la evaluación
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extracción exclusiva de la respuesta (ignorando el prompt inicial)
    input_length = inputs["input_ids"].shape[1]
    decoded_outputs = tokenizer.batch_decode(
        generated_ids[:, input_length:], 
        skip_special_tokens=True
    )
    
    return {"model_output": decoded_outputs}




def load_lora_model(model_name, model_path):
    """
    Carga un modelo base y le aplica los pesos ajustados de un entrenamiento LoRA (PEFT).
    
    Durante el hackathon, usarás esta función para cargar tu propio modelo afinao (Fine-Tuned)
    y comparar sus evaluaciones con las del modelo original.
    
    Args:
        model_name (str): Nombre o ruta del modelo base original (p. ej., "prometheus-eval/prometheus-7b-v2.0").
        model_path (str): Ruta donde se encuentran guardados los adaptadores LoRA entrenados.
        
    Returns:
        model, tokenizer: Tupla con el modelo ajustado y su tokenizador.
    """

    # 1. Load the original BASE model (the one you started with)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # 2. Load the Tokenizer (now that you've saved it to the FT path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 3. Load the LoRA adapters onto the base model
    model = PeftModel.from_pretrained(base_model, model_path)
    return model, tokenizer