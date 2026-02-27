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
    
     # 1. Cargar y configurar el Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    # Configuramos el pad_token si no existe (común en Mistral/Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Padding a la izquierda es obligatorio para modelos decodificadores (CausalLM) 
    # cuando se hace inferencia en batches
    tokenizer.padding_side = "left" 
    
    # 2. Cargar el Modelo
    model = AutoModelForCausalLM.from_pretrained(
         model_name,
         token=hf_token,
         device_map="auto",
         dtype=torch.float16, # Media precisión para ganar velocidad y ahorrar VRAM
         low_cpu_mem_usage=True
    )
    
    
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



def model_predict(model, tokenizer, prompt, max_new_tokens =200, temperature=0.7):
    """
    Realiza una inferencia simple para un único prompt utilizando el modelo y tokenizador proporcionados.

    Esta función prepara el texto, lo envía al dispositivo donde reside el modelo (GPU/CPU) 
    y genera una respuesta de forma determinista. Es ideal para pruebas rápidas o 
    validaciones unitarias durante la hackathon.

    Args:
        model (transformers.PreTrainedModel): El modelo de lenguaje ya cargado.
        tokenizer (transformers.PreTrainedTokenizer): El tokenizador correspondiente al modelo.
        prompt (str): El texto de entrada o instrucción para el modelo.

    Returns:
        str: El texto generado por el modelo, limpio de tokens especiales y del prompt original.
    """
    # 1. Identificar el dispositivo del modelo (soporta device_map="auto")
    device = model.device 
    
    # 2. Tokenizar y mover tensores al dispositivo correcto
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 3. Generación determinista (do_sample=False para evitar variabilidad en pruebas)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )

    # 4. Decodificar solo la parte nueva (ignorando los tokens del prompt)
    input_length = inputs["input_ids"].shape[1]
    prediction = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return prediction.strip()



def model_predict_batched(model, tokenizer, batch, input_col = "user_content", temperature = 0.1, max_new_tokens = 1000):
    # 1. Detectamos el dispositivo de entrada (donde está la primera capa)
    model_device = model.device 
    
    messages_list = [[{"role": "user", "content": p}] for p in batch[input_col]]


    # 2. IMPORTANTE: Pedimos que devuelva un diccionario completo (return_dict=True)
    inputs = tokenizer.apply_chat_template(
        messages_list,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        return_dict=True # Esto asegura que tengamos input_ids y attention_mask
    ).to(model_device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, # Ahora inputs es un dict con todo en la GPU correcta
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id
        )
    
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