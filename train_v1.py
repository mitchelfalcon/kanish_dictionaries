pip install transformers datasets torch sentencepiece accelerate pandas

import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

# --- 1. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'working', 'train_ready_sprint1.csv')

# Apunta esto a donde descomprimiste 'akkadian-byt5...'
# Debe contener config.json
MODEL_PATH = os.path.join(BASE_DIR, 'input', 'pretrained_models', 'byt5') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'working', 'byt5-fine-tuned')

def run_training():
    print(">>> 1. Cargando Datos...")
    try:
        df = pd.read_csv(DATA_PATH)
        # Convertir a formato HuggingFace Dataset
        # Para prueba rápida, si tienes CPU, usa solo las primeras 100 filas
        # dataset = Dataset.from_pandas(df.head(100)) 
        dataset = Dataset.from_pandas(df) # Usa todo si tienes GPU
        
        # Split Train/Validation (90% entrenamiento, 10% validación)
        dataset = dataset.train_test_split(test_size=0.1)
        print(f"    Datos cargados. Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")
    except Exception as e:
        print(f"!!! Error cargando CSV: {e}")
        return

    print(">>> 2. Cargando Tokenizer y Modelo Local...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)
    except OSError:
        print(f"!!! Error: No encuentro el modelo en {MODEL_PATH}")
        print("    Asegúrate de que 'config.json' esté directamente en esa carpeta.")
        return

    # --- 3. PREPROCESAMIENTO ---
    max_input_length = 512 # ByT5 soporta más, pero 512 ahorra memoria
    max_target_length = 512

    def preprocess_function(examples):
        inputs = [str(ex) for ex in examples["source"]]
        targets = [str(ex) for ex in examples["target"]]
        
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print(">>> 3. Tokenizando datos...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # --- 4. CONFIGURACIÓN DEL ENTRENAMIENTO ---
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Bajo para evitar error de memoria (sube a 8 si tienes buena GPU)
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,             # Solo guarda los 2 mejores checkpoints
        num_train_epochs=3,             # Pocas épocas para la primera prueba
        predict_with_generate=True,
        fp16=False,                     # Cambia a True si tienes GPU NVIDIA moderna
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(">>> 4. INICIANDO ENTRENAMIENTO...")
    trainer.train()
    
    print(">>> 5. GUARDANDO MODELO...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    print(">>> ÉXITO. Modelo guardado.")

if __name__ == "__main__":
    run_training()
