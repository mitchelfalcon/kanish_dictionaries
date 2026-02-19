import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import evaluate # Librería moderna de HF para métricas

# --- CONFIGURACIÓN DEL RIG ---
MODEL_NAME = "google/byt5-small"
OUTPUT_DIR = "./byt5-akkadian-v1"
MAX_INPUT_LENGTH = 512 # ByT5 usa más tokens (bytes), necesitamos longitud
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8         # Ajustar según VRAM (8 para 16GB VRAM, 4 para 8GB)
LEARNING_RATE = 4e-4   # ByT5 suele requerir LRs un poco más altos
EPOCHS = 5

# --- CARGA Y PREPARACIÓN DE DATOS ---
def load_and_split_data(filepath):
    print(f"[-] Cargando dataset desde {filepath}...")
    df = pd.read_csv(filepath)
    # Limpieza básica
    df = df.dropna().astype(str)
    
    # Split 90% Train / 10% Validación
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"[-] Datos cargados. Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df

# --- TOKENIZACIÓN ---
def preprocess_function(examples, tokenizer):
    inputs = [ex for ex in examples["transliteration"]]
    targets = [ex for ex in examples["translation"]]
    
    # Prefixing: T5 espera saber qué tarea realizar
    inputs = ["translate Akkadian to English: " + inp for inp in inputs]
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

    # Reemplazar tokens de padding en labels con -100 para que la pérdida (loss) los ignore
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- CLASE DE DATASET PERSONALIZADA ---
class AkkadianDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.data = df.to_dict('records')
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # Procesamiento al vuelo
        processed = preprocess_function({"transliteration": [item['transliteration']], 
                                         "translation": [item['translation']]}, self.tokenizer)
        # Aplanar listas
        return {key: torch.tensor(val[0]) for key, val in processed.items()}

# --- MÉTRICAS ---
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Reemplazar -100 en labels para decodificar
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Limpieza simple para métricas
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # 1. Inicializar Tokenizer y Modelo
    print("[-] Inicializando Arquitectura ByT5...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 2. Datos
    train_df, val_df = load_and_split_data("train.csv") # <--- TU ARCHIVO AQUÍ
    train_dataset = AkkadianDataset(train_df, tokenizer)
    val_dataset = AkkadianDataset(val_df, tokenizer)

    # 3. Configuración de Entrenamiento
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,          # Guardar solo los 2 mejores checkpoints
        num_train_epochs=EPOCHS,
        predict_with_generate=True,  # Necesario para calcular BLEU durante eval
        fp16=torch.cuda.is_available(), # Activar aceleración mixta si hay GPU
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
    )

    # 4. Data Collator (Maneja el padding dinámico en el batch)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. IGNICIÓN
    print("[-] Iniciando Secuencia de Entrenamiento...")
    trainer.train()
    
    # 7. Guardado Final
    print(f"[-] Guardando modelo final en {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
