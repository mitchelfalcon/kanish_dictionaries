import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForSeq2Seq
from neural_infer import setup_neural_model
from marduk_validator import enforce_semantic_correction
from tqdm.auto import tqdm

# Configuración
BATCH_SIZE = 32  # Ajustar según VRAM de tu GPU
MAX_LENGTH = 128

class AkkadianDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data = df
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['transliteration']
        # Preparar input para T5
        inputs = self.tokenizer("translate Akkadian to English: " + text, 
                                max_length=MAX_LENGTH, truncation=True)
        return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}

def run_pipeline():
    # 1. Carga
    test_df = pd.read_csv("test.csv")
    tokenizer, model = setup_neural_model() # Asegurar que carga pesos FINE-TUNED
    
    # 2. Preparar DataLoader para velocidad
    dataset = AkkadianDataset(test_df, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=False)
    
    raw_predictions = []
    
    # 3. Inferencia Neuronal en Lotes (GPU pura)
    print(">>> Ejecutando Fase 2: Inferencia Neuronal Masiva...")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(model.device)
            # Generación
            outputs = model.generate(input_ids, max_length=512)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            raw_predictions.extend(decoded)
            
    # 4. Fase 3: Auditoría y Corrección MARDUK (CPU Post-Processing)
    print(">>> Ejecutando Fase 3: Protocolo MARDUK...")
    final_translations = []
    audit_log = []
    
    for original_text, pred in zip(test_df['transliteration'], raw_predictions):
        # Si el texto está roto/vacío, manejarlo
        if not pred or len(pred.strip()) == 0:
            pred = "[broken text]"
            
        final_pred, status = enforce_semantic_correction(original_text, pred)
        
        final_translations.append(final_pred)
        if status != "CLEAN":
            audit_log.append({"text": original_text, "orig_pred": pred, "final": final_pred, "action": status})

    # 5. Salida
    submission = pd.DataFrame({"id": test_df["id"], "translation": final_translations})
    submission.to_csv("submission_optimized.csv", index=False)
    pd.DataFrame(audit_log).to_csv("marduk_actions.csv", index=False)
    print(f"Finalizado. MARDUK corrigió {len(audit_log)} entradas.")

if __name__ == "__main__":
    run_pipeline()
