import pandas as pd
import re
import nltk

# Descargar el tokenizador de oraciones en inglés si no lo tienes
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def align_sentences_gale_church(src_sents, tgt_sents):
    """
    Alineación simple basada en longitud relativa.
    Asume que la oración más larga en Acadio corresponde a la más larga en Inglés
    en una secuencia local.
    NOTA: Para producción, usaríamos algoritmos de programación dinámica (Gale-Church real).
    Aquí usamos una heurística 1:1 o truncamiento para el Sprint 1.
    """
    aligned = []
    # Si las cantidades coinciden, asumimos alineación 1:1 (Ideal)
    if len(src_sents) == len(tgt_sents):
        return list(zip(src_sents, tgt_sents))
    
    # Si no coinciden, intentamos alinear por proximidad relativa (Naive)
    # Estrategia defensiva: Solo tomamos hasta el mínimo común para no romper el código.
    # En Sprint 2, mejoraremos esto con 'laserembeddings' o similar.
    limit = min(len(src_sents), len(tgt_sents))
    return list(zip(src_sents[:limit], tgt_sents[:limit]))

def process_data_pipeline():
    print(">>> Cargando datasets...")
    # 1. Cargar Datos
    df_train = pd.read_csv('train.csv') # Columnas: oare_id, transliteration, translation
    df_helpers = pd.read_csv('Sentences_Oare_FirstWord_LinNum.csv') 
    
    # Preparamos lista para el nuevo dataset
    all_pairs = []
    
    # Agrupamos el helper por documento para acceso rápido
    # El helper nos dice dónde empieza cada oración en el Acadio
    doc_splits = df_helpers.groupby('text_id')

    print(f">>> Procesando {len(df_train)} documentos...")

    for idx, row in df_train.iterrows():
        doc_id = row['oare_id']
        full_translit = str(row['transliteration'])
        full_translat = str(row['translation'])
        
        # A. SEGMENTACIÓN ACADIA (Precisa, usando metadatos)
        akkadian_sentences = []
        if doc_id in doc_splits.groups:
            splits = doc_splits.get_group(doc_id).sort_values('line_start')
            
            # Aquí la lógica es compleja: el helper da el "FirstWord".
            # Para el Sprint 1, usaremos una heurística más robusta:
            # Dividir por puntos finales típicos si el helper falla o es confuso,
            # PERO dado el helper, intentamos reconstruir.
            
            # Simplificación Táctica: 
            # El helper es difícil de mapear sin un parser de ATF completo.
            # Vamos a usar NLTK/Regex para dividir el Acadio también, guiándonos por puntuación
            # si existe, o saltos de línea.
            pass
        
        # PLAN B (Más robusto para Sprint 1): Split por reglas gramaticales/puntuación
        # El Acadio transliterado suele separar frases con lógica.
        # Si no hay puntuación clara, este es el cuello de botella.
        
        # Usaremos split simple por ahora para ver qué tan ruidoso es.
        # Muchos textos en train.csv son cortos (cartas).
        
        # B. SEGMENTACIÓN INGLESA
        english_sentences = nltk.tokenize.sent_tokenize(full_translat)
        
        # C. SEGMENTACIÓN ACADIA (Heurística Fallback)
        # Asumimos que la cantidad de oraciones en inglés es la guía.
        # Intentamos dividir el acadio en N partes proporcionales? No, muy arriesgado.
        
        # ESTRATEGIA SEGURA SPRINT 1: ENTRENAMIENTO A NIVEL DOCUMENTO
        # Dado que alinear oraciones sin marcadores claros es un proyecto en sí mismo,
        # y que modelos como ByT5 soportan secuencias largas (hasta 1024 o 2048 tokens),
        # podemos empezar entrenando Doc -> Doc.
        # El test set pedirá oraciones, pero el modelo aprenderá alineación interna.
        
        # Sin embargo, si quieres intentar oraciones:
        # Dividimos el acadio por puntos '.' si existen, o lo dejamos como bloque.
        
        # DECISIÓN TÉCNICA:
        # Vamos a guardar pares (Documento, Documento) para la base,
        # Y pares (Oración, Oración) solo si logramos dividir ambos lados igual.
        
        # Intento de split acadio por signos de puntuación transliterados o saltos
        # Nota: La transliteración raw a menudo no tiene puntos.
        
        all_pairs.append({
            'source': full_translit,
            'target': full_translat,
            'type': 'document'
        })

    # Convertir a DataFrame
    df_final = pd.DataFrame(all_pairs)
    
    # Limpieza Básica (Tu función segura)
    # df_final['source'] = df_final['source'].apply(kaggle_normalize) # Usar tu función aquí
    
    print(f">>> Generados {len(df_final)} pares de entrenamiento (Nivel Documento).")
    df_final.to_csv('train_ready_sprint1.csv', index=False)

if __name__ == "__main__":
    process_data_pipeline()
