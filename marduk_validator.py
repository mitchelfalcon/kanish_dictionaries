import re

# Diccionario expandido para robustez
GEMATRIA_PRIME_MAP = {
    "KUG": 13, "BABBAR": 23, "KASPUM": 299, # Mapeo directo de compuestos
    "AN": 5, "NA": 17, 
    "DAM": 31, "GAR3": 29 
}

def calculate_gematria_signature(transliteration):
    """
    Mejorado: Normaliza índices y separa signos con Regex.
    Ej: 'KUG.BABBAR' o 'KUG-BABBAR' -> ['KUG', 'BABBAR']
    """
    # Elimina números de índice (ej: GAR3 -> GAR) si tu mapa no tiene índices,
    # o mantenlos si tu mapa es preciso. Asumiremos normalización a mayúsculas.
    clean_text = transliteration.upper()
    # Separa por cualquier no-alfanumérico
    tokens = re.split(r'[^A-Z0-9]+', clean_text)
    
    value = 1
    for token in tokens:
        if token in GEMATRIA_PRIME_MAP:
            value *= GEMATRIA_PRIME_MAP[token]
    return value

def enforce_semantic_correction(akkadian_input, english_pred):
    """
    MARDUK ACTIVO: Devuelve una tupla (predicción_final, flag_corrección)
    """
    signature = calculate_gematria_signature(akkadian_input)
    corrected_pred = english_pred
    log_msg = "CLEAN"

    # Regla: PLATA (299 = 13*23)
    if signature % 299 == 0:
        if not any(w in english_pred.lower() for w in ["silver", "money"]):
            # INYECCIÓN FORZADA: Agregamos el término faltante
            corrected_pred = f"{english_pred} [silver]"
            log_msg = "FIXED_SILVER"

    # Regla: COMERCIANTE (899 = 31*29)
    if signature % 899 == 0:
        if "merchant" not in english_pred.lower():
            corrected_pred = f"{english_pred} [merchant]"
            log_msg = "FIXED_MERCHANT"
            
    return corrected_pred, log_msg
