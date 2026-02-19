import pandas as pd
import re

def extract_golden_publications(csv_path='publications.csv', output_dir='golden_corpus/'):
    """
    Filtra los 900 PDFs para extraer solo las 'Obras Maestras' (Golden Set)
    basado en autores y palabras clave de alta densidad de traducción.
    """
    print(">>> Cargando base de datos de publicaciones...")
    # Carga selectiva para ahorrar RAM si el CSV es muy grande
    df = pd.read_csv(csv_path)
    
    # 1. Definir Filtros de 'Alta Calidad' (Heurística de Experto)
    # Buscamos coincidencias en el nombre del PDF o en el contenido inicial
    
    keywords_high_value = [
        r'Cécile.*Michel',  # Autoridad top
        r'Mogens.*Larsen',  # Autoridad top
        r'K\.R\..*Veenhof', # Autoridad top (AKT)
        r'AKT\s+\d',        # Serie Ankara Kültepe Tabletleri
        r'Kulmic',          # Abreviatura común para Michel Kültepe
        r'OAA',             # Old Assyrian Archives (Serie crítica)
    ]
    
    # Compilar regex para búsqueda rápida
    combined_pattern = "|".join(keywords_high_value)
    
    # 2. Filtrar el DataFrame
    # Asumimos que 'pdf_name' o el texto contienen estas pistas.
    # A veces la metadata está en 'bibliography.csv', pero filtramos por nombre de archivo primero.
    
    golden_df = df[df['pdf_name'].str.contains(combined_pattern, case=False, regex=True, na=False)]
    
    print(f">>> Se encontraron {len(golden_df)} documentos de 'Alta Densidad'.")
    
    # 3. Exportar para Inspección Manual o Procesamiento
    # Guardamos cada "PDF" como un archivo de texto separado para facilitar la alineación posterior.
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for index, row in golden_df.iterrows():
        # Limpiar nombre de archivo
        safe_name = re.sub(r'[^\w\-_\.]', '_', str(row['pdf_name']))
        content = row['page_text']
        
        # Guardar
        with open(f"{output_dir}/{safe_name}.txt", "w", encoding="utf-8") as f:
            f.write(str(content))
            
    print(f">>> Extracción completada en '{output_dir}'.")
    return golden_df

# Ejecutar
# extract_golden_publications()
