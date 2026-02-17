import re

class SXProcessor:
    def __init__(self):
        self.global_signs = {}
        self.global_values = {} # value -> sign_name

    def validate_asl_block(self, asl_text):
        report = {"errors": [], "warnings": [], "status": "Clean"}
        
        # 1. Parsing básico
        sign_match = re.search(r"@sign ([\w|×.()&%@+]+)", asl_text)
        if not sign_match:
            return {"status": "Error", "message": "No @sign header found."}
        
        sign_name = sign_match.group(1)
        parent_values = re.findall(r"@v\s+([a-z0-9_ₓ?]+)", asl_text)
        
        # 2. Validación de Unicidad Global de Signo
        if sign_name in self.global_signs:
            report["errors"].append(f"Global uniqueness violation: Sign '{sign_name}' already exists.")
        self.global_signs[sign_name] = True

        # 3. Procesamiento de Formas (@form)
        forms = re.findall(r"@form\s+([\w|@]+)(.*?)@@", asl_text, re.DOTALL)
        for form_name, form_body in forms:
            explicit_v = re.findall(r"@v\s+([a-z0-9_ₓ?]+)", form_body)
            
            # Regla de Herencia y Conflicto de Base
            for v in parent_values:
                base = re.sub(r'[0-9ₓ]+', '', v) # Extraer base (ej: 'du' de 'du3')
                conflicts = [ev for ev in explicit_v if re.sub(r'[0-9ₓ]+', '', ev) == base]
                
                if conflicts:
                    report["warnings"].append(f"Value conflict in form '{form_name}': Inherited '{v}' overridden by '{conflicts[0]}'.")

            # 4. Regla de Calificación (x-values)
            for v in explicit_v:
                if "ₓ" in v or v.endswith("x"):
                    report["warnings"].append(f"Qualification required: Value '{v}' in '{form_name}' must be used as {v}({sign_name}).")

        # 5. Detección de Deprecación
        if "@sign-" in asl_text or "@v-" in asl_text:
            report["warnings"].append("Deprecated elements detected. GVL will trigger errors in production.")

        if report["errors"]: report["status"] = "Fail"
        return report

# --- Ejecución de Prueba ---
sx = SXProcessor()

asl_input = """
@sign |KA×GAR|
@v	gu7
@v  du3
@form KA
@v	gu3
@v	subₓ
@@
@end sign
"""

results = sx.validate_asl_block(asl_input)
print(f"Report for Nailea:\n{results}")
