from phoenix_cleanup import clean_phoenix_2014

ejemplos = [
    "loc-DEUTSCHLAND cl-KOMMEN __EMOTION__ MORGEN",
    "ICH HABEN2 HAUS",
    "WIE AUSSEHEN DEIN AUTO",
    "D A S +BUCH",
    "MANN MANN GEHEN GEHEN NACH HAUSE"
]

print("--- DEMOSTRACION TEXTO A GLOSA (PHOENIX) ---")
print("")

for i, texto in enumerate(ejemplos):
    resultado = clean_phoenix_2014(texto)
    print(f"Original {i+1}: {texto}")
    print(f"Glosa:      {resultado}")
    print("-" * 30)
