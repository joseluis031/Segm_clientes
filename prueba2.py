import pandas as pd

# Ruta del archivo original y el archivo corregido
file_path = 'Client_segment_MODIFICADO.csv'  # Reemplaza con la ruta de tu archivo
cleaned_file_path = 'Client_segment_LIMPIO.csv'

try:
    # Intentar leer el archivo con una codificación alternativa
    df = pd.read_csv(file_path, encoding='latin1')
except UnicodeDecodeError:
    print("Error de codificación. Intentando con otra codificación.")
    df = pd.read_csv(file_path, encoding='utf-8', errors='replace')

# Limpiar la columna 'Provincia'
if 'Provincia' in df.columns:
    df['Provincia'] = df['Provincia'].apply(
        lambda x: x.encode('ascii', 'ignore').decode('ascii') if isinstance(x, str) else x
    )
    print("Columna 'Provincia' limpiada con éxito.")
else:
    print("La columna 'Provincia' no existe en el archivo.")

# Guardar el DataFrame limpio en un nuevo archivo
df.to_csv(cleaned_file_path, index=False)

print(f"Archivo corregido guardado como: {cleaned_file_path}")
