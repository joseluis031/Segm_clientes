# Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos
# Supongamos que los datos están en un archivo CSV llamado "telecom_customers.csv"
df = pd.read_csv('Client_segment_MODIFICADO.csv')

# Exploración inicial
print("Resumen de los datos:")
print(df.info())
print("\nPrimeras filas:")
print(df.head())

# Identificar características numéricas y categóricas
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# 2. Preprocesamiento de datos
# a) Manejo de valores faltantes
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# b) Codificación de variables categóricas y normalización de variables numéricas
categorical_encoder = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', numerical_imputer),
            ('scaler', scaler)
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('encoder', categorical_encoder)
        ]), categorical_features)
    ]
)

# Aplicar preprocesamiento
df_processed = preprocessor.fit_transform(df)

# 3. Determinación del número óptimo de clusters
# Probar diferentes números de clusters usando el método del codo y la silueta
inertia = []
silhouette_scores = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_processed)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_processed, kmeans.labels_))

# Graficar resultados
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')

plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Puntaje de Silueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Puntaje')

plt.tight_layout()
plt.show()

# Elegir el número óptimo de clusters (por ejemplo, k=4)
optimal_k = 4

# 4. Aplicar clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(df_processed)

# Agregar resultados al DataFrame original
df['Cluster'] = clusters

# 5. Interpretación de resultados
# Resumir las características principales de cada cluster
cluster_summary = df.groupby('Cluster').mean()
print("Resumen de clusters:")
print(cluster_summary)

# Visualización de clusters en dos dimensiones (usando PCA si hay muchas dimensiones)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_processed)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['Cluster'], palette='viridis', s=100)
plt.title('Visualización de Clusters (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster')
plt.show()

# 6. Validación del modelo
# Silhouette Score para el número óptimo de clusters
silhouette_avg = silhouette_score(df_processed, clusters)
print(f"Puntaje de Silueta para k={optimal_k}: {silhouette_avg:.2f}")

