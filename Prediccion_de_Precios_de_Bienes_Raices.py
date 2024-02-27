# Autor: Tiziana Meza
# Fecha: Feb-2024
# Descripción:Predicción de precios de bienes raíces utilizando técnicas de machine learning para ayudar en la toma de decisiones.
# Versión de Python: 3.6
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Cargar datos
data = pd.read_csv("datos_bienes_raices.csv")

# Preprocesamiento de datos
data.dropna(inplace=True)  # Eliminar filas con valores faltantes
data = pd.get_dummies(data, columns=['Categoría'])  # Codificar variables categóricas

# Dividir datos en características (X) y variable objetivo (y)
X = data.drop(columns=['Precio'])
y = data['Precio']

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar modelo de regresión Ridge con validación cruzada
model = Ridge(alpha=0.5)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("CV Error cuadrático medio:", -np.mean(cv_scores))

# Entrenar modelo final
model.fit(X_train, y_train)

# Realizar predicciones
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluar modelo
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Error cuadrático medio (conjunto de entrenamiento):", mse_train)
print("Error cuadrático medio (conjunto de prueba):", mse_test)

# Visualizar resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(y_test, y_pred_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.title("Predicción de Precios de Bienes Raíces")
plt.show()
