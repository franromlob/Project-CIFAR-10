# Usamos una imagen de Python oficial
FROM python:3.11-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos los archivos de requerimientos e instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de los archivos (incluyendo el modelo y templates)
COPY . .

# Exponemos el puerto que usa Flask (Hugging Face suele usar el 7860)
EXPOSE 7860

# Ejecutamos la app (ajustamos el host y el puerto para Hugging Face)
CMD ["python", "app.py"]