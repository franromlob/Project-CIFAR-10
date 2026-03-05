# --- EXPORTACIÓN PARA PRODUCCIÓN ---
import tensorflow as tf

# El versionado es clave (ej. carpeta '1')
model_path = "./main.ipynb"
tf.saved_model.save(model_custom_cnn, model_path)
print(f"Modelo guardado en {model_path} en formato SavedModel.")