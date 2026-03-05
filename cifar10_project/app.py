import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. CARGA DIRECTA (Sustituye a TF Serving)
# Apuntamos a la carpeta del modelo que exportamos antes
MODEL_PATH = "deployed_model/1"
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Añadir dimensión de batch
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        results = []
        for file in files:
            if file.filename == '': continue
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            # Inferencia directa sin Docker
            tensor_img = preprocess_img(path)
            output = infer(tensor_img)
            # El nombre de la salida suele ser 'output_0' o similar en SavedModel
            preds = list(output.values())[0].numpy()[0]
            
            idx = np.argmax(preds)
            all_probs = {CLASSES[i]: f"{preds[i]*100:.2f}%" for i in range(10)}
            
            results.append({
                'name': file.filename, 'path': path,
                'prediction': CLASSES[idx], 'confidence': f"{preds[idx]*100:.2f}%",
                'probs': all_probs
            })
        return render_template('index.html', results=results)
    return render_template('index.html', results=None)

if __name__ == '__main__':
    # El puerto 7860 es el estándar de Hugging Face
    app.run(host='0.0.0.0', port=7860)