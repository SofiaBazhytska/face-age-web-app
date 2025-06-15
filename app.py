from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import numpy as np
from tensorflow import keras
import base64
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

try:
    model_classification = keras.models.load_model('weights/mobilenet_age_clas.h5')
    logging.debug("Classification model loaded.")
except Exception as e:
    logging.error(f"Error loading classification model: {e}")
    model_classification = None

try:
    model_regression = keras.models.load_model('weights/mobilenet_age_reg.h5')
    logging.debug("Regression model loaded.")
except Exception as e:
    logging.error(f"Error loading regression model: {e}")
    model_regression = None

class_labels = [
    '1–9 років', '10–20 років', '21–26 років', '27–34 років',
    '35–45 років', '45–60 років', '60–99 років'
]

img_size = (100, 100)


def predict_image(file, model_type):
    """Обробка зображення та передбачення"""
    img = Image.open(file).resize(img_size).convert('RGB')
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    if model_type == 'classification':
        if model_classification is None:
            raise RuntimeError("Модель класифікації не доступна.")

        predictions = model_classification.predict(img_array)[0]
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        confidence = float(predictions[predicted_class])
        result_text = f"Передбачено вікову групу: {predicted_label}"

        return {
            'age': result_text,
            'confidence': confidence,
            'image': img_base64
        }

    elif model_type == 'regression':
        if model_regression is None:
            raise RuntimeError("Регресійна модель не доступна.")

        predicted_age = float(model_regression.predict(img_array)[0][0])
        result_text = f"Передбачено точний вік: {predicted_age:.1f} років"

        return {
            'age': result_text,
            'confidence': None,
            'image': img_base64
        }

    else:
        raise ValueError("Невідомий тип моделі.")


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('file')
        model_type = request.form.get('model_type')
        is_json = request.headers.get('Content-Type', '').startswith('application/json') or \
                  request.headers.get('Accept', '').startswith('application/json')

        if not file or file.filename == '':
            error_msg = "Файл не вибрано."
            logging.error(error_msg)
            return jsonify({'error': error_msg}), 400 if is_json else render_template('index.html', error=error_msg)

        try:
            result = predict_image(file, model_type)

            if is_json:
                return jsonify(result)

            return render_template('index.html',
                                   age=result['age'],
                                   confidence=result['confidence'],
                                   image=result['image'])

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            error_msg = f'Помилка під час передбачення: {str(e)}'
            return jsonify({'error': error_msg}), 500 if is_json else render_template('index.html', error=error_msg)

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint для передбачення віку"""
    file = request.files.get('file')
    model_type = request.form.get('model_type')

    if not file or file.filename == '':
        return jsonify({'error': 'Файл не вибрано.'}), 400

    try:
        result = predict_image(file, model_type)

        response = {
            'success': True,
            'result_text': result['age']
        }

        if model_type == 'classification':
            response['age_group'] = result['age']
            response['confidence'] = result['confidence']
        elif model_type == 'regression':
            response['predicted_age'] = float(result['age'].split(":")[1].strip().split()[0])

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': f'Помилка під час передбачення: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
