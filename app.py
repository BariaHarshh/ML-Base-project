# =============================================================================
# app.py — Flask Web Application for Bone Fracture Detection
# =============================================================================

import os
import uuid
import traceback
from flask import Flask, request, render_template, jsonify
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import config
from model import load_model
from gradcam import predict_and_gradcam

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = os.urandom(24)

os.makedirs(config.STATIC_UPLOADS,  exist_ok=True)
os.makedirs(config.STATIC_HEATMAPS, exist_ok=True)

print("[APP] Loading model at startup...")
try:
    MODEL = load_model()
    print("[APP] ✅ Model ready for inference.")
except FileNotFoundError as e:
    print(f"[APP] ❌ {e}")
    MODEL = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def cleanup_old_files(directory, keep_last=20):
    try:
        files = sorted([os.path.join(directory, f) for f in os.listdir(directory)], key=os.path.getmtime)
        for f in files[:-keep_last]:
            os.remove(f)
    except Exception:
        pass

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return render_template('result.html', error="Model not loaded. Run python train.py first.", disclaimer=config.MEDICAL_DISCLAIMER, label=None)

    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")
    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Use JPG or PNG.")

    try:
        ext         = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        upload_path = os.path.join(config.STATIC_UPLOADS, unique_name)
        file.save(upload_path)

        # Call predict_and_gradcam — returns: pred_label, pred_prob, confidence, heatmap_fname
        result = predict_and_gradcam(
            img_path    = upload_path,
            model       = MODEL,
            save_static = True,
        )

        original_web   = 'uploads/' + unique_name
        heatmap_web    = 'heatmaps/' + result['heatmap_fname']
        label          = result['pred_label']
        prob           = result['pred_prob']
        confidence     = result['confidence'] * 100
        confidence_str = f"{confidence:.1f}%"
        label_class    = 'fractured' if label == 'Fractured' else 'normal'

        if label == 'Fractured':
            risk = "High confidence fracture detected" if confidence >= 80 else "Moderate confidence — fracture likely" if confidence >= 60 else "Low confidence — possible fracture"
        else:
            risk = "High confidence — no fracture detected" if confidence >= 80 else "Moderate confidence — likely normal" if confidence >= 60 else "Low confidence — borderline case"

        cleanup_old_files(config.STATIC_UPLOADS)
        cleanup_old_files(config.STATIC_HEATMAPS)

        return render_template('result.html',
            label=label, label_class=label_class, confidence=confidence_str,
            risk=risk, probability=f"{prob:.4f}", threshold=config.THRESHOLD,
            original_img=original_web, heatmap_img=heatmap_web,
            disclaimer=config.MEDICAL_DISCLAIMER, error=None,
        )

    except Exception as e:
        traceback.print_exc()
        return render_template('result.html', error=f"Analysis error: {str(e)}", disclaimer=config.MEDICAL_DISCLAIMER, label=None)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': MODEL is not None, 'threshold': config.THRESHOLD})

if __name__ == '__main__':
    print(f"\n  URL: http://localhost:{config.FLASK_PORT}\n")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
