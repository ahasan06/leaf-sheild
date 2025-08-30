# app.py
from flask import Flask, request, jsonify, render_template
import os, io, json
import numpy as np
from PIL import Image

# Prefer tiny tflite-runtime; fallback to TensorFlow's interpreter (Railway often uses TF CPU)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # requires tensorflow / tensorflow-cpu

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
    template_folder=os.path.join(BASE_DIR, "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

# -----------------------------------------------------------------------------
# Model registry (edit labels if your training order differs, or provide *_class_indices.json)
# -----------------------------------------------------------------------------
MODELS = {
    "watermelon": {
        "path": "model/watermelon.tflite",
        "labels": ["Anthracnose", "Downy_Mildew", "Healthy", "Mosaic_Virus"],
        "input_range": os.environ.get("WATERMELON_RANGE", "0_255"),  # "0_255" or "0_1" for float32 inputs
        "class_indices_json": "model/watermelon_class_indices.json",
        "title": "Watermelon Leaf Disease Classifier",
    },
    "guava": {
        "path": "model/guava.tflite",
        "labels": ["Anthracnose", "Canker", "Dot", "Healthy", "Rust"],
        "input_range": os.environ.get("GUAVA_RANGE", "0_255"),
        "class_indices_json": "model/guava_class_indices.json",
        "title": "Guava Leaf Disease Classifier",
    },
    "grapes": {
        "path": "model/grapes.tflite",
        # Default to alphabetical folder order: Bacterial Leaf Spot, Downy Mildew, Healthy Leaves, Powdery Mildew
        "labels": ["Bacterial Leaf Spot", "Downy Mildew", "Healthy Leaves", "Powdery Mildew"],
        "input_range": os.environ.get("GRAPES_RANGE", "0_255"),
        "class_indices_json": "model/grapes_class_indices.json",
        "title": "Grapes Leaf Disease Classifier",
    },
}

CACHE = {}  # loaded interpreters and metadata


def _abs_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


def load_labels(conf, n_hint=None):
    """Load labels from class_indices.json if available; else use conf['labels']."""
    j = conf.get("class_indices_json")
    if j:
        j_abs = _abs_path(j)
        if os.path.exists(j_abs):
            try:
                with open(j_abs, "r", encoding="utf-8") as f:
                    ci = json.load(f)  # {"Label": idx, ...}
                inv = {v: k for k, v in ci.items()}
                return [inv[i] for i in range(len(inv))]
            except Exception as e:
                print(f"[WARN] Could not read {j_abs}: {e}")
    labels = conf.get("labels") or []
    if n_hint and len(labels) != n_hint:
        return [f"class_{i}" for i in range(n_hint)]
    return labels


def get_bundle(name: str):
    """Load (or return cached) interpreter + metadata for model `name`."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Valid: {list(MODELS.keys())}")

    if name in CACHE:
        return CACHE[name]

    conf = MODELS[name]
    path = _abs_path(conf["path"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"[INFO] Loading TFLite model: {name} -> {path}")
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()

    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    in_idx = in_det[0]["index"]
    out_idx = out_det[0]["index"]

    in_shape = in_det[0]["shape"].tolist()  # [1, H, W, C]
    in_dtype = in_det[0]["dtype"]
    in_quant = in_det[0].get("quantization", (0.0, 0))
    out_dtype = out_det[0]["dtype"]
    out_quant = out_det[0].get("quantization", (0.0, 0))
    _, H, W, C = in_shape

    bundle = {
        "interpreter": interp,
        "in_idx": in_idx,
        "out_idx": out_idx,
        "in_shape": in_shape,
        "in_dtype": in_dtype,
        "in_quant": in_quant,
        "out_dtype": out_dtype,
        "out_quant": out_quant,
        "H": H,
        "W": W,
        "C": C,
        "input_range": (conf.get("input_range") or "0_255").lower(),
        "labels": load_labels(conf),
        "name": name,
        "title": conf.get("title", name.title()),
        "path": path,
    }
    CACHE[name] = bundle
    print(f"[INFO] {name} input: {in_shape} dtype={in_dtype} quant={in_quant}  output dtype={out_dtype} quant={out_quant}")
    return bundle


def preprocess(img: Image.Image, b):
    """Resize and normalize / quantize according to model input."""
    img = img.convert("RGB") if b["C"] == 3 else img.convert("L")
    img = img.resize((b["W"], b["H"]), Image.BILINEAR)
    x = np.array(img).astype(np.float32)
    if b["C"] == 1:
        x = np.expand_dims(x, -1)

    if b["in_dtype"] == np.float32:
        x = (x / 255.0) if b["input_range"] == "0_1" else x  # If model has internal rescaling, keep 0_255
        return np.expand_dims(x.astype(np.float32), 0)

    # Quantized input
    scale, zp = b["in_quant"] if b["in_quant"] is not None else (0.0, 0)
    x_norm = x / 255.0
    if b["in_dtype"] == np.uint8:
        xq = (x_norm / scale + zp) if scale and scale > 0 else x
        xq = np.clip(np.round(xq), 0, 255).astype(np.uint8)
        return np.expand_dims(xq, 0)
    if b["in_dtype"] == np.int8:
        xq = (x_norm / scale + zp) if scale and scale > 0 else (x_norm * 255.0 - 128.0)
        xq = np.clip(np.round(xq), -128, 127).astype(np.int8)
        return np.expand_dims(xq, 0)

    raise ValueError(f"Unsupported input dtype: {b['in_dtype']}")


def dequantize(y, b):
    """Dequantize outputs if needed."""
    y = np.array(y)
    if b["out_dtype"] in (np.uint8, np.int8):
        scale, zp = b["out_quant"] if b["out_quant"] is not None else (0.0, 0)
        if scale and scale > 0:
            y = scale * (y.astype(np.float32) - zp)
        else:
            y = y.astype(np.float32)
    return y


def softmax_if_needed(vec):
    """Apply softmax if outputs are not already normalized probabilities."""
    vec = np.array(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec[None, :]
    sums = np.sum(vec, axis=1, keepdims=True)
    if not np.allclose(sums, 1.0, atol=1e-3) or np.any(vec < 0) or np.any(vec > 1):
        e = np.exp(vec - np.max(vec, axis=1, keepdims=True))
        vec = e / np.sum(e, axis=1, keepdims=True)
    return vec[0]


# -----------------------------------------------------------------------------
# Optional: download models at boot if IDs are provided (so you don't commit weights)
# -----------------------------------------------------------------------------
def maybe_download_models():
    try:
        import gdown
    except Exception:
        # gdown not installed: silently skip (only needed if you set the *_GDRIVE_ID env vars)
        print("[INFO] gdown not installed; skip boot-time model download.")
        return

    os.makedirs(_abs_path("model"), exist_ok=True)
    mapping = {
        "watermelon": os.getenv("WATERMELON_GDRIVE_ID"),
        "guava": os.getenv("GUAVA_GDRIVE_ID"),
        "grapes": os.getenv("GRAPES_GDRIVE_ID"),
    }
    for name, fid in mapping.items():
        if not fid:
            continue
        rel = MODELS[name]["path"]
        path = _abs_path(rel)
        if os.path.exists(path):
            continue
        url = f"https://drive.google.com/uc?id={fid}"
        print(f"[DL] {name} -> {path}")
        try:
            gdown.download(url, path, quiet=False)
        except Exception as e:
            print(f"[WARN] download failed for {name}: {e}")


# Call on import
maybe_download_models()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/watermelon")
def page_watermelon():
    c = MODELS["watermelon"]
    return render_template("watermelon.html", title=c["title"], model_name="watermelon", labels=c["labels"])


@app.get("/guava")
def page_guava():
    c = MODELS["guava"]
    return render_template("guava.html", title=c["title"], model_name="guava", labels=c["labels"])


@app.get("/grapes")
def page_grapes():
    c = MODELS["grapes"]
    return render_template("grapes.html", title=c["title"], model_name="grapes", labels=c["labels"])


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/model_info")
def model_info():
    name = request.args.get("model", "watermelon")
    try:
        b = get_bundle(name)
        return jsonify({
            "name": name,
            "path": b["path"],
            "input_shape": b["in_shape"],
            "input_dtype": str(b["in_dtype"]),
            "input_quant": b["in_quant"],
            "output_dtype": str(b["out_dtype"]),
            "output_quant": b["out_quant"],
            "labels": b["labels"],
            "input_range_float32": b["input_range"] if b["in_dtype"] == np.float32 else None
        })
    except Exception as e:
        p = MODELS.get(name, {}).get("path")
        abs_p = _abs_path(p or "")
        return jsonify({
            "model": name,
            "error": str(e),
            "cwd": os.getcwd(),
            "declared_path": p,
            "absolute_path": abs_p,
            "exists": os.path.exists(abs_p)
        }), 400


@app.post("/predict")
def predict():
    try:
        name = request.form.get("model", "watermelon")

        # Lazy-load interpreter
        try:
            b = get_bundle(name)
        except Exception as e:
            return jsonify({"error": f"Model '{name}' is not available on the server: {e}"}), 400

        # File validation
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files["image"]
        if not f.filename:
            return jsonify({"error": "No file selected"}), 400

        # Open image
        try:
            img = Image.open(io.BytesIO(f.read()))
        except Exception as e:
            return jsonify({"error": f"Invalid image: {e}"}), 400

        # Run inference
        x = preprocess(img, b)
        b["interpreter"].set_tensor(b["in_idx"], x)
        b["interpreter"].invoke()
        y = b["interpreter"].get_tensor(b["out_idx"])[0]
        y = dequantize(y, b)
        probs = softmax_if_needed(y)

        n_out = probs.shape[-1]
        labels = b["labels"] if len(b["labels"]) == n_out else [f"class_{i}" for i in range(n_out)]
        results = [{"label": labels[i], "prob": float(probs[i])} for i in range(n_out)]
        results.sort(key=lambda d: d["prob"], reverse=True)

        return jsonify({
            "model": name,
            "top_class": results[0]["label"],
            "confidence": round(results[0]["prob"], 6),
            "all_probs": results
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Image too large (max 20 MB)"}), 413


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
