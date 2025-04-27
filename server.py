import os
import django
from django.conf import settings
from django.core.management import execute_from_command_line
from django.core.wsgi import get_wsgi_application
from django.http import JsonResponse
from django.urls import path
from django.views.decorators.csrf import csrf_exempt

# Import model and helper functions
try:
    from predict import (
        initialize_device,
        load_custom_mappings,
        create_model,
        load_model_weights,
        predict_sentence
    )
except ImportError:
    print("Error: Could not import functions from predict.py.")
    exit()

try:
    from libs import sql_tokenizer
except ImportError:
    print("Error: Could not import sql_tokenizer from libs.py.")
    exit()

# --- Constants ---
EMBEDDING_DIM = 100
HIDDEN_DIM = 32
NUM_LAYERS = 1
DROPOUT = 0.2
MAX_SEQ_LEN = 128

VOCAB_STOI_PATH = "pickles/vocab_stoi.pkl"
LABEL_TO_INT_PATH = "pickles/label_to_int.pkl"
SAVED_WEIGHTS_DIR = 'saved_weights/'

# Load model and mappings
print("Loading model and mappings...")
device = initialize_device()
stoi, int_to_label = load_custom_mappings(VOCAB_STOI_PATH, LABEL_TO_INT_PATH)

if stoi is None or int_to_label is None:
    print("Failed to load necessary mappings.")
    exit()

vocab_size = len(stoi)
output_dim = len(int_to_label)

model = create_model(vocab_size, output_dim, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = load_model_weights(model, device, SAVED_WEIGHTS_DIR)

if model is None:
    print("Failed to load model.")
    exit()

print("Model loaded successfully.")

# Configure Django settings (MINIMAL)
if not settings.configured:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    settings.configure(
        DEBUG=True,
        SECRET_KEY="your-secret-key",  # Change this later
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["*"],
        INSTALLED_APPS=[],  # Empty because no sessions, no auth
        MIDDLEWARE=[
            "django.middleware.common.CommonMiddleware",
            "django.middleware.clickjacking.XFrameOptionsMiddleware",
        ],
    )

# Setup Django
try:
    django.setup()
except Exception as e:
    print(f"Error during Django setup: {e}")

# Views
def index(request):
    return JsonResponse({"message": "Welcome to the prediction server"})

def simple_pred(request):
    text = "this is a simple test"
    predicted_label, confidence, probabilities = predict_sentence(
        model, text, stoi, int_to_label, sql_tokenizer, device, MAX_SEQ_LEN
    )

    if predicted_label is not None:
        return JsonResponse({
            "sentence": text,
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%",
            "probabilities": {label: f"{prob:.2f}%" for label, prob in probabilities.items()}
        })
    else:
        return JsonResponse({"error": "Prediction failed"}, status=500)

@csrf_exempt
def check_prediction(request, check_string):
    check_string = check_string.replace("'","").replace('"', "")
    print(sql_tokenizer(check_string))
    predicted_label, confidence, probabilities = predict_sentence(
        model, check_string, stoi, int_to_label, sql_tokenizer, device, MAX_SEQ_LEN
    )

    if predicted_label is not None:
        return JsonResponse({
            "sentence": check_string,
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%",
            "probabilities": {label: f"{prob:.2f}%" for label, prob in probabilities.items()}
        })
    else:
        return JsonResponse({"error": "Prediction failed"}, status=500)

@csrf_exempt
def handle_post(request):
    if request.method == "POST":
        try:
            import json
            data = json.loads(request.body)
            text = data.get("text")

            if not text:
                return JsonResponse({"error": "Missing 'text' in request body"}, status=400)

            predicted_label, confidence, probabilities = predict_sentence(
                model, text, stoi, int_to_label, sql_tokenizer, device, MAX_SEQ_LEN
            )

            if predicted_label is not None:
                return JsonResponse({
                    "sentence": text,
                    "prediction": predicted_label,
                    "confidence": f"{confidence:.2f}%",
                    "probabilities": {label: f"{prob:.2f}%" for label, prob in probabilities.items()}
                })
            else:
                return JsonResponse({"error": "Prediction failed"}, status=500)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            print(f"Error in handle_post: {e}")
            return JsonResponse({"error": "Internal error"}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)

# URL patterns
urlpatterns = [
    path("", index),
    path("predict/simple/", simple_pred),
    path("predict/check/<str:check_string>/", check_prediction),
    path("predict/post/", handle_post),
]

# WSGI application
application = get_wsgi_application()

# Start server
if __name__ == "__main__":
    print("Starting server...")
    execute_from_command_line(["server.py", "runserver", "0.0.0.0:8000"])
