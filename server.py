import os
import django
from django.conf import settings
from django.core.management import execute_from_command_line
from django.core.wsgi import get_wsgi_application
from django.http import JsonResponse
from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from predict import *

device = initialize_device()
TEXT, LABEL = load_vocabularies()
model = create_model(len(TEXT.vocab))
model = load_latest_model(model, device)


# Ensure Django settings are only configured once
if not settings.configured:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    settings.configure(
        DEBUG=True,
        SECRET_KEY="your-secret-key",
        ROOT_URLCONF="server",
        ALLOWED_HOSTS=["*"],
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
        ],
        MIDDLEWARE=[
            "django.middleware.common.CommonMiddleware",
        ],
    )




# Initialize Django
django.setup()

# Views
def index(request):
    return JsonResponse({"message": "Welcome to the index page"})

def handle_get(request):
    return JsonResponse({"message": "This is a GET request"})

def simple_pred(request):
    text = "this is a simple test"
    prediction, pred_label, confidence, probs = predict(model, text, TEXT, device)
    return JsonResponse({"prediction": str(pred_label), "confidence": str(confidence), "probabilities": str(probs)})

def check_prediction(request, check_string):
    prediction, pred_label, confidence, probs = predict(model, check_string, TEXT, device)
    return JsonResponse({"prediction": str(pred_label), "confidence": str(confidence), "probabilities": str(probs)})

@csrf_exempt
def handle_post(request):
    if request.method == "POST":
        return JsonResponse({"message": "This is a POST request"})
    return JsonResponse({"error": "Invalid request method"}, status=400)



# URL patterns
urlpatterns = [
    path("", index),  # Index route (/)
    path("get/", simple_pred),
    path("post/", handle_post),
    path("check/<str:check_string>/", check_prediction),
]

# WSGI application
application = get_wsgi_application()

# Run Django server
if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server")
    execute_from_command_line(["server.py", "runserver", "0.0.0.0:8000"])
