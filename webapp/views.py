from django.shortcuts import render
from src.imageSegmentation.pipeline.predict import PredictionPipeline
from django.http import JsonResponse
import os



def homepage(request):
    return render(request, "index.html")



def segment_image(request):
    if request.method == "POST":
        # Check if a file was uploaded
        if "image" not in request.FILES:
            return JsonResponse({"error": "No file selected."})

        image_file = request.FILES["image"]

        if not image_file:
            return JsonResponse({"error": "No file selected."})

        allowed_formats = {"jpg", "jpeg", "png", "gif"}
        if not image_file.name.lower().endswith(tuple(allowed_formats)):
            return JsonResponse({"error": "Invalid file format."})

        temp_image_path = "temp_image.png"
        with open(temp_image_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        pipeline = PredictionPipeline(temp_image_path)
        processed_image_string = pipeline.predict()
        os.remove(temp_image_path)

        return JsonResponse({"segmented_image": processed_image_string})

    return JsonResponse({"error": "Invalid request method."})


def trainingmodel(request):
    if request.method == "GET":
        os.system('python main.py')
        return render(request, "train.html")






################ This code for showing the only segmentation image on the screen.... it won't show actual image

# def homepage(request):
#     if request.method == "POST":
#         # Check if a file was uploaded
#         if "image" not in request.FILES:
#             context = {"error": "No file selected."}
#             return render(request, "index.html", context)

#         image_file = request.FILES["image"]

#         if not image_file:
#             context = {"error": "No file selected."}
#             return render(request, "index.html", context)

#         allowed_formats = {"jpg", "jpeg", "png", "gif"}
#         if not image_file.name.lower().endswith(tuple(allowed_formats)):
#             context = {"error": "Invalid file format."}
#             return render(request, "index.html", context)

#         temp_image_path = "temp_image.png"
#         with open(temp_image_path, 'wb') as f:
#             for chunk in image_file.chunks():
#                 f.write(chunk)

#         pipeline = PredictionPipeline(temp_image_path)
#         processed_image_string = pipeline.predict()
#         os.remove(temp_image_path)

#         context = {"processed_image_string": processed_image_string}
#         return render(request, "index.html", context)
#     return render(request, "index.html")