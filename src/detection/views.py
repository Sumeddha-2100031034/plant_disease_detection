from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .models import get_disease_model
from .utils import preprocess_image, predict_disease

def detect_disease(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        image_path = fs.save(image.name, image)
        image_url = fs.url(image_path)
        
        disease_name = predict_disease(fs.path(image_path))
        
        # Add some treatment suggestions based on the predicted disease
        suggestions = {
            'Disease1': 'Effective management of pepper bacterial spot involves a combination of cultural practices, resistant varieties, chemical treatments, biological controls, and integrated pest management. Regular monitoring and early intervention are crucial for controlling the disease and minimizing its impact on pepper crops',


            
            'Disease2': 'Treatment for Disease2...',
            # Add more disease-treatment pairs
        }
        treatment = suggestions.get(disease_name, 'No suggestions available.')
        
        return render(request, 'detection_result.html', {
            'image_url': image_url,
            'disease_name': disease_name,
            'treatment': treatment
        })
    return render(request, 'upload.html')

def index(request):
    return render(request, 'index.html')

def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_url = fs.url(filename)

        # Predict disease
        file_path = fs.path(filename)
        disease, precautions = predict_disease(file_path)

        return render(request, 'result.html', {
            'file_url': file_url,
            'disease': disease,
            'precautions': precautions
        })
    return render(request, 'index.html')
