from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import json
import os
from .models import BrainScanUpload
from .ml_service import classifier

def home(request):
    """Home page view"""
    recent_scans = BrainScanUpload.objects.all()[:5]
    return render(request, 'classifier/home.html', {
        'recent_scans': recent_scans
    })

@csrf_exempt
@require_http_methods(["POST"])
def upload_and_predict(request):
    """Handle image upload and prediction"""
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image file provided'}, status=400)
        
        image_file = request.FILES['image']
        
        # Validate file type
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        file_extension = os.path.splitext(image_file.name)[1].lower()
        if file_extension not in allowed_extensions:
            return JsonResponse({'error': 'Invalid file type. Please upload JPG, PNG, or BMP files.'}, status=400)
        
        # Save uploaded file
        brain_scan = BrainScanUpload.objects.create(image=image_file)
        
        # Get full path to the uploaded image
        image_path = brain_scan.image.path
        
        # Make prediction
        prediction, confidence = classifier.predict(image_path)
        
        if prediction is None:
            return JsonResponse({'error': 'Failed to make prediction'}, status=500)
        
        # Update the database record
        brain_scan.prediction = prediction
        brain_scan.confidence = confidence
        brain_scan.save()
        
        return JsonResponse({
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'image_url': brain_scan.image.url,
            'scan_id': brain_scan.id
        })
        
    except Exception as e:
        return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

def scan_detail(request, scan_id):
    """View for individual scan details"""
    try:
        scan = BrainScanUpload.objects.get(id=scan_id)
        return render(request, 'classifier/scan_detail.html', {
            'scan': scan
        })
    except BrainScanUpload.DoesNotExist:
        return render(request, 'classifier/404.html', status=404)

def scan_history(request):
    """View for scan history"""
    scans = BrainScanUpload.objects.all()
    return render(request, 'classifier/history.html', {
        'scans': scans
    })

@csrf_exempt
def get_gradcam(request, scan_id):
    """Generate and return Grad-CAM visualization"""
    try:
        scan = BrainScanUpload.objects.get(id=scan_id)
        
        # Create output path for Grad-CAM image
        gradcam_filename = f"gradcam_{scan_id}.png"
        gradcam_path = os.path.join(settings.MEDIA_ROOT, 'gradcam', gradcam_filename)
        
        # Create gradcam directory if it doesn't exist
        os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
        
        # Generate Grad-CAM visualization
        success = classifier.generate_gradcam_visualization(scan.image.path, gradcam_path)
        
        if success:
            gradcam_url = f"{settings.MEDIA_URL}gradcam/{gradcam_filename}"
            return JsonResponse({
                'success': True,
                'gradcam_url': gradcam_url,
                'message': 'Grad-CAM visualization generated successfully'
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to generate Grad-CAM visualization'
            }, status=500)
        
    except BrainScanUpload.DoesNotExist:
        return JsonResponse({'error': 'Scan not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

