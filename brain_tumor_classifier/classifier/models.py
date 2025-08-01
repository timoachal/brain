from django.db import models
import os

class BrainScanUpload(models.Model):
    """Model to store uploaded brain scan images"""
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=50, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    
    def __str__(self):
        return f"Brain Scan - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    class Meta:
        ordering = ['-uploaded_at']

