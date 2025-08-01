# NeuroClassifier - Brain Tumor Detection Web Application

A sophisticated Django web application that uses artificial intelligence to detect brain tumors in medical scans with Grad-CAM visualization for explainable AI.

## Features

### 🧠 AI-Powered Classification
- Advanced deep learning model for brain tumor detection
- Binary classification: "No Tumor" vs "Tumor Present"
- High-accuracy predictions with confidence scores
- Support for JPG, PNG, and BMP image formats

### 🔥 Grad-CAM Visualization
- Gradient-weighted Class Activation Mapping
- Visual explanation of AI decision-making process
- Heat map overlays highlighting areas of focus
- Red/yellow regions indicate high attention areas

### 🎨 Modern UI/UX Design
- Responsive design for desktop and mobile devices
- Professional medical-themed styling
- Gradient backgrounds and smooth animations
- Bootstrap 5 integration with custom CSS
- Font Awesome icons and Google Fonts

### 📊 Scan Management
- Upload and store brain scan images
- View scan history with thumbnails
- Detailed scan analysis pages
- Confidence level visualization
- Technical metadata display

## Technology Stack

### Backend
- **Django 4.x** - Web framework
- **TensorFlow/Keras** - Deep learning model
- **OpenCV** - Image processing
- **NumPy** - Numerical computations
- **Pillow** - Image handling

### Frontend
- **HTML5/CSS3** - Structure and styling
- **Bootstrap 5** - Responsive framework
- **JavaScript (ES6+)** - Interactive functionality
- **Font Awesome** - Icon library
- **Google Fonts** - Typography

### AI/ML Components
- **Pre-trained CNN Model** - Brain tumor classification
- **Grad-CAM Implementation** - Explainable AI visualization
- **Image Preprocessing** - Standardized input pipeline

## Project Structure

```
brain_tumor_classifier/
├── brain_tumor_classifier/          # Django project settings
│   ├── __init__.py
│   ├── settings.py                  # Configuration
│   ├── urls.py                      # URL routing
│   └── wsgi.py                      # WSGI configuration
├── classifier/                      # Main Django app
│   ├── models.py                    # Database models
│   ├── views.py                     # View controllers
│   ├── urls.py                      # App URL patterns
│   ├── ml_service.py               # ML model integration
│   ├── gradcam.py                  # Grad-CAM implementation
│   └── templates/classifier/        # HTML templates
│       ├── base.html               # Base template
│       ├── home.html               # Homepage
│       ├── history.html            # Scan history
│       ├── scan_detail.html        # Individual scan details
│       └── 404.html                # Error page
├── static/                         # Static assets
│   ├── css/style.css              # Custom styles
│   ├── js/main.js                 # JavaScript functionality
│   └── images/                    # Icons and sample images
├── media/                         # User uploads
│   ├── scans/                     # Original brain scans
│   └── gradcam/                   # Generated visualizations
├── tumor_model.h5                 # Pre-trained AI model
├── manage.py                      # Django management script
└── README.md                      # This file
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone or extract the project:**
   ```bash
   cd brain_tumor_classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install django tensorflow opencv-python pillow numpy
   ```

3. **Run database migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

4. **Create static and media directories:**
   ```bash
   mkdir -p static/css static/js static/images
   mkdir -p media/scans media/gradcam
   ```

5. **Start the development server:**
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```

6. **Access the application:**
   Open your browser and navigate to `http://localhost:8000`

## Usage Instructions

### 1. Upload Brain Scan
- Navigate to the homepage
- Click "Start Analysis" or scroll to the upload section
- Drag and drop an image or click "Choose File"
- Supported formats: JPG, PNG, BMP
- Maximum file size: 10MB

### 2. View Analysis Results
- After upload, the AI will process the image
- View prediction results with confidence scores
- See technical details and metadata
- Generate Grad-CAM visualization for explainable AI

### 3. Scan History
- Access all previous scans via the "History" page
- View thumbnails and quick statistics
- Click on individual scans for detailed analysis
- Generate Grad-CAM visualizations for any scan

### 4. Grad-CAM Visualization
- Click "Generate Grad-CAM" on any analyzed scan
- View heat map overlay showing AI attention areas
- Red/yellow areas indicate high focus regions
- Blue areas indicate low attention regions

## Model Information

### Architecture
- **Type:** Convolutional Neural Network (CNN)
- **Input Size:** 224 × 224 pixels
- **Classes:** Binary classification (No Tumor, Tumor Present)
- **Framework:** TensorFlow/Keras

### Performance
- Trained on medical brain scan datasets
- Optimized for accuracy and reliability
- Includes confidence scoring for predictions
- Grad-CAM integration for explainability

## API Endpoints

### Core Views
- `/` - Homepage with upload interface
- `/history/` - Scan history listing
- `/scan/<id>/` - Individual scan details
- `/upload/` - File upload and prediction (POST)
- `/gradcam/<id>/` - Generate Grad-CAM visualization

### Static Assets
- `/static/` - CSS, JavaScript, and image assets
- `/media/` - User-uploaded scans and generated visualizations

## Security Features

- CSRF protection on all forms
- File type validation for uploads
- Secure file handling and storage
- Input sanitization and validation
- Error handling and user feedback

## Responsive Design

### Desktop Features
- Full-width layouts with sidebar navigation
- Large image displays and detailed visualizations
- Comprehensive data tables and statistics
- Advanced hover effects and animations

### Mobile Features
- Touch-friendly interface elements
- Optimized image sizing and loading
- Collapsible navigation menu
- Simplified layouts for small screens

## Medical Disclaimer

⚠️ **Important:** This application is designed for research and educational purposes only. The AI predictions should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.

## Browser Compatibility

- **Chrome** 90+ (Recommended)
- **Firefox** 88+
- **Safari** 14+
- **Edge** 90+

## Performance Optimization

- Lazy loading for images
- Compressed static assets
- Efficient database queries
- Optimized model inference
- Progressive image enhancement

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with medical data regulations and privacy laws when using with real medical data.

## Support

For technical issues or questions:
1. Check the browser console for JavaScript errors
2. Verify all dependencies are installed correctly
3. Ensure the model file is in the correct location
4. Check Django logs for backend errors

## Future Enhancements

- Multi-class tumor type classification
- DICOM file format support
- User authentication and profiles
- Advanced analytics and reporting
- Integration with medical imaging systems
- Real-time collaboration features

---

**NeuroClassifier** - Advancing medical AI through explainable deep learning technology.

