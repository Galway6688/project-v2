# Interactive Multimodal GPT

A full-stack web application for multimodal AI reasoning that combines tactile sensor data, visual images, and text to analyze material properties.

## Overview

This application lets users upload tactile and visual images along with text questions to get AI-powered analysis of material textures and properties. Built with React frontend and FastAPI backend, using LangGraph for intelligent query processing.

### Features

- **Three Analysis Modes**:
  - Tactile-only analysis from sensor data
  - Vision-only analysis from photographs  
  - Combined tactile-vision analysis
- **Smart Query Processing**: Automatically optimizes user questions for better results
- **Few-Shot Learning**: Uses example images to improve AI responses
- **Real-time Preview**: Shows the exact prompt sent to AI
- **Modern Interface**: Clean React UI with drag-and-drop image uploads

## Architecture

### Frontend
- React 18 with Tailwind CSS
- Component-based UI with image upload, mode selection, and response display
- Axios for API communication

### Backend
- FastAPI server with CORS support
- LangGraph workflow with three stages:
  1. Query optimization based on selected mode
  2. Prompt building with few-shot examples
  3. LLM call to Together AI
- Specialized prompt templates for each analysis mode

### AI Model
- **Provider**: Together AI
- **Model**: meta-llama/Llama-4-Scout-17B-16E-Instruct
- **Capabilities**: Vision and text multimodal reasoning

## Quick Start

### Requirements

- Node.js 16+
- Python 3.8+

### Setup

1. **Install frontend dependencies**
   ```bash
   npm install
   ```

2. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Start both servers**
   ```bash
   # Option 1: Use the provided scripts
   ./start-dev.sh    # Linux/Mac
   start-dev.bat     # Windows
   
   # Option 2: Start manually
   npm start                    # Frontend (port 3000)
   python start_backend.py      # Backend (port 8000)
   ```

### Configuration

The app comes with pre-configured Together AI settings. To use your own API key, go to the config.py file:
and add the api key in the config.py
```
TOGETHER_API_KEY=your_api_key_here
TOGETHER_MODEL_NAME=meta-llama/Llama-4-Scout-17B-16E-Instruct
TOGETHER_BASE_URL=https://api.together.xyz/v1
```

## Usage

1. **Upload Images**: Drag and drop or click to upload tactile sensor data and/or visual images
2. **Enter Question**: Type your question about the material properties
3. **Select Mode**: Choose tactile-only, vision-only, or combined analysis
4. **Preview**: Review the generated prompt in the preview section
5. **Analyze**: Click "Generate AI Analysis" to get results

The AI will optimize your question and provide comma-separated tactile property descriptions.

## Project Structure

```
projectagent/
├── src/                          # React frontend
│   ├── components/               # UI components
│   │   ├── DataInputSection.js   # Image upload and text input
│   │   ├── ControlsSection.js    # Mode selection and preview
│   │   ├── ResponseSection.js    # AI response display
│   │   ├── ImageUpload.js        # Drag-drop image component
│   │   └── Navbar.js            # Navigation header
│   ├── App.js                   # Main application
│   └── index.js                 # React entry point
├── backend/
│   ├── main.py                  # FastAPI server
│   ├── agent.py                 # LangGraph multimodal agent
│   ├── config.py                # Configuration settings
│   ├── prompt_templates.py      # AI prompt templates
│   ├── evaluation*.py           # Model evaluation scripts
│   ├── run_*.py                 # Analysis and prediction scripts
│   ├── noisy_questions.py       # Test question variations
│   ├── images_rgb/              # Example visual images
│   ├── images_tac/              # Example tactile images
│   └── test*.csv               # Test datasets
└── start_backend.py             # Backend startup script
```

## API Endpoints

- `POST /api/reasoning` - Main analysis endpoint
- `GET /api/modes` - Available analysis modes  
- `GET /api/model-info` - AI model information
- `GET /health` - Health check

## Testing & Evaluation

The backend includes evaluation scripts for model performance testing:

- `evaluation.py` / `evaluation_new.py` - Model performance evaluation
- `run_predictions.py` - Batch prediction generation  
- `run_analysis.py` - Results analysis and visualization
- Test datasets and example images for few-shot learning

## Troubleshooting

**CORS Errors**: Make sure both frontend (port 3000) and backend (port 8000) are running

**API Issues**: Check that your Together AI API key is valid and has sufficient quota

**Image Problems**: Only JPEG and PNG formats are supported for uploads

**Startup Issues**: Use the provided startup scripts or check that all dependencies are installed


