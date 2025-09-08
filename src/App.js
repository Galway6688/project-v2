import React, { useState, useCallback } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import DataInputSection from './components/DataInputSection';
import ControlsSection from './components/ControlsSection';
import ResponseSection from './components/ResponseSection';

function App() {
  const [tactileImage, setTactileImage] = useState(null);
  const [visualImage, setVisualImage] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [selectedMode, setSelectedMode] = useState('tactile');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [promptPreview, setPromptPreview] = useState('');
  const [optimizedQuestion, setOptimizedQuestion] = useState(null);

  const updatePromptPreview = useCallback(() => {
    let preview = '';
    
    if (selectedMode === 'tactile') {
      preview = `Mode: Tactile-Text Analysis\n\nQuestion: ${textInput || '[Your question here]'}\n\nTactile Image: ${tactileImage ? '[Tactile image uploaded]' : '[No tactile image]'}`;
    } else if (selectedMode === 'vision') {
      preview = `Mode: Vision-Text Analysis\n\nQuestion: ${textInput || '[Your question here]'}\n\nVisual Image: ${visualImage ? '[Visual image uploaded]' : '[No visual image]'}`;
    } else if (selectedMode === 'combined') {
      preview = `Mode: Tactile-Vision-Text Combined Analysis\n\nQuestion: ${textInput || '[Your question here]'}\n\nTactile Image: ${tactileImage ? '[Tactile image uploaded]' : '[No tactile image]'}\nVisual Image: ${visualImage ? '[Visual image uploaded]' : '[No visual image]'}`;
    }
    
    setPromptPreview(preview);
  }, [selectedMode, textInput, tactileImage, visualImage]);

  // Update prompt preview whenever inputs change
  React.useEffect(() => {
    updatePromptPreview();
  }, [updatePromptPreview]);

  const handleGenerate = async () => {
    if (!textInput.trim()) {
      alert('Please enter a question.');
      return;
    }

    // Check if required images for the mode have been uploaded
    if (selectedMode === 'tactile' && !tactileImage) {
      alert('Tactile mode requires a tactile image. Please upload one first.');
      return;
    }
    
    if (selectedMode === 'vision' && !visualImage) {
      alert('Vision mode requires a visual image. Please upload one first.');
      return;
    }
    
    if (selectedMode === 'combined' && !tactileImage && !visualImage) {
      alert('Combined mode requires at least one image (tactile or visual). Please upload at least one image.');
      return;
    }

    setLoading(true);
    setResponse(null);

    try {
      const requestData = {
        question: textInput,
        mode: selectedMode,
        tactile_image: tactileImage,
        vision_image: visualImage
      };

      console.log('🚀 Sending request to backend:', {
        url: 'http://localhost:8000/api/reasoning',
        data: requestData,
        mode: selectedMode
      });

      const result = await axios.post('http://localhost:8000/api/reasoning', requestData, {
        headers: {
          'Content-Type': 'application/json',
        }
      });

      console.log('✅ Backend response received:', result.data);
      setResponse(result.data.response);
      setOptimizedQuestion(result.data.optimized_question);
    } catch (error) {
      console.error('❌ Error generating response:', error);
      console.error('Error details:', {
        message: error.message,
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        config: {
          url: error.config?.url,
          method: error.config?.method,
          headers: error.config?.headers
        }
      });
      
      // Display more detailed error information
      let errorMessage = 'Error generating response. ';
      
      if (error.response) {
        // Server returned error status code
        if (error.response.status === 400) {
          errorMessage += error.response.data.detail || 'Invalid request data.';
        } else if (error.response.status === 404) {
          errorMessage += 'API endpoint not found. Please check if backend is running and has the correct routes.';
        } else if (error.response.status === 500) {
          errorMessage += 'Server error occurred.';
        } else {
          errorMessage += `Server error (${error.response.status}): ${error.response.statusText || 'Unknown error'}`;
        }
      } else if (error.request) {
        // Request was sent but no response received
        errorMessage += 'No response from server. Please check if backend is running.';
      } else {
        // Other errors
        errorMessage += error.message || 'Unknown error occurred.';
      }
      
      setResponse(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setTactileImage(null);
    setVisualImage(null);
    setTextInput('');
    setSelectedMode('tactile');
    setResponse(null);
    setPromptPreview('');
    setOptimizedQuestion(null);
  };

  return (
    <div className="min-h-screen bg-gradient-surface">
      <Navbar />
      
      <main className="container mx-auto px-6 py-12 space-y-12 max-w-7xl">
        {/* Hero Section */}
        <div className="text-center space-y-4 mb-16">
          <h1 className="text-4xl md:text-5xl font-bold gradient-text">
            AI-Powered Multimodal Analysis
          </h1>
          <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed">
            Experience cutting-edge AI that understands tactile, visual, and textual data to provide 
            intelligent insights and comprehensive analysis.
          </p>
        </div>

        {/* Data Input Section */}
        <div className="animate-fade-in">
          <DataInputSection
            tactileImage={tactileImage}
            setTactileImage={setTactileImage}
            visualImage={visualImage}
            setVisualImage={setVisualImage}
            textInput={textInput}
            setTextInput={setTextInput}
          />
        </div>

        {/* Prompt Engineering & Controls */}
        <div className="animate-fade-in">
          <ControlsSection
            selectedMode={selectedMode}
            setSelectedMode={setSelectedMode}
            promptPreview={promptPreview}
            onGenerate={handleGenerate}
            onClear={handleClear}
            loading={loading}
          />
        </div>

        {/* AI Response Output */}
        <div className="animate-fade-in">
          <ResponseSection 
            response={response}
            loading={loading}
            optimizedQuestion={optimizedQuestion}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
