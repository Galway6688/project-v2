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

    // 检查模式所需的图像是否已上传
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
      
      // 显示更详细的错误信息
      let errorMessage = 'Error generating response. ';
      
      if (error.response) {
        // 服务器返回了错误状态码
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
        // 请求已发出但没有收到响应
        errorMessage += 'No response from server. Please check if backend is running.';
      } else {
        // 其他错误
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
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      
      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* Data Input Section */}
        <DataInputSection
          tactileImage={tactileImage}
          setTactileImage={setTactileImage}
          visualImage={visualImage}
          setVisualImage={setVisualImage}
          textInput={textInput}
          setTextInput={setTextInput}
        />

        {/* Prompt Engineering & Controls */}
        <ControlsSection
          selectedMode={selectedMode}
          setSelectedMode={setSelectedMode}
          promptPreview={promptPreview}
          onGenerate={handleGenerate}
          onClear={handleClear}
          loading={loading}
        />

        {/* AI Response Output */}
        <ResponseSection 
          response={response}
          loading={loading}
        />
      </main>
    </div>
  );
}

export default App;
