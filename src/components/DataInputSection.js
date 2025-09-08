import React from 'react';
import ImageUpload from './ImageUpload';

const DataInputSection = ({ 
  tactileImage, 
  setTactileImage, 
  visualImage, 
  setVisualImage, 
  textInput, 
  setTextInput 
}) => {
  return (
    <div className="glass-effect rounded-3xl shadow-card-hover p-8 border border-white/20">
      <div className="mb-8">
        <div className="flex items-center space-x-3 mb-2">
          <div className="w-8 h-8 bg-gradient-secondary rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-neutral-800">Multimodal Data Input</h2>
        </div>
        <p className="text-neutral-600 text-lg">
          Upload your tactile and visual data, then provide your question or analysis request
        </p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
        {/* Tactile Data Input */}
        <div className="space-y-4">
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4z" />
              </svg>
            </div>
            <label className="text-lg font-semibold text-neutral-800">
              Tactile Data
            </label>
          </div>
          <p className="text-sm text-neutral-600 mb-4">
            Upload tactile sensor data or haptic feedback images
          </p>
          <ImageUpload
            image={tactileImage}
            setImage={setTactileImage}
            buttonText="Upload Tactile Data"
            placeholder="Upload tactile sensor data"
            type="tactile"
          />
        </div>

        {/* Visual Data Input */}
        <div className="space-y-4">
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 bg-gradient-to-r from-blue-500 to-cyan-600 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <label className="text-lg font-semibold text-neutral-800">
              Visual Data
            </label>
          </div>
          <p className="text-sm text-neutral-600 mb-4">
            Upload images, photos, or visual observations
          </p>
          <ImageUpload
            image={visualImage}
            setImage={setVisualImage}
            buttonText="Upload Visual Data"
            placeholder="Upload visual data"
            type="visual"
          />
        </div>

        {/* Textual Input */}
        <div className="space-y-4 lg:col-span-2 xl:col-span-1">
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </div>
            <label className="text-lg font-semibold text-neutral-800">
              Text Query
            </label>
          </div>
          <p className="text-sm text-neutral-600 mb-4">
            Describe your question or analysis requirements
          </p>
          <div className="relative">
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="What would you like me to analyze? Describe your question or requirements in detail..."
              className="input-field h-40 resize-none"
              rows={6}
            />
            <div className="absolute bottom-3 right-3 text-xs text-neutral-400">
              {textInput.length}/1000 characters
            </div>
          </div>
        </div>
      </div>

      {/* Status Indicators */}
      <div className="mt-8 flex items-center justify-center space-x-8">
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${tactileImage ? 'bg-green-400' : 'bg-neutral-300'}`}></div>
          <span className="text-sm text-neutral-600">Tactile Data</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${visualImage ? 'bg-green-400' : 'bg-neutral-300'}`}></div>
          <span className="text-sm text-neutral-600">Visual Data</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${textInput.trim() ? 'bg-green-400' : 'bg-neutral-300'}`}></div>
          <span className="text-sm text-neutral-600">Text Query</span>
        </div>
      </div>
    </div>
  );
};

export default DataInputSection;
