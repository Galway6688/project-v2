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
    <div className="bg-white rounded-xl shadow-card p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Data Input Section</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Tactile Data Input */}
        <div className="space-y-3">
          <label className="block text-sm font-medium text-gray-700">
            Tactile Data Input
          </label>
          <ImageUpload
            image={tactileImage}
            setImage={setTactileImage}
            buttonText="Upload Tactile Image"
            placeholder="No tactile image uploaded"
          />
        </div>

        {/* Visual Data Input */}
        <div className="space-y-3">
          <label className="block text-sm font-medium text-gray-700">
            Visual Data Input
          </label>
          <ImageUpload
            image={visualImage}
            setImage={setVisualImage}
            buttonText="Upload Visual Image"
            placeholder="No image uploaded"
          />
        </div>

        {/* Textual Input */}
        <div className="space-y-3">
          <label className="block text-sm font-medium text-gray-700">
            Textual Input
          </label>
          <textarea
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Enter additional text, questions, or queries here..."
            className="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={4}
          />
        </div>
      </div>
    </div>
  );
};

export default DataInputSection;
