import React from 'react';

const ControlsSection = ({ 
  selectedMode, 
  setSelectedMode, 
  promptPreview, 
  onGenerate, 
  onClear, 
  loading 
}) => {
  const modes = [
    { value: 'tactile', label: 'Tactile-Text', description: 'Analyze tactile data with text' },
    { value: 'vision', label: 'Vision-Text', description: 'Analyze visual data with text' },
    { value: 'combined', label: 'Tactile-Vision-Text Combined', description: 'Combined multimodal analysis' }
  ];

  return (
    <div className="bg-white rounded-xl shadow-card p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Prompt Engineering & Controls</h2>
      
      <div className="space-y-6">
        {/* Mode Selection */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-3">Prompt Type Selection</h3>
          <div className="space-y-3">
            {modes.map((mode) => (
              <label key={mode.value} className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="mode"
                  value={mode.value}
                  checked={selectedMode === mode.value}
                  onChange={(e) => setSelectedMode(e.target.value)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                />
                <div>
                  <span className="text-sm font-medium text-gray-900">{mode.label}</span>
                  <p className="text-xs text-gray-500">{mode.description}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Prompt Preview */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-3">Prompt Preview</h3>
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 h-32 overflow-y-auto">
            <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono">
              {promptPreview || 'Select a mode and enter your inputs to see the prompt preview...'}
            </pre>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-4 pt-4">
          <button
            onClick={onGenerate}
            disabled={loading}
            className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Generating...</span>
              </>
            ) : (
              <>
                <span>▶</span>
                <span>Generate Response</span>
              </>
            )}
          </button>
          
          <button
            onClick={onClear}
            disabled={loading}
            className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Clear Inputs
          </button>
        </div>
      </div>
    </div>
  );
};

export default ControlsSection;
