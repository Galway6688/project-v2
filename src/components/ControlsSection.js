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
    { 
      value: 'tactile', 
      label: 'Tactile Analysis', 
      description: 'Deep analysis of tactile sensor data with intelligent text interpretation',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4z" />
        </svg>
      ),
      gradient: 'from-indigo-500 to-purple-600',
      bgGradient: 'from-indigo-50 to-purple-50'
    },
    { 
      value: 'vision', 
      label: 'Visual Analysis', 
      description: 'Advanced computer vision combined with contextual understanding',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        </svg>
      ),
      gradient: 'from-blue-500 to-cyan-600',
      bgGradient: 'from-blue-50 to-cyan-50'
    },
    { 
      value: 'combined', 
      label: 'Unified Multimodal', 
      description: 'Comprehensive analysis integrating tactile, visual, and textual data',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
      ),
      gradient: 'from-emerald-500 to-teal-600',
      bgGradient: 'from-emerald-50 to-teal-50'
    }
  ];

  return (
    <div className="glass-effect rounded-3xl shadow-card-hover p-8 border border-white/20">
      <div className="mb-8">
        <div className="flex items-center space-x-3 mb-2">
          <div className="w-8 h-8 bg-gradient-secondary rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-neutral-800">Analysis Configuration</h2>
        </div>
        <p className="text-neutral-600 text-lg">
          Choose your analysis mode and review the generated prompt before processing
        </p>
      </div>
      
      <div className="space-y-8">
        {/* Mode Selection Cards */}
        <div>
          <h3 className="text-lg font-semibold text-neutral-800 mb-6">Select Analysis Mode</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {modes.map((mode) => (
              <div
                key={mode.value}
                className={`relative cursor-pointer transition-all duration-300 transform hover:scale-105 ${
                  selectedMode === mode.value 
                    ? 'ring-4 ring-secondary-200 shadow-glow' 
                    : 'hover:shadow-card-hover'
                }`}
                onClick={() => setSelectedMode(mode.value)}
              >
                <div className={`p-6 rounded-2xl border-2 transition-all duration-300 ${
                  selectedMode === mode.value
                    ? `border-transparent bg-gradient-to-br ${mode.bgGradient}`
                    : 'border-neutral-200 bg-white hover:border-neutral-300'
                }`}>
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${
                    selectedMode === mode.value 
                      ? `bg-gradient-to-r ${mode.gradient} text-white` 
                      : 'bg-neutral-100 text-neutral-600'
                  }`}>
                    {mode.icon}
                  </div>
                  <h4 className="text-lg font-bold text-neutral-800 mb-2">{mode.label}</h4>
                  <p className="text-sm text-neutral-600 leading-relaxed">{mode.description}</p>
                  
                  {selectedMode === mode.value && (
                    <div className="absolute top-4 right-4">
                      <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                        <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Prompt Preview */}
        <div>
          <h3 className="text-lg font-semibold text-neutral-800 mb-4">Generated Prompt Preview</h3>
          <div className="bg-neutral-50 border border-neutral-200 rounded-2xl p-6 h-40 overflow-y-auto">
            <pre className="text-sm text-neutral-700 whitespace-pre-wrap font-mono leading-relaxed">
              {promptPreview || 'Configure your inputs and select an analysis mode to see the generated prompt...'}
            </pre>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 pt-6">
          <button
            onClick={onGenerate}
            disabled={loading}
            className="flex-1 relative overflow-hidden group disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <div className={`btn-primary w-full text-lg py-4 ${loading ? 'animate-pulse' : ''}`}>
              {loading ? (
                <div className="flex items-center justify-center space-x-3">
                  <svg className="animate-spin h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Analyzing Data...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-3">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Generate AI Analysis</span>
                  <div className="w-2 h-2 bg-white/60 rounded-full animate-ping"></div>
                </div>
              )}
            </div>
            {!loading && (
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 transform translate-x-[-100%] group-hover:translate-x-[100%] skew-x-12"></div>
            )}
          </button>
          
          <button
            onClick={onClear}
            disabled={loading}
            className="sm:w-auto btn-secondary text-lg py-4 px-8"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Clear All
          </button>
        </div>
      </div>
    </div>
  );
};

export default ControlsSection;
