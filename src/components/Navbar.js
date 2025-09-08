import React from 'react';

const Navbar = () => {
  return (
    <nav className="glass-effect border-b border-white/20 sticky top-0 z-50">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and App Name */}
          <div className="flex items-center space-x-4">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-secondary rounded-2xl flex items-center justify-center shadow-glow">
                <svg className="w-7 h-7 text-white" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-accent-500 rounded-full animate-pulse"></div>
            </div>
            <div>
              <h1 className="text-2xl font-bold gradient-text">
                Interactive Multimodal GPT
              </h1>
              <p className="text-xs text-neutral-500 font-medium">
                Advanced AI Reasoning Across Modalities
              </p>
            </div>
          </div>
          
          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-8">
            <div className="flex items-center space-x-3">
              <button className="btn-primary text-sm">
                <span>✨</span>
                <span>New Analysis</span>
              </button>
            </div>
          </div>
          
          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button className="p-2 text-neutral-600 hover:text-neutral-800">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
