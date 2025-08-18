import React from 'react';

const Navbar = () => {
  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and App Name */}
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">AI</span>
            </div>
            <h1 className="text-xl font-semibold text-gray-900">Interactive Multimodal GPT</h1>
          </div>
          
          {/* Navigation Links */}
          <div className="flex items-center space-x-6">
            <a href="#dashboard" className="text-gray-600 hover:text-gray-900 font-medium">
              Dashboard
            </a>
            <a href="#history" className="text-gray-600 hover:text-gray-900 font-medium">
              History
            </a>
            <a href="#settings" className="text-gray-600 hover:text-gray-900 font-medium">
              Settings
            </a>
            <button className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors">
              Get Started
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
