import React, { useRef, useState } from 'react';

const ImageUpload = ({ image, setImage, buttonText, placeholder, type = "image" }) => {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileUpload = (file) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleInputChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveImage = () => {
    setImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const getIcon = () => {
    if (type === "tactile") {
      return (
        <svg className="mx-auto h-12 w-12 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zM21 5a2 2 0 00-2-2h-4a2 2 0 00-2 2v12a4 4 0 004 4h4a4 4 0 004-4V5z" />
        </svg>
      );
    } else if (type === "visual") {
      return (
        <svg className="mx-auto h-12 w-12 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        </svg>
      );
    } else {
      return (
        <svg className="mx-auto h-12 w-12 text-neutral-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      );
    }
  };

  return (
    <div className="space-y-4">
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleInputChange}
        className="hidden"
      />
      
      {/* Drag and Drop Area */}
      <div 
        className={`relative border-2 border-dashed rounded-2xl h-40 transition-all duration-300 cursor-pointer group ${
          isDragOver 
            ? 'border-secondary-400 bg-secondary-50' 
            : image 
              ? 'border-neutral-200 bg-white' 
              : 'border-neutral-300 bg-neutral-50 hover:border-secondary-300 hover:bg-secondary-50'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={!image ? handleButtonClick : undefined}
      >
        {image ? (
          <div className="relative w-full h-full">
            <img
              src={image}
              alt="Uploaded"
              className="w-full h-full object-cover rounded-2xl"
            />
            <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 rounded-2xl transition-all duration-200 flex items-center justify-center">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveImage();
                }}
                className="opacity-0 group-hover:opacity-100 bg-red-500 text-white rounded-full w-8 h-8 flex items-center justify-center text-sm hover:bg-red-600 transition-all duration-200 transform hover:scale-110"
              >
                ×
              </button>
            </div>
            <div className="absolute top-3 left-3 bg-white/90 backdrop-blur-sm rounded-lg px-3 py-1">
              <span className="text-xs font-medium text-neutral-700">
                {type === "tactile" ? "Tactile" : type === "visual" ? "Visual" : "Image"} Data
              </span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full p-4 text-center">
            {getIcon()}
            <div className="mt-4 space-y-1">
              <p className="text-sm font-medium text-neutral-700">
                {isDragOver ? "Drop your image here" : placeholder}
              </p>
              <p className="text-xs text-neutral-500">
                Drag & drop or click to browse
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Upload Button */}
      {!image && (
        <button
          onClick={handleButtonClick}
          className="w-full btn-secondary text-sm"
        >
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          {buttonText}
        </button>
      )}
    </div>
  );
};

export default ImageUpload;
