import { useState, useRef } from 'react';
import './UploadArea.css';

function UploadArea({ onUpload, status }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onUpload(files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      onUpload(e.target.files[0]);
    }
    e.target.value = '';
  };

  return (
    <div className="upload-section">
      <div
        className={`upload-area ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <p>Drop PDF here or click to upload</p>
        <input
          type="file"
          ref={fileInputRef}
          accept=".pdf"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <button
          className="upload-btn"
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
        >
          Choose File
        </button>
      </div>
      {status.message && (
        <div className={`status ${status.type}`}>
          {status.type === 'success' ? '✓ ' : status.type === 'error' ? '✗ ' : ''}
          {status.message}
        </div>
      )}
    </div>
  );
}

export default UploadArea;
