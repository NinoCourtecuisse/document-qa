import { useState } from 'react';
import UploadArea from './UploadArea';
import './Sidebar.css';

function Sidebar({ documents, selectedReader, onReaderChange, onUpload, onClearDatabase }) {
  const [status, setStatus] = useState({ message: '', type: '' });

  const handleUpload = async (file) => {
    setStatus({ message: 'Uploading...', type: 'info' });
    const result = await onUpload(file);
    setStatus({ message: result.message, type: result.success ? 'success' : 'error' });
    setTimeout(() => setStatus({ message: '', type: '' }), 3000);
  };

  const handleClearDatabase = async () => {
    setStatus({ message: 'Clearing database...', type: 'info' });
    const result = await onClearDatabase();
    setStatus({ message: result.message, type: result.success ? 'success' : 'error' });
    setTimeout(() => setStatus({ message: '', type: '' }), 3000);
  };

  return (
    <div className="sidebar">
      <h1>Documents</h1>

      <div className="controls-section">
        <div className="control-group">
          <label htmlFor="readerSelect">PDF Reader</label>
          <select
            id="readerSelect"
            value={selectedReader}
            onChange={(e) => onReaderChange(e.target.value)}
          >
            <option value="pdf_reader">PDF Reader</option>
            <option value="docling">Docling</option>
            <option value="llama_parse">LlamaParse</option>
          </select>
        </div>
        <div className="control-group">
          <button className="clear-btn" onClick={handleClearDatabase}>
            Clear Database
          </button>
        </div>
      </div>

      <UploadArea onUpload={handleUpload} status={status} />

      <h2>Uploaded Files</h2>
      <div className="documents-list">
        {documents.length === 0 ? (
          <div className="empty-state">No documents yet</div>
        ) : (
          documents.map((doc, index) => (
            <div key={index} className="document-item">
              {doc.filename}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default Sidebar;
