import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [selectedReader, setSelectedReader] = useState('docling');

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/documents`);
      const data = await response.json();
      setDocuments(data.documents);
    } catch (error) {
      console.error('Error loading documents:', error);
    }
  };

  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('reader', selectedReader);

    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData
    });

    const result = await response.json();

    if (response.ok) {
      await loadDocuments();
      return {
        success: true,
        message: `${result.filename} uploaded successfully! (${result.vectors_created} chunks, using ${result.reader_used})`
      };
    } else {
      return {
        success: false,
        message: result.detail
      };
    }
  };

  const handleClearDatabase = async () => {
    if (!confirm('Are you sure you want to clear the database? This will delete all uploaded documents and cannot be undone.')) {
      return;
    }

    const response = await fetch(`${API_URL}/clear`, {
      method: 'DELETE'
    });

    const result = await response.json();

    if (response.ok) {
      await loadDocuments();
      setMessages([]);
      return {
        success: true,
        message: `Database cleared successfully! (${result.vectors_cleared} vectors, ${result.files_deleted} files deleted)`
      };
    } else {
      return {
        success: false,
        message: result.detail
      };
    }
  };

  const handleAskQuestion = async (question) => {
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ question })
    });

    const result = await response.json();

    if (response.ok) {
      return {
        success: true,
        answer: result.answer,
        sources: result.sources
      };
    } else {
      return {
        success: false,
        message: result.detail
      };
    }
  };

  return (
    <div className="app-container">
      <Sidebar
        documents={documents}
        selectedReader={selectedReader}
        onReaderChange={setSelectedReader}
        onUpload={handleUpload}
        onClearDatabase={handleClearDatabase}
      />
      <ChatInterface
        messages={messages}
        setMessages={setMessages}
        onAskQuestion={handleAskQuestion}
      />
    </div>
  );
}

export default App;
