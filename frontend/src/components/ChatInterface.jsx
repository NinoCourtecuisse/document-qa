import { useState, useRef, useEffect } from 'react';
import Message from './Message';
import './ChatInterface.css';

function ChatInterface({ messages, setMessages, onAskQuestion }) {
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleAskQuestion = async () => {
    if (!question.trim() || isLoading) return;

    const userMessage = { type: 'user', content: question };
    setMessages([...messages, userMessage]);
    setQuestion('');
    setIsLoading(true);

    const result = await onAskQuestion(question);
    setIsLoading(false);

    if (result.success) {
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: result.answer,
        sources: result.sources
      }]);
    } else {
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: `Error: ${result.message}`,
        error: true
      }]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAskQuestion();
    }
  };

  return (
    <div className="chat-interface">
      <h1>Chat with Your Documents</h1>
      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty-state">Upload a document and ask questions about it!</div>
          ) : (
            messages.map((message, index) => (
              <Message key={index} message={message} />
            ))
          )}
          {isLoading && (
            <div className="thinking">
              <span>Thinking</span>
              <div className="thinking-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="input-area">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your documents..."
            disabled={isLoading}
          />
          <button onClick={handleAskQuestion} disabled={isLoading || !question.trim()}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
