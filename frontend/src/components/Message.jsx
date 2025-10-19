import './Message.css';

function Message({ message }) {
  return (
    <div className={`message ${message.type}`}>
      <div className="message-content">{message.content}</div>
      {message.sources && message.sources.length > 0 && (
        <div className="sources">
          <strong>Sources:</strong>
          {message.sources.map((source, index) => (
            <div key={index} className="source-item">
              <div className="source-filename">{source.filename}</div>
              <div className="source-text">
                "{source.text.substring(0, 200)}..."
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Message;
