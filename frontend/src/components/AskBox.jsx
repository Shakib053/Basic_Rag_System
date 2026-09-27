import { Send } from 'lucide-react'

function AskBox({ question, onChange, onSubmit, loading }) {
  function handleSubmit(event) {
    event.preventDefault() // stop the browser from reloading the page on submit
    onSubmit()
  }

  // Enter sends the question; Shift+Enter adds a new line.
  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onSubmit()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="card ask-box">
      <textarea
        value={question}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="e.g. What is the Salah app?"
        aria-label="Your question"
        rows={2}
        disabled={loading}
      />
      <button type="submit" className="primary" disabled={loading}>
        <Send size={16} />
        {loading ? 'Thinking…' : 'Ask'}
      </button>
    </form>
  )
}

export default AskBox
