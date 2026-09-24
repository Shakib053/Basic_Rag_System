import { useState } from 'react'
import './App.css'

// Where the FastAPI backend runs (uvicorn app.main:app --reload).
const API_URL = 'http://localhost:8000'

function App() {
  // State: values that, when changed, make React redraw the screen.
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event) {
    event.preventDefault() // stop the browser from reloading the page on submit

    if (question.trim() === '') {
      setError('Please type a question.')
      return
    }

    setLoading(true)
    setError('')
    setAnswer('')
    setSources([])

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question }),
      })

      if (response.status === 422) {
        setError('Please type a question.')
        return
      }
      if (!response.ok) {
        setError(`The server returned an error (${response.status}).`)
        return
      }

      const data = await response.json()
      setAnswer(data.answer)
      setSources(data.sources)
    } catch {
      // fetch itself failed: backend not running, or blocked by CORS
      setError('Could not reach the backend. Is uvicorn running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="container">
      <h1>Personal RAG</h1>
      <p className="subtitle">Ask a question about your documents.</p>

      <form onSubmit={handleSubmit} className="ask-form">
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="e.g. What is the Salah app?"
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Thinking…' : 'Ask'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {answer && (
        <section className="answer">
          <h2>Answer</h2>
          <p className="answer-text">{answer}</p>

          {sources.length > 0 && (
            <>
              <h2>Sources</h2>
              <ul className="sources">
                {sources.map((source, index) => (
                  <li key={index}>
                    <span className="source-name">{source.document}</span>
                    {source.page !== null && <span> · page {source.page}</span>}
                    {source.score !== null && (
                      <span className="source-score"> · score {source.score.toFixed(2)}</span>
                    )}
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}
    </main>
  )
}

export default App
