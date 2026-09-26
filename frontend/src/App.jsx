import { useEffect, useState } from 'react'
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

  // State for uploading files and listing the ones already indexed.
  const [documents, setDocuments] = useState(null) // null = still loading
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')
  const [uploadError, setUploadError] = useState('')

  async function loadDocuments() {
    try {
      const response = await fetch(`${API_URL}/documents`)
      if (!response.ok) {
        setUploadError(`Could not load your documents (${response.status}).`)
        return
      }
      const data = await response.json()
      setDocuments(data)
    } catch {
      setUploadError('Could not reach the backend. Is uvicorn running on port 8000?')
    }
  }

  // Load the document list once, when the page first opens.
  useEffect(() => {
    loadDocuments()
  }, [])

  async function handleUpload(event) {
    event.preventDefault() // stop the browser from reloading the page on submit

    if (selectedFile === null) {
      setUploadError('Please choose a file.')
      return
    }

    setUploading(true)
    setUploadMessage('')
    setUploadError('')

    // Files are sent as "form data", not JSON. The field name must be "file".
    // Don't set a Content-Type header: the browser fills it in for form data.
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (response.status === 400) {
        // e.g. unsupported file type, empty file, or file too large
        const errorData = await response.json()
        setUploadError(errorData.detail)
        return
      }
      if (!response.ok) {
        setUploadError(`The server returned an error (${response.status}).`)
        return
      }

      const data = await response.json()
      setUploadMessage(`${data.file_name} ${data.status} (${data.chunk_count} chunks)`)
      loadDocuments() // refresh the list so the new file shows up
    } catch {
      setUploadError('Could not reach the backend. Is uvicorn running on port 8000?')
    } finally {
      setUploading(false)
    }
  }

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

      <form onSubmit={handleUpload} className="ask-form">
        <input
          type="file"
          accept=".txt,.md,.pdf,.docx,.pptx,.html,.htm,.csv,.xlsx"
          onChange={(event) => setSelectedFile(event.target.files[0] ?? null)}
          disabled={uploading}
        />
        <button type="submit" disabled={uploading}>
          {uploading ? 'Uploading…' : 'Upload'}
        </button>
      </form>

      {uploadError && <p className="error">{uploadError}</p>}
      {uploadMessage && <p className="upload-message">{uploadMessage}</p>}

      <section className="documents">
        <h2>Your documents</h2>
        {documents === null ? (
          <p className="empty">Loading…</p>
        ) : documents.length === 0 ? (
          <p className="empty">No documents uploaded yet.</p>
        ) : (
          <ul className="sources">
            {documents.map((doc) => (
              <li key={doc.document_id}>
                <span className="source-name">{doc.file_name}</span>
                <span className="source-score"> · {doc.file_type} · {doc.chunk_count} chunks</span>
              </li>
            ))}
          </ul>
        )}
      </section>

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
