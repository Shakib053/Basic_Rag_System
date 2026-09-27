import { useEffect, useState } from 'react'
import './App.css'
import { API_URL, BACKEND_DOWN } from './api.js'
import Sidebar from './components/Sidebar.jsx'
import AskBox from './components/AskBox.jsx'
import { Hero, HowItWorks } from './components/EmptyState.jsx'
import AnswerView from './components/AnswerView.jsx'
import DocumentViewer from './components/DocumentViewer.jsx'

function App() {
  // State: values that, when changed, make React redraw the screen.
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null) // { question, answer, mode, citations }
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // State for uploading files and listing the ones already indexed.
  const [documents, setDocuments] = useState(null) // null = still loading
  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')
  const [uploadError, setUploadError] = useState('')

  // The document open in the viewer: { document_id, file_name, file_type, page }, or null.
  const [viewing, setViewing] = useState(null)

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
      setUploadError(BACKEND_DOWN)
    }
  }

  // Load the document list once, when the page first opens.
  useEffect(() => {
    loadDocuments()
  }, [])

  async function handleUpload(file) {
    setUploading(true)
    setUploadMessage('')
    setUploadError('')

    // Files are sent as "form data", not JSON. The field name must be "file".
    // Don't set a Content-Type header: the browser fills it in for form data.
    const formData = new FormData()
    formData.append('file', file)

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
      if (data.status === 'duplicate') {
        setUploadMessage(`Already uploaded as ${data.file_name}, nothing new was added.`)
      } else {
        setUploadMessage(`${data.file_name} ${data.status} (${data.chunk_count} chunks)`)
      }
      loadDocuments() // refresh the list so the new file shows up
    } catch {
      setUploadError(BACKEND_DOWN)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(doc) {
    if (!window.confirm(`Delete "${doc.file_name}"? It will no longer be searched.`)) {
      return
    }

    setUploadMessage('')
    setUploadError('')
    try {
      const response = await fetch(`${API_URL}/documents/${encodeURIComponent(doc.document_id)}`, {
        method: 'DELETE',
      })
      if (!response.ok && response.status !== 404) {
        setUploadError(`Could not delete ${doc.file_name} (${response.status}).`)
        return
      }
      setUploadMessage(`${doc.file_name} deleted.`)
      if (viewing?.document_id === doc.document_id) {
        setViewing(null)
      }
      loadDocuments()
    } catch {
      setUploadError(BACKEND_DOWN)
    }
  }

  async function handleAsk() {
    if (question.trim() === '') {
      setError('Please type a question.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    setViewing(null)

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
      setResult({ question, answer: data.answer, mode: data.mode, citations: data.citations })
    } catch {
      // fetch itself failed: backend not running, or blocked by CORS
      setError(BACKEND_DOWN)
    } finally {
      setLoading(false)
    }
  }

  // Open a document from the sidebar (page = null) or from a citation (page = cited page).
  function openDocument(documentId, fileName, page) {
    const doc = documents?.find((d) => d.document_id === documentId)
    const fileType = doc?.file_type ?? fileName.split('.').pop().toLowerCase()
    setViewing({ document_id: documentId, file_name: fileName, file_type: fileType, page })
  }

  // Documents the current answer was taken from, highlighted in the sidebar.
  const citedIds = new Set(result?.citations.map((citation) => citation.document_id) ?? [])

  return (
    <div className="layout">
      <Sidebar
        documents={documents}
        citedIds={citedIds}
        activeId={viewing?.document_id}
        uploading={uploading}
        uploadMessage={uploadMessage}
        uploadError={uploadError}
        onUpload={handleUpload}
        onOpen={(doc) => openDocument(doc.document_id, doc.file_name, null)}
        onDelete={handleDelete}
      />

      <main className="main-panel">
        {viewing ? (
          <DocumentViewer
            key={`${viewing.document_id}-${viewing.page}`}
            doc={viewing}
            backLabel={result ? 'Back to answer' : 'Back'}
            onBack={() => setViewing(null)}
          />
        ) : (
          <>
            {!result && !loading && <Hero />}
            <AskBox
              question={question}
              onChange={setQuestion}
              onSubmit={handleAsk}
              loading={loading}
            />
            {error && <p className="error">{error}</p>}
            {loading && <p className="thinking">Searching your documents…</p>}
            {result && (
              <AnswerView
                result={result}
                onOpenCitation={(citation) =>
                  openDocument(citation.document_id, citation.document, citation.page)
                }
              />
            )}
            {!result && !loading && <HowItWorks />}
          </>
        )}
      </main>
    </div>
  )
}

export default App
