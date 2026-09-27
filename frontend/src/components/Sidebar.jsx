import { useEffect, useRef, useState } from 'react'
import { CloudUpload, Ellipsis, FileText, Search } from 'lucide-react'

const ACCEPTED_TYPES = '.txt,.md,.pdf,.docx,.pptx,.html,.htm,.csv,.xlsx'

function DocumentItem({ doc, cited, active, onOpen, onDelete }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef(null)

  // Close the menu when the user clicks anywhere outside it.
  useEffect(() => {
    if (!menuOpen) return
    function handleClick(event) {
      if (!menuRef.current.contains(event.target)) setMenuOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [menuOpen])

  const classes = ['doc-item', cited && 'cited', active && 'active'].filter(Boolean).join(' ')

  return (
    <li className={classes}>
      <button type="button" className="doc-open" onClick={() => onOpen(doc)} title={doc.file_name}>
        <span className={`doc-icon type-${doc.file_type}`}>
          <FileText size={18} />
        </span>
        <span className="doc-text">
          <span className="doc-name">{doc.file_name}</span>
          <span className="doc-meta">
            {doc.file_type.toUpperCase()}
            {/* The date tells apart two files with the same name. */}
            {doc.ingested_at && ` • ${new Date(doc.ingested_at).toLocaleDateString()}`}
            {cited && <span className="cited-badge">Cited</span>}
          </span>
        </span>
      </button>

      <div className="doc-menu" ref={menuRef}>
        <button
          type="button"
          className="icon-button"
          aria-label={`Actions for ${doc.file_name}`}
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          <Ellipsis size={18} />
        </button>
        {menuOpen && (
          <div className="menu" role="menu">
            <button type="button" role="menuitem" onClick={() => { setMenuOpen(false); onOpen(doc) }}>
              Open
            </button>
            <button type="button" role="menuitem" className="danger" onClick={() => { setMenuOpen(false); onDelete(doc) }}>
              Delete
            </button>
          </div>
        )}
      </div>
    </li>
  )
}

function Sidebar({
  documents,
  citedIds,
  activeId,
  uploading,
  uploadMessage,
  uploadError,
  onUpload,
  onOpen,
  onDelete,
}) {
  const [search, setSearch] = useState('')
  const fileInputRef = useRef(null)

  function handleFileChosen(event) {
    const file = event.target.files[0]
    event.target.value = '' // so choosing the same file again still triggers a change
    if (file) onUpload(file)
  }

  const visible = (documents ?? []).filter((doc) =>
    doc.file_name.toLowerCase().includes(search.trim().toLowerCase()),
  )

  return (
    <aside className="sidebar">
      <div className="sidebar-tabs">
        <span className="tab active">My Files</span>
      </div>

      <div className="upload-card">
        <CloudUpload size={32} className="upload-icon" />
        <strong>Upload Files</strong>
        <span className="muted">PDF, TXT, DOCX, etc.</span>
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_TYPES}
          onChange={handleFileChosen}
          hidden
        />
        <button
          type="button"
          className="primary"
          onClick={() => fileInputRef.current.click()}
          disabled={uploading}
        >
          {uploading ? 'Uploading…' : 'Choose File'}
        </button>
      </div>
      {uploadError && <p className="error small">{uploadError}</p>}
      {uploadMessage && <p className="muted small">{uploadMessage}</p>}

      <label className="search">
        <Search size={18} />
        <input
          type="search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search files..."
          aria-label="Search files"
        />
      </label>

      <div className="doc-list-header">
        <span>Your Documents</span>
        <span className="count">{documents?.length ?? 0}</span>
      </div>

      {documents === null ? (
        <p className="muted small">Loading…</p>
      ) : documents.length === 0 ? (
        <p className="muted small">No documents uploaded yet.</p>
      ) : visible.length === 0 ? (
        <p className="muted small">No files match “{search}”.</p>
      ) : (
        <ul className="doc-list">
          {visible.map((doc) => (
            <DocumentItem
              key={doc.document_id}
              doc={doc}
              cited={citedIds.has(doc.document_id)}
              active={doc.document_id === activeId}
              onOpen={onOpen}
              onDelete={onDelete}
            />
          ))}
        </ul>
      )}
    </aside>
  )
}

export default Sidebar
