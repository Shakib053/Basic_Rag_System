import { useEffect, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import { ArrowLeft, ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react'
import { BACKEND_DOWN, documentFileUrl } from '../api.js'

// PDF.js parses PDFs in a web worker; it must be set in the module that renders <Document>.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

const TEXT_TYPES = ['txt', 'md', 'csv']

// Width of an element, kept up to date when the window is resized.
function useWidth(ref) {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [ref])
  return width
}

function PdfView({ file, initialPage }) {
  const [pageNumber, setPageNumber] = useState(initialPage ?? 1)
  const [numPages, setNumPages] = useState(null)
  const containerRef = useRef(null)
  const width = useWidth(containerRef)

  return (
    <>
      <div className="pdf-toolbar">
        <button type="button" onClick={() => setPageNumber(pageNumber - 1)} disabled={pageNumber <= 1} aria-label="Previous page">
          <ChevronLeft size={18} />
        </button>
        <span>
          Page {pageNumber}
          {numPages && ` of ${numPages}`}
        </span>
        <button
          type="button"
          onClick={() => setPageNumber(pageNumber + 1)}
          disabled={numPages === null || pageNumber >= numPages}
          aria-label="Next page"
        >
          <ChevronRight size={18} />
        </button>
      </div>
      <div className="pdf-page" ref={containerRef}>
        <Document
          file={file}
          suspense={false}
          loading={<p className="muted">Loading PDF…</p>}
          error={<p className="error">Could not display this PDF.</p>}
          onLoadSuccess={(pdf) => {
            setNumPages(pdf.numPages)
            setPageNumber((page) => Math.min(page, pdf.numPages))
          }}
        >
          {width > 0 && <Page pageNumber={pageNumber} width={Math.min(width, 900)} suspense={false} />}
        </Document>
      </div>
    </>
  )
}

function DocumentViewer({ doc, backLabel, onBack }) {
  const isPdf = doc.file_type === 'pdf'
  const isText = TEXT_TYPES.includes(doc.file_type)
  const [content, setContent] = useState(null) // Blob for PDFs, string for text files
  const [error, setError] = useState('')

  useEffect(() => {
    if (!isPdf && !isText) return
    let cancelled = false
    async function load() {
      try {
        const response = await fetch(documentFileUrl(doc.document_id))
        if (response.status === 404) {
          const data = await response.json()
          if (!cancelled) setError(data.detail)
          return
        }
        if (!response.ok) {
          if (!cancelled) setError(`The server returned an error (${response.status}).`)
          return
        }
        const body = isPdf ? await response.blob() : await response.text()
        if (!cancelled) setContent(body)
      } catch {
        if (!cancelled) setError(BACKEND_DOWN)
      }
    }
    load()
    // Ignore a response that arrives after the viewer was closed.
    return () => {
      cancelled = true
    }
  }, [doc.document_id, isPdf, isText])

  let body
  if (error) {
    body = <p className="error">{error}</p>
  } else if (!isPdf && !isText) {
    body = <p className="muted">A preview is not available for .{doc.file_type} files. Use “Open file” to view it.</p>
  } else if (content === null) {
    body = <p className="muted">Loading…</p>
  } else if (isPdf) {
    body = <PdfView file={content} initialPage={doc.page} />
  } else {
    body = <pre className="text-view">{content}</pre>
  }

  return (
    <section className="viewer">
      <header className="viewer-header">
        <button type="button" className="link-button" onClick={onBack}>
          <ArrowLeft size={16} /> {backLabel}
        </button>
        <h2 title={doc.file_name}>{doc.file_name}</h2>
        <a href={documentFileUrl(doc.document_id)} target="_blank" rel="noreferrer" className="link-button">
          Open file <ExternalLink size={14} />
        </a>
      </header>
      <div className="viewer-body">{body}</div>
    </section>
  )
}

export default DocumentViewer
