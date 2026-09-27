import { BookOpen, FileText } from 'lucide-react'

// Same pattern as retrieval/result.py: [S1] or [S1, page 2].
const CITATION_PATTERN = /\[(S\d+)(?:,[^\]]*)?\]/g

const MODE_LABELS = {
  grounded: 'From your documents',
  general: 'General knowledge, not from your files',
  error: 'Something went wrong',
}

// Where in the file a citation points, e.g. "page 4" or "slide 2".
// Plain text files have no place inside them (their locator is the server path), so show nothing.
function citationPlace(citation) {
  if (citation.page !== null) return `page ${citation.page}`
  const locator = citation.locator
  if (!locator || locator === 'document' || /[/\\]/.test(locator)) return ''
  return locator
}

// Split the answer text into plain strings and clickable [S1] chips.
function AnswerText({ text, citations, onOpenCitation }) {
  const byId = new Map(citations.map((citation) => [citation.id, citation]))
  const parts = []
  let last = 0
  for (const match of text.matchAll(CITATION_PATTERN)) {
    parts.push(text.slice(last, match.index))
    const citation = byId.get(match[1])
    if (citation) {
      const place = citationPlace(citation)
      parts.push(
        <button
          key={match.index}
          type="button"
          className="citation-chip"
          title={place ? `${citation.document} · ${place}` : citation.document}
          onClick={() => onOpenCitation(citation)}
        >
          {citation.id.slice(1)}
        </button>,
      )
    }
    last = match.index + match[0].length
  }
  parts.push(text.slice(last))
  return <p className="answer-text">{parts}</p>
}

function AnswerView({ result, onOpenCitation }) {
  // One source row per file and page; several [S] ids can point at the same place.
  const sources = []
  for (const citation of result.citations) {
    const existing = sources.find(
      (source) => source.document_id === citation.document_id && source.page === citation.page,
    )
    if (existing) {
      existing.ids.push(citation.id)
    } else {
      sources.push({ ...citation, ids: [citation.id] })
    }
  }

  return (
    <section className="card answer">
      <p className="answer-question">{result.question}</p>
      <span className={`mode-badge mode-${result.mode}`}>{MODE_LABELS[result.mode] ?? result.mode}</span>

      <AnswerText text={result.answer} citations={result.citations} onOpenCitation={onOpenCitation} />

      {sources.length > 0 && (
        <>
          <h2 className="section-title">
            <BookOpen size={16} /> Sources
          </h2>
          <ul className="source-list">
            {sources.map((source) => {
              const place = citationPlace(source)
              return (
                <li key={source.ids[0]}>
                  <button type="button" onClick={() => onOpenCitation(source)}>
                    <span className="source-ids">{source.ids.map((id) => id.slice(1)).join(', ')}</span>
                    <FileText size={16} className="source-icon" />
                    <span className="source-name">{source.document}</span>
                    {place && <span className="muted"> · {place}</span>}
                  </button>
                </li>
              )
            })}
          </ul>
        </>
      )}
    </section>
  )
}

export default AnswerView
