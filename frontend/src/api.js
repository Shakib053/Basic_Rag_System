// Where the FastAPI backend runs. In dev: uvicorn on port 8000.
// The production build sets VITE_API_URL to '' because FastAPI serves the page itself.
export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export const BACKEND_DOWN = 'Could not reach the backend. Is uvicorn running on port 8000?'

export function documentFileUrl(documentId) {
  return `${API_URL}/documents/${encodeURIComponent(documentId)}/file`
}
