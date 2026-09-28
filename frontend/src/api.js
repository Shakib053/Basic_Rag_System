// Use the local backend during development and the same origin when deployed.
export const API_URL = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? 'http://localhost:8000' : '')

export const BACKEND_DOWN = 'Could not reach the backend. Is uvicorn running on port 8000?'

export function documentFileUrl(documentId) {
  return `${API_URL}/documents/${encodeURIComponent(documentId)}/file`
}
