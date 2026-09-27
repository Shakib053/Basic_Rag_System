import { ChevronRight, CloudUpload, FileSearch, Lightbulb, MessageCircleMore, Search } from 'lucide-react'

export function Hero() {
  return (
    <div className="hero">
      <div className="hero-icon">
        <FileSearch size={56} strokeWidth={1.5} />
      </div>
      <h1>Ask anything about your documents</h1>
      <p>
        Your uploaded files are ready. Ask a question and get context-aware answers based on
        your documents.
      </p>
    </div>
  )
}

const STEPS = [
  { icon: CloudUpload, title: '1. Upload', text: 'Add your PDF, TXT or other documents.' },
  { icon: Search, title: '2. Search', text: 'We find relevant content from your files.' },
  { icon: MessageCircleMore, title: '3. Get Answer', text: 'Receive accurate, sourced responses.' },
]

export function HowItWorks() {
  return (
    <section className="card how-it-works">
      <h2>
        <span className="how-icon"><Lightbulb size={18} /></span>
        How it works
      </h2>
      <ol className="steps">
        {STEPS.map((step, index) => (
          <li key={step.title}>
            {index > 0 && <ChevronRight className="step-arrow" size={16} aria-hidden="true" />}
            <div className="step">
              <span className="step-icon"><step.icon size={22} /></span>
              <strong>{step.title}</strong>
              <span>{step.text}</span>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}
