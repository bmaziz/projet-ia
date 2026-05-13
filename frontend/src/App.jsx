import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

const WELCOME = 'Bonjour ! 👋 Je suis votre assistant médical spécialisé en pharmacologie. Posez-moi une question sur un médicament, ses effets, sa posologie ou ses interactions.'

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
}

function Avatar({ role }) {
  return (
    <div className={`avatar ${role}`}>
      {role === 'assistant' ? '💊' : '👤'}
    </div>
  )
}

function Message({ msg, onSpeak, speaking, speakingId }) {
  const isAssistant = msg.role === 'assistant'
  const isSpeaking = speaking && speakingId === msg.id

  return (
    <div className={`message ${msg.role}`}>
      {isAssistant && <Avatar role="assistant" />}
      <div className="msg-content">
        <div className={`bubble ${msg.role}`}>
          {msg.text.split('\n').map((line, i) => (
            <span key={i}>{line}{i < msg.text.split('\n').length - 1 && <br />}</span>
          ))}
          {msg.pdfId && (
            <a
              className="pdf-btn"
              href={`/pdf/${msg.pdfId}`}
              target="_blank"
              rel="noopener noreferrer"
              download
            >
              📄 Télécharger le PDF
            </a>
          )}
        </div>
        <div className="msg-meta">
          <span className="timestamp">{formatTime(msg.ts)}</span>
          {isAssistant && (
            <button
              className={`speak-btn ${isSpeaking ? 'active' : ''}`}
              onClick={() => onSpeak(msg)}
              title={isSpeaking ? 'Arrêter' : 'Écouter'}
            >
              {isSpeaking ? '⏹' : '🔊'}
            </button>
          )}
        </div>
      </div>
      {!isAssistant && <Avatar role="user" />}
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <Avatar role="assistant" />
      <div className="msg-content">
        <div className="bubble assistant typing">
          <span /><span /><span />
        </div>
      </div>
    </div>
  )
}

function Sidebar({ conversations, activeId, onSelect, onCreate, onDelete, onRename, collapsed, onToggle }) {
  const [editingId, setEditingId] = useState(null)
  const [editVal, setEditVal] = useState('')

  function startEdit(e, conv) {
    e.stopPropagation()
    setEditingId(conv.id)
    setEditVal(conv.title)
  }

  function commitEdit(id) {
    if (editVal.trim()) onRename(id, editVal.trim())
    setEditingId(null)
  }

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        {!collapsed && <span className="sidebar-title">💬 Historique</span>}
        <button className="icon-btn" onClick={onToggle} title="Réduire">
          {collapsed ? '▶' : '◀'}
        </button>
      </div>

      {!collapsed && (
        <>
          <button className="new-chat-btn" onClick={onCreate}>
            ✚ Nouvelle conversation
          </button>
          <div className="conv-list">
            {conversations.map(conv => (
              <div
                key={conv.id}
                className={`conv-item ${conv.id === activeId ? 'active' : ''}`}
                onClick={() => onSelect(conv.id)}
              >
                {editingId === conv.id ? (
                  <input
                    className="rename-input"
                    value={editVal}
                    autoFocus
                    onChange={e => setEditVal(e.target.value)}
                    onBlur={() => commitEdit(conv.id)}
                    onKeyDown={e => e.key === 'Enter' && commitEdit(conv.id)}
                    onClick={e => e.stopPropagation()}
                  />
                ) : (
                  <>
                    <span className="conv-title">{conv.title}</span>
                    <div className="conv-actions">
                      <button className="icon-btn small" onClick={e => startEdit(e, conv)} title="Renommer">✏️</button>
                      <button className="icon-btn small danger" onClick={e => { e.stopPropagation(); onDelete(conv.id) }} title="Supprimer">🗑</button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </aside>
  )
}

function newConv(title = 'Nouvelle conversation') {
  return {
    id: Date.now().toString(),
    title,
    messages: [{ id: 'w', role: 'assistant', text: WELCOME, ts: Date.now() }],
  }
}

export default function App() {
  const [conversations, setConversations] = useState(() => {
    try {
      const saved = localStorage.getItem('rag_conversations')
      return saved ? JSON.parse(saved) : [newConv('Conversation 1')]
    } catch { return [newConv('Conversation 1')] }
  })
  const [activeId, setActiveId] = useState(() => conversations[0]?.id)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [darkMode, setDarkMode] = useState(false)
  const [listening, setListening] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [speakingId, setSpeakingId] = useState(null)
  const lastMedIdRef = useRef(null)

  const bottomRef = useRef(null)
  const recognitionRef = useRef(null)
  const inputRef = useRef(null)

  const activeConv = conversations.find(c => c.id === activeId)

  useEffect(() => {
    localStorage.setItem('rag_conversations', JSON.stringify(conversations))
  }, [conversations])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeConv?.messages, loading])

  useEffect(() => {
    document.body.dataset.theme = darkMode ? 'dark' : 'light'
  }, [darkMode])

  // Speech Recognition setup
  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) return
    const rec = new SR()
    rec.lang = 'fr-FR'
    rec.continuous = false
    rec.interimResults = false
    rec.onresult = e => {
      const transcript = e.results[0][0].transcript
      setInput(prev => prev + transcript)
      setListening(false)
    }
    rec.onerror = () => setListening(false)
    rec.onend = () => setListening(false)
    recognitionRef.current = rec
  }, [])

  function updateConv(id, updater) {
    setConversations(prev => prev.map(c => c.id === id ? updater(c) : c))
  }

  function createConv() {
    const conv = newConv(`Conversation ${conversations.length + 1}`)
    setConversations(prev => [conv, ...prev])
    setActiveId(conv.id)
  }

  function deleteConv(id) {
    const remaining = conversations.filter(c => c.id !== id)
    if (remaining.length === 0) {
      const fresh = newConv('Conversation 1')
      setConversations([fresh])
      setActiveId(fresh.id)
    } else {
      setConversations(remaining)
      if (activeId === id) setActiveId(remaining[0].id)
    }
  }

  function renameConv(id, title) {
    updateConv(id, c => ({ ...c, title }))
  }

  async function sendMessage(e) {
    e?.preventDefault()
    const question = input.trim()
    if (!question || loading) return

    const userMsg = { id: Date.now().toString(), role: 'user', text: question, ts: Date.now() }
    updateConv(activeId, c => ({
      ...c,
      title: c.messages.length === 1 ? question.slice(0, 35) + (question.length > 35 ? '…' : '') : c.title,
      messages: [...c.messages, userMsg],
    }))
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, last_med_id: lastMedIdRef.current }),
      })
      const data = await res.json()
      if (data.last_med_id) lastMedIdRef.current = data.last_med_id
      const assistantMsg = { id: Date.now().toString(), role: 'assistant', text: data.answer, pdfId: data.pdf_id || null, ts: Date.now() }
      updateConv(activeId, c => ({ ...c, messages: [...c.messages, assistantMsg] }))
    } catch {
      const errMsg = { id: Date.now().toString(), role: 'assistant', text: '❌ Erreur de connexion au serveur.', ts: Date.now() }
      updateConv(activeId, c => ({ ...c, messages: [...c.messages, errMsg] }))
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function toggleListen() {
    if (!recognitionRef.current) return alert('Reconnaissance vocale non supportée par ce navigateur.')
    if (listening) {
      recognitionRef.current.stop()
      setListening(false)
    } else {
      recognitionRef.current.start()
      setListening(true)
    }
  }

  const handleSpeak = useCallback((msg) => {
    if (!window.speechSynthesis) return
    if (speaking && speakingId === msg.id) {
      window.speechSynthesis.cancel()
      setSpeaking(false)
      setSpeakingId(null)
      return
    }
    window.speechSynthesis.cancel()
    const utt = new SpeechSynthesisUtterance(msg.text)
    utt.lang = 'fr-FR'
    utt.rate = 1
    utt.onstart = () => { setSpeaking(true); setSpeakingId(msg.id) }
    utt.onend = () => { setSpeaking(false); setSpeakingId(null) }
    utt.onerror = () => { setSpeaking(false); setSpeakingId(null) }
    window.speechSynthesis.speak(utt)
  }, [speaking, speakingId])

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="layout">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onCreate={createConv}
        onDelete={deleteConv}
        onRename={renameConv}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(p => !p)}
      />

      <div className="main">
        <header className="topbar">
          <div className="topbar-left">
            <span className="app-logo">💊</span>
            <div>
              <h1 className="app-title">Assistant Médical RAG</h1>
              <p className="app-sub">Pharmacologie · IA générative</p>
            </div>
          </div>
          <div className="topbar-right">
            <span className={`status-dot ${loading ? 'loading' : 'online'}`} />
            <span className="status-label">{loading ? 'Analyse…' : 'En ligne'}</span>
            <button className="icon-btn theme-btn" onClick={() => setDarkMode(p => !p)} title="Thème">
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </header>

        <div className="chat-area">
          {activeConv?.messages.map(msg => (
            <Message
              key={msg.id}
              msg={msg}
              onSpeak={handleSpeak}
              speaking={speaking}
              speakingId={speakingId}
            />
          ))}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        <div className="input-zone">
          <form className="input-bar" onSubmit={sendMessage}>
            <button
              type="button"
              className={`mic-btn ${listening ? 'listening' : ''}`}
              onClick={toggleListen}
              title={listening ? 'Arrêter l\'écoute' : 'Dicter'}
            >
              {listening ? '🔴' : '🎙️'}
            </button>
            <textarea
              ref={inputRef}
              className="chat-input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={listening ? '🎙️ Écoute en cours…' : 'Posez votre question… (Entrée pour envoyer)'}
              disabled={loading}
              rows={1}
            />
            <button
              type="submit"
              className="send-btn"
              disabled={loading || !input.trim()}
              title="Envoyer"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </form>
          <p className="hint">Shift+Entrée pour nouvelle ligne · 🎙️ pour dicter · 🔊 pour écouter</p>
        </div>
      </div>
    </div>
  )
}
