/**
 * MessageBubble Component
 * Renders a single chat message with user/assistant styling.
 * User messages: right-aligned blue. AI messages: left-aligned gray with optional sources.
 */

import { useMemo, useState, useEffect } from 'react'
import { Bot, User } from 'lucide-react'
import type { ChatMessage, ChatSource } from '../../../services/chatService'
import { cn } from '@/lib/utils'

// =============================================================================
// Relative Time Formatter
// =============================================================================

function formatRelativeTime(isoDate: string): string {
  const now = Date.now()
  const then = new Date(isoDate).getTime()
  const diffMs = now - then

  if (diffMs < 0) return 'most'

  const seconds = Math.floor(diffMs / 1000)
  if (seconds < 60) return 'most'

  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes} perce`

  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours} oraja`

  const days = Math.floor(hours / 24)
  return `${days} napja`
}

// =============================================================================
// Source Card
// =============================================================================

const sourceTypeLabels: Record<string, string> = {
  dtc: 'DTC',
  recall: 'Visszahivas',
  complaint: 'Panasz',
  tsb: 'TSB',
  manual: 'Kezikonyv',
  database: 'Adatbazis',
}

function SourceCard({ source }: { source: ChatSource }) {
  const label = sourceTypeLabels[source.type] || source.type

  return (
    <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-white border border-gray-200 rounded-lg text-xs shadow-sm">
      <span className="font-bold text-primary-600 uppercase tracking-wider">
        {label}
      </span>
      <span className="text-gray-700 truncate max-w-[200px]">{source.title}</span>
      {source.url && (
        <a
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary-500 hover:text-primary-700 underline flex-shrink-0"
        >
          Link
        </a>
      )}
    </div>
  )
}

// =============================================================================
// Text Renderer (whitespace-safe, no HTML injection)
// =============================================================================

function renderText(text: string): React.ReactNode {
  // Split on newlines and preserve whitespace structure
  const lines = text.split('\n')
  return lines.map((line, i) => (
    <span key={i}>
      {line}
      {i < lines.length - 1 && <br />}
    </span>
  ))
}

// =============================================================================
// MessageBubble
// =============================================================================

interface MessageBubbleProps {
  message: ChatMessage
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  // Re-render relative time every 60 seconds
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), 60_000)
    return () => clearInterval(interval)
  }, [])

  const relativeTime = useMemo(
    () => formatRelativeTime(message.timestamp),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [message.timestamp, tick]
  )

  return (
    <div
      className={cn(
        'flex gap-3 max-w-[85%]',
        isUser ? 'ml-auto flex-row-reverse' : 'mr-auto'
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full',
          isUser ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-600'
        )}
      >
        {isUser ? (
          <User className="h-4 w-4" aria-hidden="true" />
        ) : (
          <Bot className="h-4 w-4" aria-hidden="true" />
        )}
      </div>

      {/* Bubble + meta */}
      <div className={cn('flex flex-col', isUser ? 'items-end' : 'items-start')}>
        <div
          className={cn(
            'px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap break-words',
            isUser
              ? 'bg-primary-600 text-white rounded-2xl rounded-br-md'
              : 'bg-gray-100 text-gray-900 rounded-2xl rounded-bl-md'
          )}
        >
          {renderText(message.content)}
        </div>

        {/* Sources (AI only) */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {message.sources.map((source, idx) => (
              <SourceCard key={`${source.type}-${idx}`} source={source} />
            ))}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-[11px] text-gray-400 mt-1 px-1">{relativeTime}</span>
      </div>
    </div>
  )
}

export default MessageBubble
