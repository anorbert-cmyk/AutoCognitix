/**
 * ChatPage - AI Chat Asszisztens
 * Full-height chat page with message history, streaming responses, and suggestions.
 * Hungarian UI throughout.
 */

import { useEffect, useRef, useMemo, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Bot, Car, MessageSquare, Trash2 } from 'lucide-react'
import { useChat } from '../services/hooks/useChat'
import { MessageBubble } from '../components/features/chat/MessageBubble'
import { ChatInput } from '../components/features/chat/ChatInput'
import { SuggestionChips } from '../components/features/chat/SuggestionChips'
import { TypingIndicator } from '../components/features/chat/TypingIndicator'
import type { VehicleContext } from '../services/chatService'
import { cn } from '@/lib/utils'

// =============================================================================
// Initial Suggestions
// =============================================================================

const INITIAL_SUGGESTIONS = [
  'Mi az a P0300 hibakod?',
  'Motor berreges okai',
  'Muszaki vizsgara keszulok',
]

// =============================================================================
// Vehicle Context Badge
// =============================================================================

function VehicleBadge({ context }: { context: VehicleContext }) {
  const parts: string[] = []
  if (context.make) parts.push(context.make)
  if (context.model) parts.push(context.model)
  if (context.year) parts.push(String(context.year))

  if (parts.length === 0) return null

  return (
    <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary-50 border border-primary-200 text-xs font-medium text-primary-700">
      <Car className="h-3 w-3" aria-hidden="true" />
      {parts.join(' ')}
      {context.dtc_codes && context.dtc_codes.length > 0 && (
        <span className="font-mono font-bold ml-1">
          {context.dtc_codes.join(', ')}
        </span>
      )}
    </div>
  )
}

// =============================================================================
// Welcome Screen
// =============================================================================

function WelcomeScreen() {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center max-w-md">
        <div className="mx-auto w-16 h-16 rounded-2xl bg-primary-100 flex items-center justify-center mb-6">
          <Bot className="h-8 w-8 text-primary-600" aria-hidden="true" />
        </div>
        <h2 className="text-2xl font-bold text-gray-900 mb-3">
          Udvozollek!
        </h2>
        <p className="text-gray-600 leading-relaxed">
          Kerdezz barmit a jarmuved problemjarol. Segithetek hibakodok
          ertelmezeseben, tunetek diagnosztizalasaban es javitasi
          tanacsadasban.
        </p>
      </div>
    </div>
  )
}

// =============================================================================
// Streaming Bubble (partial response being typed)
// =============================================================================

function StreamingBubble({ content }: { content: string }) {
  if (!content) return null

  return (
    <div className="flex gap-3 mr-auto max-w-[85%]">
      <div className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-gray-200 text-gray-600">
        <Bot className="h-4 w-4" aria-hidden="true" />
      </div>
      <div className="bg-gray-100 text-gray-900 rounded-2xl rounded-bl-md px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap break-words">
        {content}
        <span className="inline-block w-1.5 h-4 bg-primary-500 rounded-sm ml-0.5 animate-pulse align-text-bottom" />
      </div>
    </div>
  )
}

// =============================================================================
// ChatPage
// =============================================================================

export default function ChatPage() {
  const [searchParams] = useSearchParams()

  // Extract vehicle context from URL params
  const vehicleContext = useMemo<VehicleContext | undefined>(() => {
    const make = searchParams.get('make')
    const model = searchParams.get('model')
    const year = searchParams.get('year')
    const dtc = searchParams.get('dtc')

    if (!make && !model && !year && !dtc) return undefined

    return {
      make: make || undefined,
      model: model || undefined,
      year: year ? parseInt(year, 10) : undefined,
      dtc_codes: dtc ? dtc.split(',') : undefined,
    }
  }, [searchParams])

  const diagnosisId = searchParams.get('diagnosis_id') || undefined

  const {
    messages,
    isStreaming,
    currentResponse,
    sendMessage,
    clearMessages,
    suggestions,
  } = useChat({ vehicleContext, diagnosisId })

  // Auto-scroll refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages or streaming content
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentResponse, isStreaming])

  const handleSuggestionSelect = useCallback(
    (suggestion: string) => {
      sendMessage(suggestion)
    },
    [sendMessage]
  )

  const hasMessages = messages.length > 0
  const showInitialSuggestions = !hasMessages && !isStreaming
  const showResponseSuggestions = !isStreaming && suggestions.length > 0

  return (
    <div
      className="flex flex-col bg-white"
      style={{ height: 'calc(100vh - var(--header-height, 64px))' }}
    >
      {/* ═══ Top Bar ═══ */}
      <div className="flex-shrink-0 flex items-center justify-between gap-4 px-4 py-3 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center gap-3">
          <MessageSquare className="h-5 w-5 text-primary-600" aria-hidden="true" />
          <h1 className="text-lg font-bold text-gray-900">AI Chat Asszisztens</h1>
          {vehicleContext && <VehicleBadge context={vehicleContext} />}
        </div>
        {hasMessages && (
          <button
            type="button"
            onClick={clearMessages}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-red-600 transition-colors font-medium"
            title="Beszelgetes torlese"
          >
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
            Torles
          </button>
        )}
      </div>

      {/* ═══ Messages Area ═══ */}
      <div
        ref={messagesContainerRef}
        className={cn(
          'flex-1 overflow-y-auto',
          !hasMessages && !isStreaming ? 'flex flex-col' : 'p-4 space-y-4'
        )}
      >
        {!hasMessages && !isStreaming ? (
          <WelcomeScreen />
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {/* Streaming partial response */}
            {isStreaming && currentResponse && (
              <StreamingBubble content={currentResponse} />
            )}

            {/* Typing indicator (before first token arrives) */}
            {isStreaming && !currentResponse && <TypingIndicator />}

            {/* Scroll anchor */}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* ═══ Suggestions ═══ */}
      {showInitialSuggestions && (
        <SuggestionChips
          suggestions={INITIAL_SUGGESTIONS}
          onSelect={handleSuggestionSelect}
        />
      )}
      {showResponseSuggestions && (
        <SuggestionChips
          suggestions={suggestions}
          onSelect={handleSuggestionSelect}
        />
      )}

      {/* ═══ Input Area ═══ */}
      <ChatInput
        onSend={sendMessage}
        disabled={isStreaming}
        placeholder="Kerjen segitseget a jarmuvehez..."
      />
    </div>
  )
}
