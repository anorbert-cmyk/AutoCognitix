/**
 * ChatInput Component
 * Text input with send button, mic button, and character counter.
 * Enter to send, Shift+Enter for newline.
 */

import { useState, useCallback, useRef, useEffect, type KeyboardEvent, type ChangeEvent } from 'react'
import { Send, Mic } from 'lucide-react'
import { cn } from '@/lib/utils'

const MAX_CHARS = 1000

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
  placeholder?: string
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Irjon uzenetet...',
}: ChatInputProps) {
  const [text, setText] = useState('')
  const [isListening, setIsListening] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null)

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort()
      }
    }
  }, [])

  const charCount = text.length
  const canSend = text.trim().length > 0 && !disabled && charCount <= MAX_CHARS

  const handleSend = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed || disabled || charCount > MAX_CHARS) return
    onSend(trimmed)
    setText('')
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [text, disabled, charCount, onSend])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend]
  )

  const handleChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value
    if (value.length <= MAX_CHARS) {
      setText(value)
    }
    // Auto-resize textarea
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`
  }, [])

  const handleMic = useCallback(() => {
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const SpeechRecognitionConstructor =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    /* eslint-enable @typescript-eslint/no-explicit-any */
    if (!SpeechRecognitionConstructor) {
      return
    }
    const recognition = new SpeechRecognitionConstructor()
    recognition.lang = 'hu-HU'
    recognition.continuous = false
    recognition.interimResults = false
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (event: any) => {
      const transcript: string = event.results[0][0].transcript
      setText((prev) => {
        const combined = prev + (prev ? ' ' : '') + transcript
        return combined.slice(0, MAX_CHARS)
      })
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error)
      setIsListening(false)
    }
    recognition.onend = () => {
      setIsListening(false)
    }
    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
  }, [])

  const hasSpeechRecognition =
    typeof window !== 'undefined' &&
    ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)

  return (
    <div className="border-t border-gray-200 bg-white px-4 py-3">
      <div className="flex items-end gap-2 max-w-4xl mx-auto">
        {/* Mic button */}
        {hasSpeechRecognition && (
          <button
            type="button"
            onClick={handleMic}
            disabled={disabled}
            className={cn(
              'flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full transition-colors',
              disabled
                ? 'text-gray-300 cursor-not-allowed'
                : isListening
                  ? 'text-red-500 animate-pulse'
                  : 'text-gray-500 hover:text-primary-600 hover:bg-primary-50'
            )}
            title="Diktalas"
            aria-label="Beszedfelismeres inditasa"
          >
            <Mic className="h-5 w-5" aria-hidden="true" />
          </button>
        )}

        {/* Text input */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={placeholder}
            rows={1}
            className={cn(
              'w-full resize-none rounded-2xl border border-gray-300 bg-gray-50 px-4 py-2.5 pr-12 text-sm text-gray-900 placeholder:text-gray-400',
              'focus:border-primary-500 focus:bg-white focus:ring-1 focus:ring-primary-500 focus:outline-none',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'transition-colors'
            )}
            style={{ maxHeight: '120px' }}
          />
          {/* Char counter */}
          {charCount > MAX_CHARS * 0.8 && (
            <span
              className={cn(
                'absolute bottom-1 right-12 text-[10px] tabular-nums',
                charCount > MAX_CHARS ? 'text-red-500 font-bold' : 'text-gray-400'
              )}
            >
              {charCount}/{MAX_CHARS}
            </span>
          )}
        </div>

        {/* Send button */}
        <button
          type="button"
          onClick={handleSend}
          disabled={!canSend}
          className={cn(
            'flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full transition-colors',
            canSend
              ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-sm'
              : 'bg-gray-100 text-gray-300 cursor-not-allowed'
          )}
          title="Kuldes"
          aria-label="Uzenet kuldese"
        >
          <Send className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </div>
  )
}

export default ChatInput
