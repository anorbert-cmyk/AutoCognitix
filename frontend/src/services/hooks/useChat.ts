/**
 * useChat Hook
 * Custom hook for managing chat state, streaming, and speech recognition.
 * Does NOT use React Query - manages message history in local state.
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import {
  streamChatMessage,
  type ChatMessage,
  type ChatSource,
  type VehicleContext,
} from '../chatService'

interface UseChatOptions {
  vehicleContext?: VehicleContext
  diagnosisId?: string
}

interface UseChatReturn {
  messages: ChatMessage[]
  isStreaming: boolean
  currentResponse: string
  sendMessage: (text: string) => void
  clearMessages: () => void
  conversationId: string | undefined
  suggestions: string[]
}

let messageIdCounter = 0

function generateMessageId(): string {
  messageIdCounter += 1
  return `msg_${Date.now()}_${messageIdCounter}`
}

export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const { vehicleContext, diagnosisId } = options

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentResponse, setCurrentResponse] = useState('')
  const [conversationId, setConversationId] = useState<string | undefined>(undefined)
  const [suggestions, setSuggestions] = useState<string[]>([])

  const abortControllerRef = useRef<AbortController | null>(null)
  const currentSourcesRef = useRef<ChatSource[]>([])
  const currentResponseRef = useRef('')

  // Cleanup on unmount — abort any ongoing stream
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
        abortControllerRef.current = null
      }
    }
  }, [])

  const sendMessage = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || isStreaming) return

      // Clear previous suggestions
      setSuggestions([])

      // Add user message
      const userMessage: ChatMessage = {
        id: generateMessageId(),
        role: 'user',
        content: trimmed,
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, userMessage])

      // Reset streaming state
      setCurrentResponse('')
      currentResponseRef.current = ''
      setIsStreaming(true)
      currentSourcesRef.current = []

      const newSuggestions: string[] = []

      // Start streaming
      const controller = streamChatMessage(
        trimmed,
        conversationId,
        vehicleContext,
        diagnosisId,
        {
          onStart: () => {
            // Streaming started, state already set
          },
          onToken: (token: string) => {
            currentResponseRef.current += token
            setCurrentResponse(currentResponseRef.current)
          },
          onSource: (source: ChatSource) => {
            currentSourcesRef.current = [...currentSourcesRef.current, source]
          },
          onSuggestion: (suggestion: string) => {
            newSuggestions.push(suggestion)
          },
          onComplete: () => {
            const assistantMessage: ChatMessage = {
              id: generateMessageId(),
              role: 'assistant',
              content: currentResponseRef.current,
              timestamp: new Date().toISOString(),
              sources:
                currentSourcesRef.current.length > 0
                  ? [...currentSourcesRef.current]
                  : undefined,
            }
            setMessages((prev) => [...prev, assistantMessage])
            setCurrentResponse('')
            setIsStreaming(false)
            currentSourcesRef.current = []

            if (newSuggestions.length > 0) {
              setSuggestions(newSuggestions)
            }

            // Extract conversation_id from first response if not set
            if (!conversationId) {
              setConversationId(`conv_${Date.now()}`)
            }
          },
          onError: (error) => {
            const errorMessage: ChatMessage = {
              id: generateMessageId(),
              role: 'assistant',
              content: `Hiba tortent: ${error.message}`,
              timestamp: new Date().toISOString(),
            }
            setMessages((prev) => [...prev, errorMessage])
            setCurrentResponse('')
            setIsStreaming(false)
            currentSourcesRef.current = []
          },
        }
      )

      abortControllerRef.current = controller
    },
    [isStreaming, conversationId, vehicleContext, diagnosisId]
  )

  const clearMessages = useCallback(() => {
    // Abort any ongoing stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setMessages([])
    setCurrentResponse('')
    setIsStreaming(false)
    setConversationId(undefined)
    setSuggestions([])
    currentSourcesRef.current = []
  }, [])

  return {
    messages,
    isStreaming,
    currentResponse,
    sendMessage,
    clearMessages,
    conversationId,
    suggestions,
  }
}
