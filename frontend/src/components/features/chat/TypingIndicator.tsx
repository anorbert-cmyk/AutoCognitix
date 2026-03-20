/**
 * TypingIndicator Component
 * Animated three-dot typing indicator styled as an AI message bubble.
 */

import { Bot } from 'lucide-react'

export function TypingIndicator() {
  return (
    <div className="flex gap-3 mr-auto max-w-[85%]">
      {/* Avatar */}
      <div className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-gray-200 text-gray-600">
        <Bot className="h-4 w-4" aria-hidden="true" />
      </div>

      {/* Dots bubble */}
      <div className="bg-gray-100 rounded-2xl rounded-bl-md px-4 py-3 flex items-center gap-1.5">
        <span
          className="block w-2 h-2 rounded-full bg-gray-400 animate-bounce"
          style={{ animationDelay: '0ms', animationDuration: '600ms' }}
        />
        <span
          className="block w-2 h-2 rounded-full bg-gray-400 animate-bounce"
          style={{ animationDelay: '150ms', animationDuration: '600ms' }}
        />
        <span
          className="block w-2 h-2 rounded-full bg-gray-400 animate-bounce"
          style={{ animationDelay: '300ms', animationDuration: '600ms' }}
        />
      </div>
    </div>
  )
}

export default TypingIndicator
