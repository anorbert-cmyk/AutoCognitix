import { useRef, useEffect, ReactNode } from 'react'

interface ChatWindowProps {
  children: ReactNode
}

export default function ChatWindow({ children }: ChatWindowProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sentinelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    sentinelRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [children])

  return (
    <div
      ref={containerRef}
      className="flex h-full flex-col overflow-y-auto p-4 space-y-4"
    >
      {children}
      <div ref={sentinelRef} />
    </div>
  )
}
