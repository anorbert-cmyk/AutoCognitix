import { describe, it, expect } from 'vitest'
import { render } from '../../../test/test-utils'
import { Skeleton } from '../Skeleton'

describe('Skeleton', () => {
  it('is decorative and hidden from assistive tech', () => {
    const { container } = render(<Skeleton />)

    expect(container.firstChild).toHaveAttribute('aria-hidden', 'true')
  })

  it('carries the base pulse classes', () => {
    const { container } = render(<Skeleton />)

    expect(container.firstChild).toHaveClass('animate-pulse', 'bg-muted')
  })

  it('merges the provided className with the base classes', () => {
    const { container } = render(<Skeleton className="h-10 w-full" />)

    expect(container.firstChild).toHaveClass('animate-pulse', 'bg-muted', 'h-10', 'w-full')
  })
})
