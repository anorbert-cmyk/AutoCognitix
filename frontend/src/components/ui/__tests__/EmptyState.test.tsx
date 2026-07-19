import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '../../../test/test-utils'
import { EmptyState } from '../EmptyState'

describe('EmptyState', () => {
  it('renders the title', () => {
    render(<EmptyState title="Még nincs adat" />)

    expect(screen.getByText('Még nincs adat')).toBeInTheDocument()
  })

  it('renders the description when provided', () => {
    render(<EmptyState title="Cím" description="Részletes leírás a teendőről" />)

    expect(screen.getByText('Részletes leírás a teendőről')).toBeInTheDocument()
  })

  it('omits the description when not provided', () => {
    render(<EmptyState title="Cím" />)

    expect(screen.queryByText('Részletes leírás a teendőről')).not.toBeInTheDocument()
  })

  it('renders a link action pointing at the given route', () => {
    render(<EmptyState title="Cím" action={{ label: 'Tovább', to: '/diagnosis' }} />)

    const link = screen.getByRole('link', { name: 'Tovább' })
    expect(link).toHaveAttribute('href', '/diagnosis')
  })

  it('renders a button action that fires onClick', () => {
    const onClick = vi.fn()
    render(<EmptyState title="Cím" action={{ label: 'Megnyitás', onClick }} />)

    fireEvent.click(screen.getByRole('button', { name: 'Megnyitás' }))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('wraps the icon in an aria-hidden container', () => {
    render(<EmptyState icon={<span data-testid="ico" />} title="Cím" />)

    expect(screen.getByTestId('ico').parentElement).toHaveAttribute('aria-hidden', 'true')
  })

  it('is not announced as an alert (empty is not an error)', () => {
    render(<EmptyState title="Cím" description="leírás" />)

    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
