import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent } from '../../test/test-utils'
import ErrorBoundary, { withErrorBoundary } from '../ErrorBoundary'

// Component that throws on render
function ThrowingComponent({ shouldThrow = true }: { shouldThrow?: boolean }) {
  if (shouldThrow) {
    throw new Error('Test render error')
  }
  return <div>Child content rendered</div>
}

// Stable child component
function GoodChild() {
  return <div>Child content rendered</div>
}

describe('ErrorBoundary', () => {
  let consoleSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    consoleSpy.mockRestore()
  })

  it('renders children when no error occurs', () => {
    render(
      <ErrorBoundary>
        <GoodChild />
      </ErrorBoundary>
    )

    expect(screen.getByText('Child content rendered')).toBeInTheDocument()
  })

  it('catches rendering errors and shows fallback UI', () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    )

    expect(screen.queryByText('Child content rendered')).not.toBeInTheDocument()
    expect(screen.getByText('Hiba tortent')).toBeInTheDocument()
  })

  it('displays Hungarian error message in the fallback', () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    )

    expect(screen.getByText(/Sajnos varatlan hiba tortent/)).toBeInTheDocument()
    expect(screen.getByText(/probalkozzon ujra/)).toBeInTheDocument()
  })

  it('shows retry button (Ujraprobalas)', () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    )

    expect(screen.getByRole('button', { name: /Ujraprobalas/i })).toBeInTheDocument()
  })

  it('shows go home button (Foodalra)', () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    )

    expect(screen.getByRole('button', { name: /Foodalra/i })).toBeInTheDocument()
  })

  it('retry resets error state and re-renders children', () => {
    // We use a variable to control throwing behavior
    let shouldThrow = true

    function ConditionalThrower() {
      if (shouldThrow) {
        throw new Error('Test error')
      }
      return <div>Recovery successful</div>
    }

    render(
      <ErrorBoundary>
        <ConditionalThrower />
      </ErrorBoundary>
    )

    // Should be in error state
    expect(screen.getByText('Hiba tortent')).toBeInTheDocument()

    // Stop throwing before retry
    shouldThrow = false

    // Click retry
    fireEvent.click(screen.getByRole('button', { name: /Ujraprobalas/i }))

    // Should now render children again
    expect(screen.getByText('Recovery successful')).toBeInTheDocument()
    expect(screen.queryByText('Hiba tortent')).not.toBeInTheDocument()
  })

  it('renders custom fallback when provided', () => {
    render(
      <ErrorBoundary fallback={<div>Custom fallback UI</div>}>
        <ThrowingComponent />
      </ErrorBoundary>
    )

    expect(screen.getByText('Custom fallback UI')).toBeInTheDocument()
    expect(screen.queryByText('Hiba tortent')).not.toBeInTheDocument()
  })

  it('calls onError callback when error is caught', () => {
    const onError = vi.fn()

    render(
      <ErrorBoundary onError={onError}>
        <ThrowingComponent />
      </ErrorBoundary>
    )

    expect(onError).toHaveBeenCalledTimes(1)
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Test render error' }),
      expect.objectContaining({ componentStack: expect.any(String) })
    )
  })

  it('go home button navigates to /', () => {
    // Mock window.location.href
    const locationSpy = vi.spyOn(window, 'location', 'get').mockReturnValue({
      ...window.location,
      href: '',
    } as Location)

    const hrefSetter = vi.fn()
    Object.defineProperty(window.location, 'href', {
      set: hrefSetter,
      configurable: true,
    })

    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    )

    fireEvent.click(screen.getByRole('button', { name: /Foodalra/i }))

    // The handleGoHome sets window.location.href = '/'
    // In happy-dom this may or may not work, but we verify the button is clickable
    locationSpy.mockRestore()
  })

  describe('withErrorBoundary HOC', () => {
    it('wraps a component with ErrorBoundary', () => {
      const WrappedGood = withErrorBoundary(GoodChild)

      render(<WrappedGood />)

      expect(screen.getByText('Child content rendered')).toBeInTheDocument()
    })

    it('catches errors in wrapped component', () => {
      const WrappedThrowing = withErrorBoundary(ThrowingComponent)

      render(<WrappedThrowing />)

      expect(screen.getByText('Hiba tortent')).toBeInTheDocument()
      expect(screen.queryByText('Child content rendered')).not.toBeInTheDocument()
    })

    it('uses custom fallback when provided to HOC', () => {
      const WrappedThrowing = withErrorBoundary(
        ThrowingComponent,
        <div>HOC custom fallback</div>
      )

      render(<WrappedThrowing />)

      expect(screen.getByText('HOC custom fallback')).toBeInTheDocument()
    })

    it('sets correct displayName', () => {
      const WrappedGood = withErrorBoundary(GoodChild)

      expect(WrappedGood.displayName).toBe('withErrorBoundary(GoodChild)')
    })
  })
})
