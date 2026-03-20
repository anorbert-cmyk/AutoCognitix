import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '../../../test/test-utils'
import ErrorState, { InlineError, FieldError } from '../ErrorState'
import { ApiError } from '../../../services/api'

describe('ErrorState', () => {
  describe('error types render correctly', () => {
    it('renders network error type', () => {
      render(<ErrorState type="network" />)

      expect(screen.getByText('Kapcsolati hiba')).toBeInTheDocument()
      expect(
        screen.getByText(/Nincs internetkapcsolat/)
      ).toBeInTheDocument()
      expect(screen.getByRole('alert')).toBeInTheDocument()
    })

    it('renders server error type', () => {
      render(<ErrorState type="server" />)

      expect(screen.getByText('Szerver hiba')).toBeInTheDocument()
      expect(
        screen.getByText(/szerver atmeneti hibaba/)
      ).toBeInTheDocument()
    })

    it('renders not_found error type', () => {
      render(<ErrorState type="not_found" />)

      expect(screen.getByText('Nem talalhato')).toBeInTheDocument()
      expect(
        screen.getByText(/eroforras nem talalhato/)
      ).toBeInTheDocument()
    })

    it('renders unauthorized error type', () => {
      render(<ErrorState type="unauthorized" />)

      expect(screen.getByText('Bejelentkezes szukseges')).toBeInTheDocument()
      expect(
        screen.getByText(/jelentkezzen be/)
      ).toBeInTheDocument()
    })

    it('renders forbidden error type', () => {
      render(<ErrorState type="forbidden" />)

      expect(screen.getByText('Hozzaferes megtagadva')).toBeInTheDocument()
    })

    it('renders validation error type', () => {
      render(<ErrorState type="validation" />)

      expect(screen.getByText('Ervenytelen adatok')).toBeInTheDocument()
    })

    it('renders timeout error type', () => {
      render(<ErrorState type="timeout" />)

      expect(screen.getByText('Idotullepes')).toBeInTheDocument()
    })

    it('renders rate_limit error type', () => {
      render(<ErrorState type="rate_limit" />)

      expect(screen.getByText('Tul sok keres')).toBeInTheDocument()
    })

    it('renders generic error type', () => {
      render(<ErrorState type="generic" />)

      expect(screen.getByText('Hiba tortent')).toBeInTheDocument()
    })
  })

  describe('auto-detection from ApiError', () => {
    it('detects network error from ApiError', () => {
      const error = new ApiError('Network error', 0, undefined, undefined, undefined, true)

      render(<ErrorState error={error} />)

      expect(screen.getByText('Kapcsolati hiba')).toBeInTheDocument()
    })

    it('detects 404 as not_found', () => {
      const error = new ApiError('Not found', 404)

      render(<ErrorState error={error} />)

      expect(screen.getByText('Nem talalhato')).toBeInTheDocument()
    })

    it('detects 500 as server error', () => {
      const error = new ApiError('Server error', 500)

      render(<ErrorState error={error} />)

      expect(screen.getByText('Szerver hiba')).toBeInTheDocument()
    })

    it('detects 401 as unauthorized', () => {
      const error = new ApiError('Unauthorized', 401)

      render(<ErrorState error={error} />)

      expect(screen.getByText('Bejelentkezes szukseges')).toBeInTheDocument()
    })

    it('shows ApiError detail as message', () => {
      const error = new ApiError('err', 404, 'Jarmu nem talalhato a rendszerben')

      render(<ErrorState error={error} />)

      expect(screen.getByText('Jarmu nem talalhato a rendszerben')).toBeInTheDocument()
    })
  })

  describe('custom title and message', () => {
    it('overrides title when provided', () => {
      render(<ErrorState type="server" title="Egyedi cim" />)

      expect(screen.getByText('Egyedi cim')).toBeInTheDocument()
      expect(screen.queryByText('Szerver hiba')).not.toBeInTheDocument()
    })

    it('overrides message when provided', () => {
      render(<ErrorState type="server" message="Egyedi uzenet szoveg" />)

      expect(screen.getByText('Egyedi uzenet szoveg')).toBeInTheDocument()
    })
  })

  describe('action buttons', () => {
    it('shows retry button when onRetry provided', () => {
      const onRetry = vi.fn()

      render(<ErrorState type="server" onRetry={onRetry} />)

      const retryButton = screen.getByRole('button', { name: /Ujraprobalas/i })
      expect(retryButton).toBeInTheDocument()

      fireEvent.click(retryButton)
      expect(onRetry).toHaveBeenCalledTimes(1)
    })

    it('does not show retry button when onRetry not provided', () => {
      render(<ErrorState type="server" />)

      expect(screen.queryByRole('button', { name: /Ujraprobalas/i })).not.toBeInTheDocument()
    })

    it('shows back button when onBack provided', () => {
      const onBack = vi.fn()

      render(<ErrorState type="not_found" onBack={onBack} />)

      const backButton = screen.getByRole('button', { name: /Vissza/i })
      expect(backButton).toBeInTheDocument()

      fireEvent.click(backButton)
      expect(onBack).toHaveBeenCalledTimes(1)
    })

    it('shows home button when onHome provided', () => {
      const onHome = vi.fn()

      render(<ErrorState type="server" onHome={onHome} />)

      const homeButton = screen.getByRole('button', { name: /Foodalra/i })
      expect(homeButton).toBeInTheDocument()

      fireEvent.click(homeButton)
      expect(onHome).toHaveBeenCalledTimes(1)
    })

    it('renders custom actions', () => {
      render(
        <ErrorState
          type="generic"
          actions={<button>Custom action</button>}
        />
      )

      expect(screen.getByRole('button', { name: 'Custom action' })).toBeInTheDocument()
    })
  })

  describe('compact mode', () => {
    it('renders compact UI with role=alert', () => {
      render(<ErrorState type="network" compact />)

      const alert = screen.getByRole('alert')
      expect(alert).toBeInTheDocument()
      expect(screen.getByText('Kapcsolati hiba')).toBeInTheDocument()
    })

    it('shows retry button in compact mode when onRetry provided', () => {
      const onRetry = vi.fn()

      render(<ErrorState type="network" compact onRetry={onRetry} />)

      const retryButton = screen.getByTitle('Ujraprobalas')
      expect(retryButton).toBeInTheDocument()

      fireEvent.click(retryButton)
      expect(onRetry).toHaveBeenCalledTimes(1)
    })

    it('does not show retry in compact mode when onRetry not provided', () => {
      render(<ErrorState type="network" compact />)

      expect(screen.queryByTitle('Ujraprobalas')).not.toBeInTheDocument()
    })
  })

  describe('Hungarian error messages', () => {
    it('all error types have Hungarian titles', () => {
      const errorTypes = [
        { type: 'network' as const, title: 'Kapcsolati hiba' },
        { type: 'server' as const, title: 'Szerver hiba' },
        { type: 'not_found' as const, title: 'Nem talalhato' },
        { type: 'unauthorized' as const, title: 'Bejelentkezes szukseges' },
        { type: 'forbidden' as const, title: 'Hozzaferes megtagadva' },
        { type: 'validation' as const, title: 'Ervenytelen adatok' },
        { type: 'timeout' as const, title: 'Idotullepes' },
        { type: 'rate_limit' as const, title: 'Tul sok keres' },
        { type: 'generic' as const, title: 'Hiba tortent' },
      ]

      for (const { type, title } of errorTypes) {
        const { unmount } = render(<ErrorState type={type} />)
        expect(screen.getByText(title)).toBeInTheDocument()
        unmount()
      }
    })
  })

  describe('defaults', () => {
    it('defaults to generic type when no error or type provided', () => {
      render(<ErrorState />)

      expect(screen.getByText('Hiba tortent')).toBeInTheDocument()
    })

    it('defaults to generic for plain Error objects', () => {
      render(<ErrorState error={new Error('Something went wrong')} />)

      expect(screen.getByText('Hiba tortent')).toBeInTheDocument()
    })

    it('detects network from Error message containing "network"', () => {
      render(<ErrorState error={new Error('network failure')} />)

      expect(screen.getByText('Kapcsolati hiba')).toBeInTheDocument()
    })

    it('detects timeout from Error message containing "timeout"', () => {
      render(<ErrorState error={new Error('request timeout')} />)

      expect(screen.getByText('Idotullepes')).toBeInTheDocument()
    })
  })
})

describe('InlineError', () => {
  it('renders the error message inline', () => {
    render(<InlineError message="Kotelezo mezo" />)

    expect(screen.getByText('Kotelezo mezo')).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(
      <InlineError message="Hiba" className="mt-2" />
    )

    expect(container.firstChild).toHaveClass('mt-2')
  })
})

describe('FieldError', () => {
  it('renders error message for form fields', () => {
    render(<FieldError error="A mezo kitoltese kotelezo" />)

    expect(screen.getByText('A mezo kitoltese kotelezo')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('renders nothing when error is null', () => {
    const { container } = render(<FieldError error={null} />)

    expect(container.firstChild).toBeNull()
  })

  it('renders nothing when error is undefined', () => {
    const { container } = render(<FieldError />)

    expect(container.firstChild).toBeNull()
  })

  it('applies custom className', () => {
    render(<FieldError error="Hiba" className="custom-class" />)

    expect(screen.getByRole('alert')).toHaveClass('custom-class')
  })
})
