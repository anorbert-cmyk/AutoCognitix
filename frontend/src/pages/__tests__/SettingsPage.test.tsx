import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '../../test/test-utils'
import SettingsPage from '../SettingsPage'

// -----------------------------------------------------------------------------
// Per-file mocks (documented trap): test-utils exposes no AuthProvider, and other
// suites mock the whole AuthContext module (without a Provider export), so each
// page test mocks useAuth itself rather than relying on a shared provider.
// -----------------------------------------------------------------------------

const authState = vi.hoisted(() => ({
  value: {
    user: {
      id: 'u1',
      email: 'a@b.hu',
      full_name: 'Teszt Elek',
      role: 'user',
      created_at: '2026-01-05T10:00:00Z',
      is_active: true,
    },
    updateProfile: vi.fn(),
    changePassword: vi.fn(),
    deleteAccount: vi.fn(),
  },
}))

const toastMock = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
const navigateMock = vi.hoisted(() => vi.fn())
const exportDataMock = vi.hoisted(() => vi.fn())

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => authState.value,
}))

vi.mock('../../contexts/ToastContext', () => ({
  useToast: () => toastMock,
}))

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useNavigate: () => navigateMock }
})

vi.mock('../../services/authService', () => ({
  exportData: exportDataMock,
}))

beforeEach(() => {
  authState.value = {
    user: {
      id: 'u1',
      email: 'a@b.hu',
      full_name: 'Teszt Elek',
      role: 'user',
      created_at: '2026-01-05T10:00:00Z',
      is_active: true,
    },
    updateProfile: vi.fn(),
    changePassword: vi.fn(),
    deleteAccount: vi.fn(),
  }
  toastMock.success.mockReset()
  toastMock.error.mockReset()
  navigateMock.mockReset()
  exportDataMock.mockReset()
})

// jsdom does not implement URL.createObjectURL/revokeObjectURL, so the export
// test defines them ad hoc. Snapshot the pristine descriptors at load time and
// restore (or delete) them after each test so the stubs never leak into other
// suites — beforeEach's mock resets don't cover these globals.
const originalCreateObjectURL = Object.getOwnPropertyDescriptor(URL, 'createObjectURL')
const originalRevokeObjectURL = Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL')

afterEach(() => {
  if (originalCreateObjectURL) {
    Object.defineProperty(URL, 'createObjectURL', originalCreateObjectURL)
  } else {
    delete (URL as unknown as { createObjectURL?: unknown }).createObjectURL
  }
  if (originalRevokeObjectURL) {
    Object.defineProperty(URL, 'revokeObjectURL', originalRevokeObjectURL)
  } else {
    delete (URL as unknown as { revokeObjectURL?: unknown }).revokeObjectURL
  }
})

describe('SettingsPage', () => {
  it('rendereli a három fület, alapból a Profil aktív', () => {
    render(<SettingsPage />)
    expect(screen.getByRole('button', { name: 'Profil' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Jelszó' })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('button', { name: 'Adatok és adatvédelem' })).toHaveAttribute(
      'aria-pressed',
      'false'
    )
  })

  it('a profil mentése az updateProfile-t a mezőértékekkel hívja', async () => {
    render(<SettingsPage />)
    fireEvent.change(screen.getByLabelText('Teljes név'), { target: { value: 'Új Név' } })
    fireEvent.change(screen.getByLabelText('Email cím'), { target: { value: 'uj@b.hu' } })
    const form = screen.getByRole('button', { name: 'Mentés' }).closest('form')!
    fireEvent.submit(form)
    await waitFor(() =>
      expect(authState.value.updateProfile).toHaveBeenCalledWith({
        full_name: 'Új Név',
        email: 'uj@b.hu',
      })
    )
  })

  it('eltérő jelszavaknál nem hívja a changePassword-ot és hibát jelez', async () => {
    render(<SettingsPage />)
    fireEvent.click(screen.getByRole('button', { name: 'Jelszó' }))
    fireEvent.change(screen.getByLabelText('Jelenlegi jelszó'), { target: { value: 'Regi123!' } })
    fireEvent.change(screen.getByLabelText('Új jelszó'), { target: { value: 'UjJelszo123!' } })
    fireEvent.change(screen.getByLabelText('Új jelszó megerősítése'), {
      target: { value: 'MasJelszo123!' },
    })
    const form = screen.getByRole('button', { name: 'Jelszó módosítása' }).closest('form')!
    fireEvent.submit(form)
    expect(await screen.findByText('A jelszavak nem egyeznek')).toBeInTheDocument()
    expect(authState.value.changePassword).not.toHaveBeenCalled()
  })

  it('gyenge jelszónál a szabály-hibát jelzi és nem hívja a changePassword-ot', async () => {
    render(<SettingsPage />)
    fireEvent.click(screen.getByRole('button', { name: 'Jelszó' }))
    fireEvent.change(screen.getByLabelText('Jelenlegi jelszó'), { target: { value: 'Regi123!' } })
    fireEvent.change(screen.getByLabelText('Új jelszó'), { target: { value: 'abc' } })
    fireEvent.change(screen.getByLabelText('Új jelszó megerősítése'), { target: { value: 'abc' } })
    const form = screen.getByRole('button', { name: 'Jelszó módosítása' }).closest('form')!
    fireEvent.submit(form)
    expect(
      await screen.findByText('A jelszó nem felel meg a követelményeknek')
    ).toBeInTheDocument()
    expect(authState.value.changePassword).not.toHaveBeenCalled()
  })

  it('sikeres jelszóváltoztatásnál meghívja a changePassword-ot és üríti a mezőket', async () => {
    authState.value.changePassword.mockResolvedValue(undefined)
    render(<SettingsPage />)
    fireEvent.click(screen.getByRole('button', { name: 'Jelszó' }))
    const current = screen.getByLabelText('Jelenlegi jelszó')
    const next = screen.getByLabelText('Új jelszó')
    const confirm = screen.getByLabelText('Új jelszó megerősítése')
    fireEvent.change(current, { target: { value: 'Regi123!' } })
    fireEvent.change(next, { target: { value: 'UjJelszo123!' } })
    fireEvent.change(confirm, { target: { value: 'UjJelszo123!' } })
    const form = screen.getByRole('button', { name: 'Jelszó módosítása' }).closest('form')!
    fireEvent.submit(form)
    await waitFor(() =>
      expect(authState.value.changePassword).toHaveBeenCalledWith({
        current_password: 'Regi123!',
        new_password: 'UjJelszo123!',
      })
    )
    await waitFor(() => {
      expect((current as HTMLInputElement).value).toBe('')
      expect((next as HTMLInputElement).value).toBe('')
      expect((confirm as HTMLInputElement).value).toBe('')
    })
  })

  it('hibás jelenlegi jelszónál megjeleníti a szerver hibaüzenetét', async () => {
    authState.value.changePassword.mockRejectedValue({ detail: 'Hibás jelenlegi jelszó', status: 400 })
    render(<SettingsPage />)
    fireEvent.click(screen.getByRole('button', { name: 'Jelszó' }))
    fireEvent.change(screen.getByLabelText('Jelenlegi jelszó'), { target: { value: 'Rossz123!' } })
    fireEvent.change(screen.getByLabelText('Új jelszó'), { target: { value: 'UjJelszo123!' } })
    fireEvent.change(screen.getByLabelText('Új jelszó megerősítése'), {
      target: { value: 'UjJelszo123!' },
    })
    const form = screen.getByRole('button', { name: 'Jelszó módosítása' }).closest('form')!
    fireEvent.submit(form)
    expect(await screen.findByText('Hibás jelenlegi jelszó')).toBeInTheDocument()
  })

  it('a törlés gomb csak pontos email egyezésnél aktív, majd töröl és navigál', async () => {
    authState.value.deleteAccount.mockResolvedValue(undefined)
    render(<SettingsPage />)
    fireEvent.click(screen.getByRole('button', { name: 'Adatok és adatvédelem' }))
    const deleteBtn = screen.getByRole('button', { name: 'Fiók végleges törlése' })
    expect(deleteBtn).toBeDisabled()
    const confirmInput = screen.getByLabelText('Email-cím megerősítése')
    fireEvent.change(confirmInput, { target: { value: 'rossz@b.hu' } })
    expect(deleteBtn).toBeDisabled()
    fireEvent.change(confirmInput, { target: { value: 'a@b.hu' } })
    expect(deleteBtn).toBeEnabled()
    fireEvent.click(deleteBtn)
    await waitFor(() => expect(authState.value.deleteAccount).toHaveBeenCalled())
    await waitFor(() => expect(navigateMock).toHaveBeenCalledWith('/'))
  })

  it('az export gomb meghívja az exportData-t és letöltést indít', async () => {
    exportDataMock.mockResolvedValue({
      gdpr_article: '20',
      export_date: '2026-07-19',
      user: {
        id: 'u1',
        email: 'a@b.hu',
        full_name: 'Teszt Elek',
        role: 'user',
        created_at: '2026-01-05T10:00:00Z',
        last_login_at: null,
      },
      diagnosis_sessions: [],
    })
    const createObjectURL = vi.fn(() => 'blob:mock')
    const revokeObjectURL = vi.fn()
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      writable: true,
      value: createObjectURL,
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      writable: true,
      value: revokeObjectURL,
    })
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    render(<SettingsPage />)
    fireEvent.click(screen.getByRole('button', { name: 'Adatok és adatvédelem' }))
    fireEvent.click(screen.getByRole('button', { name: 'Adataim letöltése (JSON)' }))
    await waitFor(() => expect(exportDataMock).toHaveBeenCalled())
    await waitFor(() => expect(clickSpy).toHaveBeenCalled())
    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob))
    clickSpy.mockRestore()
  })

  it('megjeleníti a szerepkör badge-et és a regisztráció dátumát', () => {
    render(<SettingsPage />)
    expect(screen.getByText('Felhasználó')).toBeInTheDocument()
    const expectedDate = new Date('2026-01-05T10:00:00Z').toLocaleDateString('hu-HU', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })
    expect(screen.getByText(expectedDate, { exact: false })).toBeInTheDocument()
  })
})
