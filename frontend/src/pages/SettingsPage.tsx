/**
 * SettingsPage — Felhasználói beállítások
 *
 * Három fül: Profil, Jelszó, Adatok és adatvédelem (GDPR export + fióktörlés).
 * Védett útvonal: /settings
 */

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertTriangle, Download, ShieldCheck, Trash2 } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { useToast } from '../contexts/ToastContext'
import { Badge, Button, Input } from '../components/lib'
import { PasswordStrengthMeter } from '../components/ui'
import { exportData, type User } from '../services/authService'
import { ApiError } from '../services/api'

// =============================================================================
// Constants
// =============================================================================

type SettingsTab = 'profil' | 'jelszo' | 'adatvedelem'

const TABS: ReadonlyArray<{ id: SettingsTab; label: string }> = [
  { id: 'profil', label: 'Profil' },
  { id: 'jelszo', label: 'Jelszó' },
  { id: 'adatvedelem', label: 'Adatok és adatvédelem' },
]

const ROLE_HU: Record<User['role'], string> = {
  user: 'Felhasználó',
  mechanic: 'Szerelő',
  admin: 'Adminisztrátor',
}

// A jelszó-követelmények a RegisterPage szabályaival azonosak (replikálva, mivel
// a RegisterPage nem exportálja őket).
const PASSWORD_REQUIREMENTS: ReadonlyArray<(p: string) => boolean> = [
  (p) => p.length >= 8,
  (p) => /[A-Z]/.test(p),
  (p) => /[a-z]/.test(p),
  (p) => /\d/.test(p),
  // eslint-disable-next-line no-useless-escape
  (p) => /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(p),
]

const focusRing =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2'

// =============================================================================
// SettingsPage
// =============================================================================

export default function SettingsPage() {
  const { user, updateProfile, changePassword, deleteAccount } = useAuth()
  const toast = useToast()
  const navigate = useNavigate()

  const [activeTab, setActiveTab] = useState<SettingsTab>('profil')

  // Profil űrlap
  const [fullName, setFullName] = useState(user?.full_name ?? '')
  const [email, setEmail] = useState(user?.email ?? '')
  const [emailErr, setEmailErr] = useState<string | null>(null)
  const [savingProfile, setSavingProfile] = useState(false)

  // Jelszó űrlap
  const [currentPw, setCurrentPw] = useState('')
  const [newPw, setNewPw] = useState('')
  const [confPw, setConfPw] = useState('')
  const [pwErr, setPwErr] = useState<string | null>(null)
  const [savingPw, setSavingPw] = useState(false)

  // Adatvédelem
  const [exporting, setExporting] = useState(false)
  const [confirmEmail, setConfirmEmail] = useState('')
  const [deleting, setDeleting] = useState(false)

  // A ProtectedRoute garantálja a bejelentkezett felhasználót; ez csak TS-védelem.
  if (!user) {
    return null
  }

  const registeredLabel = user.created_at
    ? new Date(user.created_at).toLocaleDateString('hu-HU', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      })
    : '—'

  const emailMatches =
    confirmEmail.trim().toLowerCase() === user.email.toLowerCase()

  // ── Profil ──────────────────────────────────────────────────────────────────

  const handleProfileSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setEmailErr(null)
    setSavingProfile(true)
    try {
      await updateProfile({
        full_name: fullName || undefined,
        email: email.toLowerCase(),
      })
      toast.success('Profil frissítve')
    } catch (err) {
      const detail = (err as ApiError).detail || 'A profil frissítése nem sikerült'
      setEmailErr(detail)
      toast.error(detail)
    } finally {
      setSavingProfile(false)
    }
  }

  // ── Jelszó ──────────────────────────────────────────────────────────────────

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setPwErr(null)

    const rulesPass = PASSWORD_REQUIREMENTS.every((test) => test(newPw))
    if (!rulesPass) {
      setPwErr('A jelszó nem felel meg a követelményeknek')
      return
    }
    if (newPw !== confPw) {
      setPwErr('A jelszavak nem egyeznek')
      return
    }

    setSavingPw(true)
    try {
      await changePassword({ current_password: currentPw, new_password: newPw })
      toast.success('A jelszó sikeresen megváltozott')
      setCurrentPw('')
      setNewPw('')
      setConfPw('')
    } catch (err) {
      setPwErr((err as ApiError).detail || 'A jelszó megváltoztatása nem sikerült')
    } finally {
      setSavingPw(false)
    }
  }

  // ── Adatvédelem ─────────────────────────────────────────────────────────────

  const handleExport = async () => {
    setExporting(true)
    try {
      const data = await exportData()
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json',
      })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `autocognitix-adatexport-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
      URL.revokeObjectURL(url)
      toast.success('Adatexport letöltve')
    } catch {
      toast.error('Az exportálás nem sikerült')
    } finally {
      setExporting(false)
    }
  }

  const handleDelete = async () => {
    setDeleting(true)
    try {
      await deleteAccount()
      toast.success('Fiók törölve')
      navigate('/')
    } catch {
      toast.error('A törlés nem sikerült. Kérjük próbálja újra később.')
      setDeleting(false)
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-3xl mx-auto px-4 py-8 md:py-12">
        {/* Fejléc */}
        <header className="mb-8">
          <h1 className="text-2xl font-bold text-foreground">Beállítások</h1>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <Badge variant="info">{ROLE_HU[user.role]}</Badge>
            <span className="text-sm text-muted-foreground">
              Regisztráció: {registeredLabel}
            </span>
          </div>
        </header>

        {/* Fül navigáció */}
        <div className="flex gap-1 p-1 bg-slate-100 rounded-xl mb-6 w-fit">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              aria-pressed={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`h-9 px-5 rounded-lg text-sm font-semibold transition-colors ${focusRing} ${
                activeTab === tab.id
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* ── Profil fül ─────────────────────────────────────────────────────── */}
        {activeTab === 'profil' && (
          <section className="bg-card border border-border rounded-xl p-6">
            <form onSubmit={handleProfileSubmit} className="space-y-5">
              <Input
                label="Teljes név"
                type="text"
                autoComplete="name"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="pl. Kovács János"
              />
              <Input
                label="Email cím"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                error={emailErr ?? undefined}
              />
              <div className="flex justify-end">
                <Button type="submit" isLoading={savingProfile}>
                  Mentés
                </Button>
              </div>
            </form>
          </section>
        )}

        {/* ── Jelszó fül ─────────────────────────────────────────────────────── */}
        {activeTab === 'jelszo' && (
          <section className="bg-card border border-border rounded-xl p-6">
            <form onSubmit={handlePasswordSubmit} className="space-y-5">
              <Input
                label="Jelenlegi jelszó"
                type="password"
                autoComplete="current-password"
                value={currentPw}
                onChange={(e) => setCurrentPw(e.target.value)}
              />
              <div className="space-y-2">
                <Input
                  label="Új jelszó"
                  type="password"
                  autoComplete="new-password"
                  value={newPw}
                  onChange={(e) => setNewPw(e.target.value)}
                />
                <PasswordStrengthMeter password={newPw} />
              </div>
              <Input
                label="Új jelszó megerősítése"
                type="password"
                autoComplete="new-password"
                value={confPw}
                onChange={(e) => setConfPw(e.target.value)}
              />
              {pwErr && (
                <p role="alert" className="text-sm text-red-600">
                  {pwErr}
                </p>
              )}
              <div className="flex justify-end">
                <Button type="submit" isLoading={savingPw}>
                  Jelszó módosítása
                </Button>
              </div>
            </form>
          </section>
        )}

        {/* ── Adatok és adatvédelem fül ──────────────────────────────────────── */}
        {activeTab === 'adatvedelem' && (
          <div className="space-y-6">
            {/* Adatexport kártya */}
            <section className="bg-card border border-border rounded-xl p-6">
              <div className="flex items-start gap-3">
                <div className="h-10 w-10 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
                  <ShieldCheck className="h-5 w-5 text-foreground" aria-hidden="true" />
                </div>
                <div className="flex-1">
                  <h2 className="text-base font-semibold text-foreground">
                    Adatok letöltése
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    A GDPR 20. cikke (adathordozhatóság) alapján letöltheti minden
                    személyes adatát és diagnosztikai előzményét géppel olvasható JSON
                    formátumban.
                  </p>
                  <div className="mt-4">
                    <Button
                      type="button"
                      variant="outline"
                      leftIcon={<Download className="h-4 w-4" />}
                      isLoading={exporting}
                      onClick={handleExport}
                    >
                      Adataim letöltése (JSON)
                    </Button>
                  </div>
                </div>
              </div>
            </section>

            {/* Fióktörlés kártya (veszélyzóna) */}
            <section className="bg-destructive/5 border border-destructive/40 rounded-xl p-6">
              <div className="flex items-start gap-3">
                <div className="h-10 w-10 rounded-lg bg-destructive/10 flex items-center justify-center flex-shrink-0">
                  <AlertTriangle className="h-5 w-5 text-destructive" aria-hidden="true" />
                </div>
                <div className="flex-1">
                  <h2 className="text-base font-semibold text-foreground">Fiók törlése</h2>
                  <p className="mt-1 text-sm text-foreground/80">
                    A fiók végleges törlése nem vonható vissza. Minden adata törlődik.
                    Írja be az email-címét a megerősítéshez.
                  </p>
                  <div className="mt-4 space-y-3">
                    <Input
                      label="Email-cím megerősítése"
                      type="email"
                      autoComplete="off"
                      value={confirmEmail}
                      onChange={(e) => setConfirmEmail(e.target.value)}
                      placeholder={user.email}
                    />
                    <Button
                      type="button"
                      variant="destructive"
                      leftIcon={<Trash2 className="h-4 w-4" />}
                      disabled={!emailMatches || deleting}
                      isLoading={deleting}
                      onClick={handleDelete}
                    >
                      Fiók végleges törlése
                    </Button>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  )
}
