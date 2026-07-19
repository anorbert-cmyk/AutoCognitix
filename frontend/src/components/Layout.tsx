import { Outlet, Link, NavLink, useLocation } from 'react-router-dom';
import {
  Menu,
  X,
  Plus,
  History,
  User,
  Wrench,
  LogOut,
  LogIn,
  UserPlus,
  Car,
  ClipboardCheck,
  Calculator,
  MessageSquare,
  MapPin,
  Clock,
  CreditCard,
  BookOpen,
  Sparkles,
  Settings,
} from 'lucide-react';
import { useRef, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { cn } from '@/lib/utils';
import { Button } from '@/components/lib';
import NavDropdown, { type NavChild } from './NavDropdown';

interface NavGroup {
  label: string;
  children: NavChild[];
}

// Eleven flat links collapsed into four intent-based groups. Every `href`
// maps to a real route in App.tsx — no dead destinations.
const navGroups: NavGroup[] = [
  {
    label: 'Diagnosztika',
    children: [
      { label: 'Új diagnosztika', href: '/diagnosis', icon: <Plus className="h-4 w-4" />, lead: true },
      { label: 'Előzmények', href: '/history', icon: <History className="h-4 w-4" />, requiresAuth: true },
      { label: 'AI Chat', href: '/chat', icon: <MessageSquare className="h-4 w-4" /> },
    ],
  },
  {
    label: 'Garázs',
    children: [
      { label: 'Járműveim', href: '/garage', icon: <Car className="h-4 w-4" />, requiresAuth: true },
      { label: 'Műszaki vizsga', href: '/inspection', icon: <ClipboardCheck className="h-4 w-4" /> },
    ],
  },
  {
    label: 'Szerviz & Árak',
    children: [
      { label: 'Szervizkereső', href: '/services', icon: <MapPin className="h-4 w-4" /> },
      { label: 'Költségkalkulátor', href: '/calculator', icon: <Calculator className="h-4 w-4" /> },
      { label: 'Árazás', href: '/pricing', icon: <CreditCard className="h-4 w-4" /> },
    ],
  },
  {
    label: 'Tudástár',
    children: [
      { label: 'Blog', href: '/blog', icon: <BookOpen className="h-4 w-4" /> },
      { label: 'Changelog', href: '/changelog', icon: <Clock className="h-4 w-4" /> },
      { label: 'Demo', href: '/demo', icon: <Sparkles className="h-4 w-4" /> },
    ],
  },
];

const isPathActive = (pathname: string, href: string) =>
  pathname === href || pathname.startsWith(href + '/');

export default function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const { user, isAuthenticated, logout, isLoading } = useAuth();
  const location = useLocation();
  const accountRef = useRef<HTMLDivElement>(null);
  const accountBtnRef = useRef<HTMLButtonElement>(null);

  const handleLogout = async () => {
    if (!window.confirm('Biztosan ki szeretne jelentkezni?')) return;
    await logout();
    setUserMenuOpen(false);
    setMobileMenuOpen(false);
  };

  const closeMobileMenu = () => setMobileMenuOpen(false);
  const toggleMobileMenu = () => setMobileMenuOpen(!mobileMenuOpen);

  // Hide auth-only entries when logged out, then drop any group left empty.
  const visibleGroups = navGroups
    .map((group) => ({
      ...group,
      children: group.children.filter((item) => !item.requiresAuth || isAuthenticated),
    }))
    .filter((group) => group.children.length > 0);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-sticky bg-background border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14 md:h-16">
            {/* Logo */}
            <Link
              to="/"
              className="flex items-center gap-2 font-semibold text-lg"
              onClick={closeMobileMenu}
            >
              <Wrench className="h-6 w-6 text-primary-600" />
              <span className="hidden sm:inline">
                <span className="text-primary-600">MechanicAI</span>
                <span className="text-foreground"> PRO</span>
              </span>
              <span className="sm:hidden text-primary-600">MechanicAI</span>
            </Link>

            {/* Desktop Navigation — four grouped dropdowns */}
            <nav className="hidden md:flex items-center gap-1" aria-label="Fő navigáció">
              {visibleGroups.map((group) => (
                <NavDropdown key={group.label} label={group.label} items={group.children} />
              ))}
            </nav>

            {/* Right side actions */}
            <div className="flex items-center gap-2">
              {/* Desktop Auth */}
              <div className="hidden md:flex items-center gap-2">
                {isAuthenticated ? (
                  <div
                    className="relative"
                    ref={accountRef}
                    onKeyDown={(e) => {
                      if (e.key === 'Escape' && userMenuOpen) {
                        e.stopPropagation();
                        setUserMenuOpen(false);
                        accountBtnRef.current?.focus();
                      }
                    }}
                    onBlur={(e) => {
                      if (!accountRef.current?.contains(e.relatedTarget as Node)) setUserMenuOpen(false);
                    }}
                  >
                    <button
                      ref={accountBtnRef}
                      onClick={() => setUserMenuOpen(!userMenuOpen)}
                      aria-label="Fiók menü"
                      aria-expanded={userMenuOpen}
                      className={cn(
                        'flex items-center gap-2 px-3 py-2 rounded-lg',
                        'text-sm font-medium transition-colors',
                        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500',
                        userMenuOpen
                          ? 'bg-primary-50 text-primary-700'
                          : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                      )}
                    >
                      <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                        <User className="h-5 w-5 text-primary-600" aria-hidden="true" />
                      </div>
                      <span className="hidden lg:inline max-w-[120px] truncate">
                        {user?.full_name || user?.email || 'Felhasználó'}
                      </span>
                    </button>

                    {/* Account menu */}
                    {userMenuOpen && (
                      <>
                        <div
                          className="fixed inset-0 z-40"
                          onClick={() => setUserMenuOpen(false)}
                        />
                        <div className="absolute right-0 mt-2 w-56 bg-card rounded-xl shadow-lg py-1 z-50 border border-border">
                          <div className="px-4 py-3 border-b border-border">
                            <p className="text-xs uppercase tracking-wider text-muted-foreground/70">Fiók</p>
                            <p className="mt-1 text-sm font-medium text-foreground truncate">
                              {user?.full_name || 'Felhasználó'}
                            </p>
                            <p className="text-xs text-muted-foreground truncate">
                              {user?.email}
                            </p>
                          </div>
                          <div className="py-1">
                            <Link
                              to="/settings"
                              onClick={() => setUserMenuOpen(false)}
                              className="w-full flex items-center gap-2 px-4 py-2 text-sm text-foreground hover:bg-muted transition-colors focus-visible:outline-none focus-visible:bg-muted"
                            >
                              <Settings className="h-4 w-4" aria-hidden="true" />
                              Beállítások
                            </Link>
                            <button
                              onClick={handleLogout}
                              disabled={isLoading}
                              className="w-full flex items-center gap-2 px-4 py-2 text-sm text-status-error hover:bg-muted transition-colors focus-visible:outline-none focus-visible:bg-muted"
                            >
                              <LogOut className="h-4 w-4" aria-hidden="true" />
                              Kijelentkezés
                            </button>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <>
                    <Link to="/login">
                      <Button variant="ghost" size="sm">
                        <LogIn className="h-4 w-4 mr-2" />
                        Bejelentkezés
                      </Button>
                    </Link>
                    <Link to="/register">
                      <Button variant="primary" size="sm">
                        <UserPlus className="h-4 w-4 mr-2" />
                        Regisztráció
                      </Button>
                    </Link>
                  </>
                )}
              </div>

              {/* Mobile menu button */}
              <button
                type="button"
                className="md:hidden p-2 rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                onClick={toggleMobileMenu}
                aria-expanded={mobileMenuOpen}
                aria-controls="mobile-nav"
                aria-label={mobileMenuOpen ? 'Menü bezárása' : 'Menü megnyitása'}
              >
                {mobileMenuOpen ? (
                  <X className="h-6 w-6" />
                ) : (
                  <Menu className="h-6 w-6" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Menu — grouped sections */}
        {mobileMenuOpen && (
          <div id="mobile-nav" className="md:hidden border-t border-border bg-background">
            <nav className="px-4 py-3 space-y-4" aria-label="Mobil navigáció">
              {visibleGroups.map((group) => (
                <div key={group.label} className="space-y-1">
                  <p className="px-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground/70">
                    {group.label}
                  </p>
                  {group.children.map((item) => {
                    const active = isPathActive(location.pathname, item.href);
                    return (
                      <NavLink
                        key={item.href}
                        to={item.href}
                        onClick={closeMobileMenu}
                        className={cn(
                          'flex items-center gap-3 px-4 py-3 text-sm font-medium rounded-lg',
                          'transition-colors duration-150',
                          active
                            ? 'bg-primary-50 text-primary-700'
                            : item.lead
                              ? 'text-primary-700 hover:bg-primary-50'
                              : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                        )}
                      >
                        {item.icon}
                        {item.label}
                      </NavLink>
                    );
                  })}
                </div>
              ))}
            </nav>

            {/* Mobile auth section */}
            <div className="px-4 py-4 border-t border-border">
              {isAuthenticated ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-3 px-4 py-2">
                    <div className="h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
                      <User className="h-6 w-6 text-primary-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">
                        {user?.full_name || 'Felhasználó'}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">
                        {user?.email}
                      </p>
                    </div>
                  </div>
                  <Link to="/settings" onClick={closeMobileMenu}>
                    <Button
                      variant="outline"
                      fullWidth
                      leftIcon={<Settings className="h-4 w-4" />}
                    >
                      Beállítások
                    </Button>
                  </Link>
                  <Button
                    variant="outline"
                    fullWidth
                    onClick={handleLogout}
                    disabled={isLoading}
                    leftIcon={<LogOut className="h-4 w-4" />}
                    className="text-status-error border-status-error hover:bg-status-error/10"
                  >
                    Kijelentkezés
                  </Button>
                </div>
              ) : (
                <div className="space-y-2">
                  <Link to="/login" onClick={closeMobileMenu}>
                    <Button variant="outline" fullWidth leftIcon={<LogIn className="h-4 w-4" />}>
                      Bejelentkezés
                    </Button>
                  </Link>
                  <Link to="/register" onClick={closeMobileMenu}>
                    <Button variant="primary" fullWidth leftIcon={<UserPlus className="h-4 w-4" />}>
                      Regisztráció
                    </Button>
                  </Link>
                </div>
              )}
            </div>
          </div>
        )}
      </header>

      {/* Main content */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="bg-card border-t border-border mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-2">
              <Wrench className="h-5 w-5 text-primary-600" />
              <span className="font-semibold text-foreground">MechanicAI PRO</span>
            </div>
            <p className="text-sm text-muted-foreground">
              © {new Date().getFullYear()} MechanicAI PRO. AI-alapú gépjármű-diagnosztika.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
