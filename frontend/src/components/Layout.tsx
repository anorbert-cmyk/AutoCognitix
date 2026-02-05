import { Outlet, Link, NavLink, useLocation } from 'react-router-dom';
import {
  Menu,
  X,
  Plus,
  History,
  Settings,
  User,
  Wrench,
  LogOut,
  LogIn,
  UserPlus,
} from 'lucide-react';
import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { cn } from '@/lib/utils';
import { Button } from '@/components/lib';

interface NavItem {
  label: string;
  href: string;
  icon?: React.ReactNode;
  requiresAuth?: boolean;
}

const navigationItems: NavItem[] = [
  { label: 'Új diagnosztika', href: '/diagnosis', icon: <Plus className="h-4 w-4" /> },
  { label: 'Előzmények', href: '/history', icon: <History className="h-4 w-4" />, requiresAuth: true },
  { label: 'Beállítások', href: '/settings', icon: <Settings className="h-4 w-4" /> },
];

export default function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const { user, isAuthenticated, logout, isLoading } = useAuth();
  const location = useLocation();

  const handleLogout = async () => {
    await logout();
    setUserMenuOpen(false);
    setMobileMenuOpen(false);
  };

  const closeMobileMenu = () => setMobileMenuOpen(false);
  const toggleMobileMenu = () => setMobileMenuOpen(!mobileMenuOpen);

  // Filter navigation based on auth state
  const filteredNavigation = navigationItems.filter(
    (item) => !item.requiresAuth || isAuthenticated
  );

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

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center gap-1">
              {filteredNavigation.map((item) => (
                <NavLink
                  key={item.href}
                  to={item.href}
                  className={({ isActive }) =>
                    cn(
                      'flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg',
                      'transition-colors duration-150',
                      isActive
                        ? 'bg-primary-50 text-primary-700'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                    )
                  }
                >
                  {item.icon}
                  {item.label}
                </NavLink>
              ))}
            </nav>

            {/* Right side actions */}
            <div className="flex items-center gap-2">
              {/* Desktop Auth */}
              <div className="hidden md:flex items-center gap-2">
                {isAuthenticated ? (
                  <div className="relative">
                    <button
                      onClick={() => setUserMenuOpen(!userMenuOpen)}
                      className={cn(
                        'flex items-center gap-2 px-3 py-2 rounded-lg',
                        'text-sm font-medium transition-colors',
                        userMenuOpen
                          ? 'bg-primary-50 text-primary-700'
                          : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                      )}
                    >
                      <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                        <User className="h-5 w-5 text-primary-600" />
                      </div>
                      <span className="hidden lg:inline max-w-[120px] truncate">
                        {user?.full_name || user?.email || 'Felhasználó'}
                      </span>
                    </button>

                    {/* User dropdown menu */}
                    {userMenuOpen && (
                      <>
                        <div
                          className="fixed inset-0 z-40"
                          onClick={() => setUserMenuOpen(false)}
                        />
                        <div className="absolute right-0 mt-2 w-56 bg-card rounded-lg shadow-lg py-1 z-50 border border-border">
                          <div className="px-4 py-3 border-b border-border">
                            <p className="text-sm font-medium text-foreground truncate">
                              {user?.full_name || 'Felhasználó'}
                            </p>
                            <p className="text-xs text-muted-foreground truncate">
                              {user?.email}
                            </p>
                          </div>
                          <div className="py-1">
                            <Link
                              to="/profile"
                              className="flex items-center gap-2 px-4 py-2 text-sm text-foreground hover:bg-muted"
                              onClick={() => setUserMenuOpen(false)}
                            >
                              <User className="h-4 w-4" />
                              Profil
                            </Link>
                            <Link
                              to="/history"
                              className="flex items-center gap-2 px-4 py-2 text-sm text-foreground hover:bg-muted"
                              onClick={() => setUserMenuOpen(false)}
                            >
                              <History className="h-4 w-4" />
                              Diagnosztika előzmények
                            </Link>
                          </div>
                          <div className="py-1 border-t border-border">
                            <button
                              onClick={handleLogout}
                              disabled={isLoading}
                              className="w-full flex items-center gap-2 px-4 py-2 text-sm text-status-error hover:bg-muted"
                            >
                              <LogOut className="h-4 w-4" />
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

        {/* Mobile Navigation Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border bg-background">
            <nav className="px-4 py-2 space-y-1">
              {filteredNavigation.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <NavLink
                    key={item.href}
                    to={item.href}
                    onClick={closeMobileMenu}
                    className={cn(
                      'flex items-center gap-3 px-4 py-3 text-sm font-medium rounded-lg',
                      'transition-colors duration-150',
                      isActive
                        ? 'bg-primary-50 text-primary-700'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                    )}
                  >
                    {item.icon}
                    {item.label}
                  </NavLink>
                );
              })}
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
