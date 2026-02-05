import { useState } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import {
  Menu,
  X,
  Plus,
  History,
  Settings,
  User,
  Wrench,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export interface NavItem {
  label: string;
  href: string;
  icon?: React.ReactNode;
}

export interface HeaderProps {
  /** Custom logo component or text */
  logo?: React.ReactNode;
  /** Navigation items */
  navigation?: NavItem[];
  /** Right-side actions (user menu, etc.) */
  actions?: React.ReactNode;
  /** Additional className */
  className?: string;
}

const defaultNavigation: NavItem[] = [
  { label: 'Új diagnosztika', href: '/diagnosis', icon: <Plus className="h-4 w-4" /> },
  { label: 'Előzmények', href: '/history', icon: <History className="h-4 w-4" /> },
  { label: 'Beállítások', href: '/settings', icon: <Settings className="h-4 w-4" /> },
  { label: 'Profil', href: '/profile', icon: <User className="h-4 w-4" /> },
];

/**
 * Main application header with responsive navigation.
 *
 * @example
 * ```tsx
 * <Header />
 *
 * <Header
 *   navigation={[
 *     { label: 'Kezdőlap', href: '/' },
 *     { label: 'Diagnosztika', href: '/diagnosis' },
 *   ]}
 * />
 * ```
 */
export function Header({
  logo,
  navigation = defaultNavigation,
  actions,
  className,
}: HeaderProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);
  const closeMobileMenu = () => setIsMobileMenuOpen(false);

  return (
    <header
      className={cn(
        'sticky top-0 z-sticky bg-background border-b border-border',
        className
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-header-mobile md:h-header">
          {/* Logo */}
          <Link
            to="/"
            className="flex items-center gap-2 font-semibold text-lg"
            onClick={closeMobileMenu}
          >
            {logo || (
              <>
                <Wrench className="h-6 w-6 text-primary-600" />
                <span className="hidden sm:inline">
                  <span className="text-primary-600">MechanicAI</span>
                  <span className="text-foreground"> PRO</span>
                </span>
                <span className="sm:hidden text-primary-600">MechanicAI</span>
              </>
            )}
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {navigation.map((item) => (
              <NavLink
                key={item.href}
                to={item.href}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg',
                    'transition-colors duration-fast',
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
            {actions}

            {/* Mobile menu button */}
            <button
              type="button"
              className="md:hidden p-2 rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
              onClick={toggleMobileMenu}
              aria-expanded={isMobileMenuOpen}
              aria-label={isMobileMenuOpen ? 'Menü bezárása' : 'Menü megnyitása'}
            >
              {isMobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {isMobileMenuOpen && (
        <div className="md:hidden border-t border-border bg-background">
          <nav className="px-4 py-2 space-y-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <NavLink
                  key={item.href}
                  to={item.href}
                  onClick={closeMobileMenu}
                  className={cn(
                    'flex items-center gap-3 px-4 py-3 text-sm font-medium rounded-lg',
                    'transition-colors duration-fast',
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
        </div>
      )}
    </header>
  );
}

export default Header;
