import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, within } from '../../test/test-utils';
import Layout from '../Layout';

// Mutable auth state so individual tests can flip logged-in / logged-out.
const authState = vi.hoisted(() => ({
  value: {
    user: null as { full_name?: string; email?: string } | null,
    isAuthenticated: false,
    isLoading: false,
    logout: vi.fn(),
  },
}));

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => authState.value,
}));

beforeEach(() => {
  authState.value = {
    user: null,
    isAuthenticated: false,
    isLoading: false,
    logout: vi.fn(),
  };
});

describe('Layout — grouped navigation', () => {
  it('renders the four intent groups instead of eleven flat links', () => {
    render(<Layout />);
    ['Diagnosztika', 'Garázs', 'Szerviz & Árak', 'Tudástár'].forEach((label) => {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument();
    });
  });

  it('keeps a group panel collapsed until its trigger is activated', () => {
    render(<Layout />);
    // Panel closed: the child link is not in the DOM yet.
    expect(screen.queryByRole('link', { name: /Új diagnosztika/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Diagnosztika' }));

    const newDiag = screen.getByRole('link', { name: /Új diagnosztika/ });
    expect(newDiag).toHaveAttribute('href', '/diagnosis');
    expect(screen.getByRole('link', { name: /AI Chat/ })).toHaveAttribute('href', '/chat');
  });

  it('hides auth-only entries when logged out', () => {
    render(<Layout />);
    fireEvent.click(screen.getByRole('button', { name: 'Diagnosztika' }));
    // "Előzmények" requires auth → absent when logged out.
    expect(screen.queryByRole('link', { name: /Előzmények/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Garázs' }));
    // "Járműveim" requires auth → absent; the public "Műszaki vizsga" stays.
    expect(screen.queryByRole('link', { name: /Járműveim/ })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Műszaki vizsga/ })).toHaveAttribute('href', '/inspection');
  });

  it('reveals auth-only entries when logged in', () => {
    authState.value = {
      user: { full_name: 'Barna Norbert', email: 'barna@example.com' },
      isAuthenticated: true,
      isLoading: false,
      logout: vi.fn(),
    };
    render(<Layout />);

    fireEvent.click(screen.getByRole('button', { name: 'Diagnosztika' }));
    expect(screen.getByRole('link', { name: /Előzmények/ })).toHaveAttribute('href', '/history');

    fireEvent.click(screen.getByRole('button', { name: 'Garázs' }));
    expect(screen.getByRole('link', { name: /Járműveim/ })).toHaveAttribute('href', '/garage');
  });

  it('keeps Beállítások out of the main grouped navigation', () => {
    authState.value = {
      user: { full_name: 'Barna Norbert', email: 'barna@example.com' },
      isAuthenticated: true,
      isLoading: false,
      logout: vi.fn(),
    };
    render(<Layout />);
    // Open every intent group; the flat Beállítások link must not live in the main nav.
    ['Diagnosztika', 'Garázs', 'Szerviz & Árak', 'Tudástár'].forEach((label) => {
      fireEvent.click(screen.getByRole('button', { name: label }));
    });
    const mainNav = screen.getByRole('navigation', { name: 'Fő navigáció' });
    expect(within(mainNav).queryByText('Beállítások')).not.toBeInTheDocument();
    expect(within(mainNav).queryByRole('link', { name: /Beállítások/ })).not.toBeInTheDocument();
  });

  it('exposes a Beállítások link to /settings in the desktop account menu when authenticated', () => {
    authState.value = {
      user: { full_name: 'Barna Norbert', email: 'barna@example.com' },
      isAuthenticated: true,
      isLoading: false,
      logout: vi.fn(),
    };
    render(<Layout />);
    // Account menu is collapsed initially → no settings link exposed yet.
    expect(document.querySelector('a[href="/settings"]')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Fiók menü' }));
    const settingsLink = screen.getByRole('link', { name: /Beállítások/ });
    expect(settingsLink).toHaveAttribute('href', '/settings');
  });

  it('closes an open group when Escape is pressed', () => {
    render(<Layout />);
    const trigger = screen.getByRole('button', { name: 'Diagnosztika' });
    fireEvent.click(trigger);
    expect(screen.getByRole('link', { name: /Új diagnosztika/ })).toBeInTheDocument();
    fireEvent.keyDown(trigger, { key: 'Escape' });
    expect(screen.queryByRole('link', { name: /Új diagnosztika/ })).not.toBeInTheDocument();
  });

  it('every grouped destination points at a real route', () => {
    authState.value = {
      user: { full_name: 'Barna Norbert', email: 'barna@example.com' },
      isAuthenticated: true,
      isLoading: false,
      logout: vi.fn(),
    };
    render(<Layout />);

    const realRoutes = new Set([
      '/diagnosis',
      '/history',
      '/chat',
      '/garage',
      '/inspection',
      '/services',
      '/calculator',
      '/pricing',
      '/blog',
      '/changelog',
      '/demo',
    ]);

    ['Diagnosztika', 'Garázs', 'Szerviz & Árak', 'Tudástár'].forEach((label) => {
      const trigger = screen.getByRole('button', { name: label });
      fireEvent.click(trigger);
      const panel = document.getElementById(trigger.getAttribute('aria-controls') || '');
      expect(panel).not.toBeNull();
      within(panel as HTMLElement)
        .getAllByRole('link')
        .forEach((link) => {
          expect(realRoutes.has(link.getAttribute('href') || '')).toBe(true);
        });
    });
  });
});
