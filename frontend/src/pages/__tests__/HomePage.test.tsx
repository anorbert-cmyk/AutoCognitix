import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../test/test-utils';
import HomePage from '../HomePage';

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ user: null, isAuthenticated: false, isLoading: false }),
}));

vi.mock('../../services/hooks/useGarage', () => ({
  useUpcomingReminders: () => ({ data: undefined, isLoading: false }),
}));

describe('HomePage', () => {
  it('should render the hero section with title', () => {
    render(<HomePage />);
    expect(
      screen.getByText('AI-alapú Gépjármű Diagnosztika'),
    ).toBeInTheDocument();
  });

  it('should render the features section', () => {
    render(<HomePage />);
    expect(screen.getByText('Hogyan működik?')).toBeInTheDocument();
    expect(screen.getByText('Hibakód bevitel')).toBeInTheDocument();
    expect(screen.getByText('AI elemzés')).toBeInTheDocument();
    expect(screen.getByText('Részletes diagnózis')).toBeInTheDocument();
  });

  it('should render diagnosis CTA links', () => {
    render(<HomePage />);
    const ctaLinks = screen.getAllByText('Diagnózis indítása');
    expect(ctaLinks.length).toBeGreaterThanOrEqual(2);
    ctaLinks.forEach((link) => {
      expect(link.closest('a')).toHaveAttribute('href', '/diagnosis');
    });
  });

  it('should render the CTA section', () => {
    render(<HomePage />);
    expect(
      screen.getByText('Készen áll a diagnosztikára?'),
    ).toBeInTheDocument();
  });
});
