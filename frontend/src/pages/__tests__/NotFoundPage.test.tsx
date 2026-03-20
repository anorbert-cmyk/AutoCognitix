import { describe, it, expect } from 'vitest';
import { render, screen } from '../../test/test-utils';
import NotFoundPage from '../NotFoundPage';

describe('NotFoundPage', () => {
  it('should render 404 message', () => {
    render(<NotFoundPage />);
    expect(screen.getByText('404')).toBeInTheDocument();
  });

  it('should display Hungarian "not found" message', () => {
    render(<NotFoundPage />);
    expect(screen.getByText('Az oldal nem található')).toBeInTheDocument();
  });

  it('should display Hungarian description text', () => {
    render(<NotFoundPage />);
    expect(
      screen.getByText('A keresett oldal nem létezik vagy el lett távolítva.'),
    ).toBeInTheDocument();
  });

  it('should render a link to the home page', () => {
    render(<NotFoundPage />);
    const homeLink = screen.getByText('Kezdőlap');
    expect(homeLink).toBeInTheDocument();
    expect(homeLink.closest('a')).toHaveAttribute('href', '/');
  });

  it('should render a link to start a new diagnosis', () => {
    render(<NotFoundPage />);
    const diagnosisLink = screen.getByText('Új diagnózis');
    expect(diagnosisLink).toBeInTheDocument();
    expect(diagnosisLink.closest('a')).toHaveAttribute('href', '/diagnosis');
  });
});
