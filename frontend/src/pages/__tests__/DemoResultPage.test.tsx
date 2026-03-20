import { describe, it, expect } from 'vitest';
import { render, screen } from '../../test/test-utils';
import DemoResultPage from '../DemoResultPage';
import { demoDiagnosisResponse, demoParts } from '../../data/demoData';

describe('DemoResultPage', () => {
  it('renders the page with VW Golf VII vehicle info', () => {
    render(<DemoResultPage />);
    expect(screen.getByText('Volkswagen Golf VII')).toBeInTheDocument();
    expect(screen.getByText(/1\.4 TSI 125 LE/)).toBeInTheDocument();
    expect(screen.getByText('98.420 km')).toBeInTheDocument();
    expect(screen.getByText('Benzin')).toBeInTheDocument();
    expect(screen.getByText('DSG-7 automata')).toBeInTheDocument();
  });

  it('shows P0300 as the primary DTC code', () => {
    render(<DemoResultPage />);
    // P0300 appears multiple times (primary box + probable causes)
    const p0300Elements = screen.getAllByText('P0300');
    expect(p0300Elements.length).toBeGreaterThanOrEqual(1);
  });

  it('shows P0301 and P0304 as secondary DTC codes', () => {
    render(<DemoResultPage />);
    // These codes appear in the DTC box and in probable causes sections
    const p0301 = screen.getAllByText('P0301');
    expect(p0301.length).toBeGreaterThanOrEqual(1);
    const p0304 = screen.getAllByText('P0304');
    expect(p0304.length).toBeGreaterThanOrEqual(1);
  });

  it('displays all 6 parts with names', () => {
    render(<DemoResultPage />);
    for (const part of demoParts) {
      // Some part names may appear in multiple places (card + repair steps)
      const elements = screen.getAllByText(part.name);
      expect(elements.length).toBeGreaterThanOrEqual(1);
    }
  });

  it('shows store names in parts section', () => {
    render(<DemoResultPage />);
    // Each store name appears multiple times (once per part card), check at least one
    const bardiElements = screen.getAllByText('Bárdi Autó');
    expect(bardiElements.length).toBeGreaterThanOrEqual(1);

    const unixElements = screen.getAllByText('Unix Autó');
    expect(unixElements.length).toBeGreaterThanOrEqual(1);

    const autodocElements = screen.getAllByText('AUTODOC');
    expect(autodocElements.length).toBeGreaterThanOrEqual(1);
  });

  it('renders repair steps', () => {
    render(<DemoResultPage />);
    for (const repair of demoDiagnosisResponse.recommended_repairs) {
      expect(screen.getByText(repair.title)).toBeInTheDocument();
    }
  });

  it('shows the confidence score (87%)', () => {
    render(<DemoResultPage />);
    const percentage = Math.round(demoDiagnosisResponse.confidence_score * 100);
    expect(screen.getByText(`${percentage}%`)).toBeInTheDocument();
  });

  it('shows the demo banner indicator', () => {
    render(<DemoResultPage />);
    expect(
      screen.getByText(/Bemutató mód/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Szimulált P0300 hibakód/),
    ).toBeInTheDocument();
  });

  it('shows the "Demo" badge in the header', () => {
    render(<DemoResultPage />);
    expect(screen.getByText('Demo')).toBeInTheDocument();
  });

  it('renders the bottom bar demo notice', () => {
    render(<DemoResultPage />);
    expect(screen.getByText(/bemutató jelentés/)).toBeInTheDocument();
  });

  it('shows probable causes with confidence percentages', () => {
    render(<DemoResultPage />);
    for (const cause of demoDiagnosisResponse.probable_causes) {
      // Cause titles may appear more than once (e.g. first cause also used as dtcDescription)
      const titleElements = screen.getAllByText(cause.title);
      expect(titleElements.length).toBeGreaterThanOrEqual(1);
    }
    // Check that confidence percentages are rendered
    // 92%, 78%, 45% from probable_causes, plus 87% from confidence_score
    expect(screen.getByText('92%')).toBeInTheDocument();
    expect(screen.getByText('78%')).toBeInTheDocument();
    expect(screen.getByText('45%')).toBeInTheDocument();
  });

  it('shows the total cost estimate section', () => {
    render(<DemoResultPage />);
    expect(
      screen.getByText('Becsült Javítási Összköltség'),
    ).toBeInTheDocument();
  });

  it('renders "Szükséges alkatrészek" section header', () => {
    render(<DemoResultPage />);
    expect(
      screen.getByText('Szükséges alkatrészek'),
    ).toBeInTheDocument();
  });

  it('renders sources section', () => {
    render(<DemoResultPage />);
    expect(
      screen.getByText('Felhasznált források'),
    ).toBeInTheDocument();
    for (const source of demoDiagnosisResponse.sources) {
      expect(screen.getByText(source.title)).toBeInTheDocument();
    }
  });

  it('displays the vehicle VIN', () => {
    render(<DemoResultPage />);
    expect(screen.getByText('WVWZZZAUZJW123456')).toBeInTheDocument();
  });

  it('displays the license plate', () => {
    render(<DemoResultPage />);
    expect(screen.getByText('ABC-123')).toBeInTheDocument();
  });
});
