import { describe, it, expect, vi } from 'vitest';
import userEvent from '@testing-library/user-event';
import { render, screen } from '../../../../test/test-utils';
import { DiagnosisForm } from '../DiagnosisForm';

/**
 * Client-side DTC validation must mirror the server rule, which lives in
 * `backend/app/core/dtc_codes.py` (`DTC_CODE_STRICT`, SAE J2012):
 * P/B/C/U, then 0-3, then 3 hex digits.
 *
 * Two failure modes are guarded here:
 *  - too strict: a real hex code ("P26B7") must not be refused locally;
 *  - too loose: a hex-shaped English word ("PEACE") must not be waved through
 *    only for the server to answer 422.
 */

/** Fill the form and submit; returns the onSubmit spy. */
async function submitWith(dtcInput: string) {
  const user = userEvent.setup();
  const onSubmit = vi.fn();

  render(<DiagnosisForm onSubmit={onSubmit} />);

  await user.type(screen.getByLabelText('DTC kód(ok)'), dtcInput);
  await user.selectOptions(screen.getByLabelText('Gyártó'), 'volkswagen');
  await user.click(screen.getByRole('button', { name: /AI MEGOLDÁS GENERÁLÁSA/i }));

  return onSubmit;
}

describe('DiagnosisForm DTC validation', () => {
  it.each(['P0300', 'P26B7', 'P090C', 'P0A94', 'B00A0', 'U0100'])(
    'accepts the real DTC code %s',
    async (code) => {
      const onSubmit = await submitWith(code);

      expect(onSubmit).toHaveBeenCalledTimes(1);
      expect(onSubmit.mock.calls[0][0]).toMatchObject({
        dtcCodes: [code],
        vehicleMake: 'volkswagen',
      });
      expect(screen.queryByText(/Érvénytelen hibakód/)).not.toBeInTheDocument();
    }
  );

  it.each(['PEACE', 'PACED', 'P9324', 'UA80E', 'U760E', 'P030', 'X0300'])(
    'rejects %s, which is not a DTC code',
    async (junk) => {
      const onSubmit = await submitWith(junk);

      expect(onSubmit).not.toHaveBeenCalled();
      expect(
        screen.getByText(`Érvénytelen hibakód(ok): ${junk}`)
      ).toBeInTheDocument();
    }
  );

  it('accepts several codes at once and normalises case', async () => {
    const onSubmit = await submitWith('p26b7, p0300 b00a0');

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0].dtcCodes).toEqual([
      'P26B7',
      'P0300',
      'B00A0',
    ]);
  });

  it('names only the offending code when the list is mixed', async () => {
    const onSubmit = await submitWith('P0300, PEACE');

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByText('Érvénytelen hibakód(ok): PEACE')).toBeInTheDocument();
  });

  it('still requires at least one code', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<DiagnosisForm onSubmit={onSubmit} />);
    await user.click(screen.getByRole('button', { name: /AI MEGOLDÁS GENERÁLÁSA/i }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(
      screen.getByText('Legalább egy hibakód megadása kötelező')
    ).toBeInTheDocument();
  });
});
