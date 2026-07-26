import { describe, it, expect, vi, beforeEach } from 'vitest';

// =============================================================================
// Mock the api default export (per-file mock pattern — test-utils.tsx
// deliberately provides no shared providers/mocks)
// =============================================================================

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>();
  return {
    ...actual,
    default: {
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
    },
  };
});

describe('dtcService', () => {
  let mockApi: { get: ReturnType<typeof vi.fn>; post: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    vi.clearAllMocks();
    const apiModule = await import('../api');
    mockApi = apiModule.default as unknown as typeof mockApi;
  });

  // ===========================================================================
  // isValidDTCFormat — canonical rule mirrored from
  // backend/app/core/dtc_codes.py (DTC_CODE_STRICT, SAE J2012):
  //   ^[PBCU][0-3][0-9A-F]{3}$
  // ===========================================================================

  describe('isValidDTCFormat', () => {
    it('should accept real HEX manufacturer codes a scan tool reports', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      // These are the codes the old decimal-only /^\d{4}$/ tail rejected,
      // which made the detail page unreachable for the user.
      for (const code of ['P26B7', 'P090C', 'P0A94', 'B00A0', 'U0100', 'P17F1', 'P324E']) {
        expect(isValidDTCFormat(code), `${code} must be valid`).toBe(true);
      }
    });

    it('should accept plain decimal codes across all four system prefixes', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      for (const code of ['P0300', 'P0101', 'B1234', 'C0567', 'U0100']) {
        expect(isValidDTCFormat(code), `${code} must be valid`).toBe(true);
      }
    });

    it('should accept every legal second character 0-3 (SAE generic + manufacturer)', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      for (const code of ['P0300', 'P1300', 'P2300', 'P3300']) {
        expect(isValidDTCFormat(code), `${code} must be valid`).toBe(true);
      }
    });

    it('should reject English words that are P/B/C/U followed by hex letters', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      // The [0-3] second-character constraint is what kills these — a plain
      // [PBCU][0-9A-F]{4} pattern would accept them.
      for (const code of ['PEACE', 'PACED', 'BEEDA']) {
        expect(isValidDTCFormat(code), `${code} must be rejected`).toBe(false);
      }
    });

    it('should reject campaign ids and transmission designations', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      // P9324 = Nissan service campaign, U760E = Toyota transmission,
      // PC861 = Nissan campaign id, UA80E = transmission designation.
      for (const code of ['P9324', 'UA80E', 'U760E', 'PC861', 'P93AF']) {
        expect(isValidDTCFormat(code), `${code} must be rejected`).toBe(false);
      }
    });

    it('should reject a second character outside 0-3', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      for (const code of ['P4300', 'P9300', 'PA300', 'PF300']) {
        expect(isValidDTCFormat(code), `${code} must be rejected`).toBe(false);
      }
    });

    it('should reject non-P/B/C/U system letters', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      for (const code of ['A0300', 'X0300', '10300']) {
        expect(isValidDTCFormat(code), `${code} must be rejected`).toBe(false);
      }
    });

    it('should reject non-hex characters in the trailing three digits', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      for (const code of ['P030G', 'P03Z0', 'P0-00']) {
        expect(isValidDTCFormat(code), `${code} must be rejected`).toBe(false);
      }
    });

    it('should reject wrong-length input', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      for (const code of ['', 'P', 'P0', 'P030', 'P03000', 'P0300P0301']) {
        expect(isValidDTCFormat(code), `${code} must be rejected`).toBe(false);
      }
    });

    it('should normalise lower case (mirrors backend .strip().upper())', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      expect(isValidDTCFormat('p0300')).toBe(true);
      expect(isValidDTCFormat('p26b7')).toBe(true);
      expect(isValidDTCFormat('b00a0')).toBe(true);
      expect(isValidDTCFormat('P26b7')).toBe(true);
      // Normalisation must not smuggle junk through.
      expect(isValidDTCFormat('peace')).toBe(false);
    });

    it('should tolerate surrounding whitespace', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      expect(isValidDTCFormat(' P0300 ')).toBe(true);
      expect(isValidDTCFormat('\tp26b7\n')).toBe(true);
      // Whitespace is stripped, not ignored internally.
      expect(isValidDTCFormat('P 0300')).toBe(false);
    });

    it('should reject null/undefined without throwing', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      expect(isValidDTCFormat(undefined as unknown as string)).toBe(false);
      expect(isValidDTCFormat(null as unknown as string)).toBe(false);
    });

    it('should be stateless across repeated calls (no /g lastIndex bug)', async () => {
      const { isValidDTCFormat } = await import('../dtcService');

      // A shared RegExp carrying the `g` flag would alternate true/false here.
      for (let i = 0; i < 5; i++) {
        expect(isValidDTCFormat('P26B7')).toBe(true);
      }
    });
  });

  // ===========================================================================
  // getDTCCodeDetail — the user-facing bug: hex codes threw before the
  // request was ever sent, so the detail page could not be opened.
  // ===========================================================================

  describe('getDTCCodeDetail', () => {
    it('should open the detail page for the hex code P26B7', async () => {
      const mockResponse = { data: { code: 'P26B7', description_en: 'Hex code' } };
      mockApi.get = vi.fn().mockResolvedValue(mockResponse);

      const { getDTCCodeDetail } = await import('../dtcService');
      const result = await getDTCCodeDetail('P26B7');

      expect(mockApi.get).toHaveBeenCalledWith('/dtc/P26B7');
      expect(result).toEqual(mockResponse.data);
    });

    it('should request every real hex code instead of throwing locally', async () => {
      const { getDTCCodeDetail } = await import('../dtcService');

      for (const code of ['P090C', 'P0A94', 'B00A0', 'P17F1']) {
        mockApi.get = vi.fn().mockResolvedValue({ data: { code } });
        await expect(getDTCCodeDetail(code)).resolves.toEqual({ code });
        expect(mockApi.get).toHaveBeenCalledWith(`/dtc/${code}`);
      }
    });

    it('should uppercase and trim before requesting', async () => {
      mockApi.get = vi.fn().mockResolvedValue({ data: { code: 'P26B7' } });

      const { getDTCCodeDetail } = await import('../dtcService');
      await getDTCCodeDetail('  p26b7 ');

      expect(mockApi.get).toHaveBeenCalledWith('/dtc/P26B7');
    });

    it('should reject junk before sending a request', async () => {
      mockApi.get = vi.fn();

      const { getDTCCodeDetail } = await import('../dtcService');
      const { ApiError } = await import('../api');

      for (const code of ['PEACE', 'PACED', 'P9324', 'UA80E', 'U760E']) {
        await expect(getDTCCodeDetail(code)).rejects.toBeInstanceOf(ApiError);
      }
      expect(mockApi.get).not.toHaveBeenCalled();
    });

    it('should reject an empty code', async () => {
      const { getDTCCodeDetail } = await import('../dtcService');
      const { ApiError } = await import('../api');

      await expect(getDTCCodeDetail('')).rejects.toBeInstanceOf(ApiError);
      await expect(getDTCCodeDetail('   ')).rejects.toBeInstanceOf(ApiError);
    });
  });

  // ===========================================================================
  // searchDTCCodes — partial-input path (DTCAutocomplete search-as-you-type)
  // ===========================================================================

  describe('searchDTCCodes', () => {
    it('partial-input regression guard: 2-3 character prefixes still reach the API', async () => {
      // DTCAutocomplete fires useDTCSearch from `debouncedQuery.length >= 2`.
      // `isValidDTCFormat` is an ANCHORED whole-code check and correctly
      // rejects every one of these prefixes — which is exactly why the search
      // path must NOT be gated on it. If someone "tightens" searchDTCCodes with
      // the strict validator, search-as-you-type dies at the second keystroke
      // and this test fails.
      const { searchDTCCodes, isValidDTCFormat } = await import('../dtcService');

      for (const partial of ['P0', 'P26', 'B0', 'U01', 'p0']) {
        expect(isValidDTCFormat(partial), `${partial} is not a complete code`).toBe(false);

        mockApi.get = vi.fn().mockResolvedValue({ data: [{ code: 'P0300' }] });
        const result = await searchDTCCodes({ query: partial });

        expect(mockApi.get, `search must fire for "${partial}"`).toHaveBeenCalledOnce();
        const [endpoint, config] = mockApi.get.mock.calls[0];
        expect(endpoint).toBe('/dtc/search');
        expect(config.params.q).toBe(partial);
        expect(result).toEqual([{ code: 'P0300' }]);
      }
    });

    it('should pass free-text description queries through unvalidated', async () => {
      mockApi.get = vi.fn().mockResolvedValue({ data: [] });

      const { searchDTCCodes } = await import('../dtcService');
      await searchDTCCodes({ query: 'egeskimaradas' });

      const [, config] = mockApi.get.mock.calls[0];
      expect(config.params.q).toBe('egeskimaradas');
    });

    it('should short-circuit on an empty query without calling the API', async () => {
      mockApi.get = vi.fn();

      const { searchDTCCodes } = await import('../dtcService');

      expect(await searchDTCCodes({ query: '' })).toEqual([]);
      expect(await searchDTCCodes({ query: '   ' })).toEqual([]);
      expect(mockApi.get).not.toHaveBeenCalled();
    });

    it('should forward category, make and limit filters', async () => {
      mockApi.get = vi.fn().mockResolvedValue({ data: [] });

      const { searchDTCCodes } = await import('../dtcService');
      await searchDTCCodes({ query: 'P0', category: 'powertrain', make: 'VW', limit: 5 });

      const [, config] = mockApi.get.mock.calls[0];
      expect(config.params).toMatchObject({
        q: 'P0',
        category: 'powertrain',
        make: 'VW',
        limit: 5,
      });
    });
  });

  // ===========================================================================
  // Prefix-only helpers — a genuinely different concept (first character only,
  // mirrors backend `dtc_category`), so they are NOT tightened to [0-3].
  // ===========================================================================

  describe('getCategoryFromCode', () => {
    it('should map each system prefix to its category', async () => {
      const { getCategoryFromCode } = await import('../dtcService');

      expect(getCategoryFromCode('P26B7')).toBe('powertrain');
      expect(getCategoryFromCode('B00A0')).toBe('body');
      expect(getCategoryFromCode('C0567')).toBe('chassis');
      expect(getCategoryFromCode('U0100')).toBe('network');
    });

    it('should work on a lower-case and on a partial input', async () => {
      const { getCategoryFromCode } = await import('../dtcService');

      expect(getCategoryFromCode('p26b7')).toBe('powertrain');
      // Prefix lookup deliberately answers for in-progress input too.
      expect(getCategoryFromCode('P0')).toBe('powertrain');
    });

    it('should return undefined for an unknown prefix or empty input', async () => {
      const { getCategoryFromCode } = await import('../dtcService');

      expect(getCategoryFromCode('X0300')).toBeUndefined();
      expect(getCategoryFromCode('')).toBeUndefined();
    });
  });

  describe('formatDTCCode', () => {
    it('should append the Hungarian category name for a hex code', async () => {
      const { formatDTCCode } = await import('../dtcService');

      expect(formatDTCCode('P26B7')).toBe('P26B7 (Hajtaslanc)');
      expect(formatDTCCode('B00A0')).toBe('B00A0 (Karosszeria)');
    });

    it('should return the raw string for an unknown prefix', async () => {
      const { formatDTCCode } = await import('../dtcService');

      expect(formatDTCCode('X0300')).toBe('X0300');
    });
  });

  // ===========================================================================
  // Severity rendering — single source of truth for four render sites
  //
  // `DTCDetailPage`, `DTCAutocomplete` and `CommonIssuesPanel` all render DTC
  // severity. Each used to carry its own copy: three different palettes and two
  // different spellings of "Közepes". These tests pin the unified contract so a
  // future copy cannot silently diverge again.
  // ===========================================================================

  describe('severity rendering', () => {
    it('labels every severity in ACCENTED Hungarian', async () => {
      const { getSeverityLabelHu } = await import('../dtcService');

      expect(getSeverityLabelHu('low')).toBe('Alacsony');
      expect(getSeverityLabelHu('high')).toBe('Magas');
      expect(getSeverityLabelHu('critical')).toBe('Kritikus');

      // Regression guard: this used to be the ASCII-folded 'Kozepes', which is
      // how the same value rendered two different ways in one app. Every
      // recently written screen spells it with accents (ResultPage,
      // DemoResultPage, PasswordStrengthMeter).
      expect(getSeverityLabelHu('medium')).toBe('Közepes');
    });

    it('uses only WCAG AA compliant chip colours', async () => {
      const { getSeverityColorClass } = await import('../dtcService');

      // MEASURED contrast of the -800 text on its -100 background (WCAG 2.1,
      // sRGB relative luminance). These chips render at 10-14px, i.e. "normal
      // text", so the threshold is 4.5:1 — the 3:1 large-text allowance does
      // not apply:
      //   green-800  #166534 on #dcfce7 = 6.49:1
      //   yellow-800 #854d0e on #fef9c3 = 6.38:1
      //   orange-800 #9a3412 on #ffedd5 = 6.38:1
      //   red-800    #991b1b on #fee2e2 = 6.80:1
      // The previous -600 shades failed at every level (red 3.95:1, orange
      // 3.11:1, green 3.00:1, yellow 2.74:1 — below even the 3:1 floor).
      expect(getSeverityColorClass('low')).toBe('bg-green-100 text-green-800');
      expect(getSeverityColorClass('medium')).toBe('bg-yellow-100 text-yellow-800');
      expect(getSeverityColorClass('high')).toBe('bg-orange-100 text-orange-800');
      expect(getSeverityColorClass('critical')).toBe('bg-red-100 text-red-800');
    });

    it('never emits a -600 text shade for a severity chip', async () => {
      const { getSeverityColorClass } = await import('../dtcService');

      for (const severity of ['low', 'medium', 'high', 'critical', 'nonsense']) {
        expect(getSeverityColorClass(severity)).not.toMatch(
          /text-(green|yellow|orange|red)-600/
        );
      }
    });

    it('falls back to a quiet, AA compliant grey for an unknown severity', async () => {
      const { getSeverityColorClass, getSeverityLabelHu } = await import('../dtcService');

      // gray-600 #4b5563 on gray-100 #f3f4f6 = 6.87:1 — passes AA, and reads
      // quieter than a real severity rather than heavier.
      expect(getSeverityColorClass('made-up')).toBe('bg-gray-100 text-gray-600');
      expect(getSeverityLabelHu('made-up')).toBe('Ismeretlen');
    });

    it('getSeverityChip is PARTIAL so callers can render nothing', async () => {
      const { getSeverityChip } = await import('../dtcService');

      expect(getSeverityChip('medium')).toEqual({
        label: 'Közepes',
        className: 'bg-yellow-100 text-yellow-800',
      });

      // CommonIssuesPanel draws no chip at all for these rather than inventing
      // a placeholder severity — `VehicleCommonIssue.severity` is `string|null`.
      expect(getSeverityChip(null)).toBeUndefined();
      expect(getSeverityChip(undefined)).toBeUndefined();
      expect(getSeverityChip('')).toBeUndefined();
      expect(getSeverityChip('unrecognised')).toBeUndefined();
    });

    it('agrees with itself across the label and colour views', async () => {
      const { getSeverityChip, getSeverityColorClass, getSeverityLabelHu } = await import(
        '../dtcService'
      );

      // The three exported functions are views over ONE record. If a future
      // edit reintroduces a second table, this drifts and fails.
      for (const severity of ['low', 'medium', 'high', 'critical']) {
        const chip = getSeverityChip(severity);
        expect(chip).toBeDefined();
        expect(getSeverityLabelHu(severity)).toBe(chip?.label);
        expect(getSeverityColorClass(severity)).toBe(chip?.className);
      }
    });
  });
});

// =============================================================================
// Cross-service: diagnosisService validates DTC codes through the same helper
// =============================================================================

describe('diagnosisService DTC validation (shared isValidDTCFormat)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should accept hex codes in validateDiagnosisRequest', async () => {
    const { validateDiagnosisRequest } = await import('../diagnosisService');

    const errors = validateDiagnosisRequest({
      vehicleMake: 'Volkswagen',
      vehicleModel: 'Golf',
      vehicleYear: 2018,
      dtcCodes: ['P26B7', 'P090C', 'B00A0'],
      symptoms: 'Motor razkodik es egeskimaradas tapasztalhato',
    });

    expect(errors).toHaveLength(0);
  });

  it('should accept raw lower-case codes (validation runs before normalisation)', async () => {
    const { validateDiagnosisRequest } = await import('../diagnosisService');

    const errors = validateDiagnosisRequest({
      vehicleMake: 'Volkswagen',
      vehicleModel: 'Golf',
      vehicleYear: 2018,
      dtcCodes: ['p26b7', 'p0300'],
      symptoms: 'Motor razkodik es egeskimaradas tapasztalhato',
    });

    expect(errors).toHaveLength(0);
  });

  it('should still reject junk codes in validateDiagnosisRequest', async () => {
    const { validateDiagnosisRequest } = await import('../diagnosisService');

    const errors = validateDiagnosisRequest({
      vehicleMake: 'Volkswagen',
      vehicleModel: 'Golf',
      vehicleYear: 2018,
      dtcCodes: ['PEACE', 'P9324'],
      symptoms: 'Motor razkodik es egeskimaradas tapasztalhato',
    });

    expect(errors.some((e) => e.includes('PEACE'))).toBe(true);
    expect(errors.some((e) => e.includes('P9324'))).toBe(true);
  });

  it('should let hex codes through quickAnalyze and reject junk', async () => {
    const apiModule = await import('../api');
    const mockApi = apiModule.default as unknown as { post: ReturnType<typeof vi.fn> };
    mockApi.post = vi.fn().mockResolvedValue({ data: { results: [] } });

    const { quickAnalyze } = await import('../diagnosisService');
    const { ApiError } = await import('../api');

    await quickAnalyze(['p26b7', 'P0A94']);
    const [, , config] = mockApi.post.mock.calls[0];
    expect(config.params.dtc_codes).toEqual(['P26B7', 'P0A94']);

    await expect(quickAnalyze(['U760E'])).rejects.toBeInstanceOf(ApiError);
  });
});
