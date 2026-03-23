/**
 * Web Speech API helper — typed wrapper to eliminate `as any` casts.
 *
 * The SpeechRecognition constructor is vendor-prefixed in some browsers
 * and not fully typed in lib.dom, so we declare our own types in
 * `src/types/web-speech-api.d.ts` and use a safe runtime lookup here.
 */

/**
 * Returns the SpeechRecognition constructor if available, otherwise `null`.
 */
export function getSpeechRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === 'undefined') return null;

  if ('SpeechRecognition' in window) {
    return (window as unknown as Record<string, unknown>)
      .SpeechRecognition as SpeechRecognitionCtor;
  }

  if ('webkitSpeechRecognition' in window) {
    return (window as unknown as Record<string, unknown>)
      .webkitSpeechRecognition as SpeechRecognitionCtor;
  }

  return null;
}

/**
 * Whether the browser supports the Web Speech API.
 */
export function hasSpeechRecognition(): boolean {
  return getSpeechRecognitionCtor() !== null;
}
