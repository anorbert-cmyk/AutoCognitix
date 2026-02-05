import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Utility function to merge Tailwind CSS classes with clsx.
 * Handles conditional classes and class conflicts properly.
 *
 * @example
 * ```tsx
 * cn('px-4 py-2', isPrimary && 'bg-blue-500', 'px-2')
 * // Result: 'py-2 bg-blue-500 px-2' (px-4 is overwritten by px-2)
 * ```
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
