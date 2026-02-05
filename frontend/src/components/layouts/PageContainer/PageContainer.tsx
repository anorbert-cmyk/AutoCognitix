import { type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface PageContainerProps {
  /** Page content */
  children: ReactNode;
  /** Maximum width constraint */
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full';
  /** Padding size */
  padding?: 'none' | 'sm' | 'md' | 'lg';
  /** Add padding at bottom for floating bar */
  hasFloatingBar?: boolean;
  /** Additional className */
  className?: string;
}

const maxWidthStyles = {
  sm: 'max-w-2xl',
  md: 'max-w-4xl',
  lg: 'max-w-5xl',
  xl: 'max-w-6xl',
  '2xl': 'max-w-7xl',
  full: 'max-w-full',
};

const paddingStyles = {
  none: '',
  sm: 'px-4 py-4',
  md: 'px-4 sm:px-6 lg:px-8 py-6',
  lg: 'px-4 sm:px-6 lg:px-8 py-8',
};

/**
 * Page container component for consistent layout.
 *
 * @example
 * ```tsx
 * <PageContainer maxWidth="xl" padding="md">
 *   <h1>Page Title</h1>
 *   <p>Page content</p>
 * </PageContainer>
 *
 * // With floating bar
 * <PageContainer hasFloatingBar>
 *   <Content />
 * </PageContainer>
 * ```
 */
export function PageContainer({
  children,
  maxWidth = '2xl',
  padding = 'md',
  hasFloatingBar = false,
  className,
}: PageContainerProps) {
  return (
    <main
      className={cn(
        'flex-1 min-h-screen-minus-header',
        maxWidthStyles[maxWidth],
        'mx-auto w-full',
        paddingStyles[padding],
        hasFloatingBar && 'pb-24', // Extra padding for floating bar
        className
      )}
    >
      {children}
    </main>
  );
}

export default PageContainer;
