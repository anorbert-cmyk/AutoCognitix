import { type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface FloatingBottomBarProps {
  /** Content for the left side (e.g., recent analyses list) */
  leftContent?: ReactNode;
  /** Content for the right side (e.g., help link) */
  rightContent?: ReactNode;
  /** Control visibility */
  visible?: boolean;
  /** Additional className */
  className?: string;
}

/**
 * Floating bottom bar for displaying contextual information.
 * Fixed to the bottom of the viewport.
 *
 * @example
 * ```tsx
 * <FloatingBottomBar
 *   visible={isAnalyzing}
 *   leftContent={<RecentAnalysisList items={recentItems} />}
 *   rightContent={
 *     <a href="/help">Segítségre van szükséged?</a>
 *   }
 * />
 * ```
 */
export function FloatingBottomBar({
  leftContent,
  rightContent,
  visible = true,
  className,
}: FloatingBottomBarProps) {
  if (!visible) return null;

  return (
    <div
      className={cn(
        'fixed bottom-0 left-0 right-0',
        'bg-background border-t border-border',
        'shadow-floating-bar',
        'z-fixed',
        'animate-slide-up',
        className
      )}
      role="complementary"
      aria-label="Információs sáv"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-floating-bar py-3">
          {/* Left Content */}
          <div className="flex-1 min-w-0 pr-4">
            {leftContent}
          </div>

          {/* Right Content */}
          {rightContent && (
            <div className="flex-shrink-0">
              {rightContent}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default FloatingBottomBar;
