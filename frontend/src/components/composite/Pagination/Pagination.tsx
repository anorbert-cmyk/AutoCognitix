import { ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface PaginationProps {
  /** Current page (1-indexed) */
  currentPage: number;
  /** Total number of pages */
  totalPages: number;
  /** Callback when page changes */
  onPageChange: (page: number) => void;
  /** Number of page numbers to show on each side of current page */
  siblingCount?: number;
  /** Always show first and last page */
  showEdges?: boolean;
  /** Additional className */
  className?: string;
}

/**
 * Generate page numbers array with ellipsis
 */
function generatePageNumbers(
  currentPage: number,
  totalPages: number,
  siblingCount: number,
  showEdges: boolean
): (number | 'ellipsis')[] {
  const pages: (number | 'ellipsis')[] = [];

  if (totalPages <= 7) {
    // Show all pages if total is small
    for (let i = 1; i <= totalPages; i++) {
      pages.push(i);
    }
    return pages;
  }

  // Always show first page
  if (showEdges) {
    pages.push(1);
  }

  // Calculate range around current page
  const leftSibling = Math.max(currentPage - siblingCount, showEdges ? 2 : 1);
  const rightSibling = Math.min(
    currentPage + siblingCount,
    showEdges ? totalPages - 1 : totalPages
  );

  // Add left ellipsis if needed
  if (showEdges && leftSibling > 2) {
    pages.push('ellipsis');
  }

  // Add page numbers around current
  for (let i = leftSibling; i <= rightSibling; i++) {
    if (!pages.includes(i)) {
      pages.push(i);
    }
  }

  // Add right ellipsis if needed
  if (showEdges && rightSibling < totalPages - 1) {
    pages.push('ellipsis');
  }

  // Always show last page
  if (showEdges && !pages.includes(totalPages)) {
    pages.push(totalPages);
  }

  return pages;
}

/**
 * Pagination component for navigating through pages.
 *
 * @example
 * ```tsx
 * <Pagination
 *   currentPage={page}
 *   totalPages={12}
 *   onPageChange={setPage}
 * />
 * ```
 */
export function Pagination({
  currentPage,
  totalPages,
  onPageChange,
  siblingCount = 1,
  showEdges = true,
  className,
}: PaginationProps) {
  if (totalPages <= 1) return null;

  const pages = generatePageNumbers(
    currentPage,
    totalPages,
    siblingCount,
    showEdges
  );

  const canGoPrevious = currentPage > 1;
  const canGoNext = currentPage < totalPages;

  return (
    <nav
      className={cn('flex items-center justify-center gap-1', className)}
      aria-label="Lapozás"
    >
      {/* Previous Button */}
      <button
        type="button"
        onClick={() => canGoPrevious && onPageChange(currentPage - 1)}
        disabled={!canGoPrevious}
        className={cn(
          'flex items-center gap-1 px-3 py-2 text-sm font-medium rounded-lg',
          'transition-colors duration-fast',
          canGoPrevious
            ? 'text-muted-foreground hover:bg-muted hover:text-foreground'
            : 'text-muted-foreground/50 cursor-not-allowed'
        )}
        aria-label="Előző oldal"
      >
        <ChevronLeft className="h-4 w-4" />
        <span className="hidden sm:inline">Előző</span>
      </button>

      {/* Page Numbers */}
      <div className="flex items-center gap-1">
        {pages.map((page, index) =>
          page === 'ellipsis' ? (
            <span
              key={`ellipsis-${index}`}
              className="px-2 py-2 text-muted-foreground"
            >
              ...
            </span>
          ) : (
            <button
              key={page}
              type="button"
              onClick={() => onPageChange(page)}
              className={cn(
                'min-w-[40px] px-3 py-2 text-sm font-medium rounded-lg',
                'transition-colors duration-fast',
                page === currentPage
                  ? 'bg-primary-600 text-white'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              )}
              aria-label={`${page}. oldal`}
              aria-current={page === currentPage ? 'page' : undefined}
            >
              {page}
            </button>
          )
        )}
      </div>

      {/* Next Button */}
      <button
        type="button"
        onClick={() => canGoNext && onPageChange(currentPage + 1)}
        disabled={!canGoNext}
        className={cn(
          'flex items-center gap-1 px-3 py-2 text-sm font-medium rounded-lg',
          'transition-colors duration-fast',
          canGoNext
            ? 'text-muted-foreground hover:bg-muted hover:text-foreground'
            : 'text-muted-foreground/50 cursor-not-allowed'
        )}
        aria-label="Következő oldal"
      >
        <span className="hidden sm:inline">Tovább</span>
        <ChevronRight className="h-4 w-4" />
      </button>
    </nav>
  );
}

export default Pagination;
