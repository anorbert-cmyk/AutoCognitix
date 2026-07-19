import { useEffect, useId, useRef, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface NavChild {
  label: string;
  href: string;
  icon?: React.ReactNode;
  /** Hidden from the menu unless the user is authenticated. */
  requiresAuth?: boolean;
  /** Primary action of the group — visually emphasised. */
  lead?: boolean;
}

interface NavDropdownProps {
  label: string;
  items: NavChild[];
}

const isPathActive = (pathname: string, href: string) =>
  pathname === href || pathname.startsWith(href + '/');

/**
 * Desktop grouped-navigation dropdown following the W3C "disclosure navigation"
 * pattern: a button toggles a panel of plain links (no `role=menu` keyboard
 * contract). Opens on hover (pointer devices only) and on click/Enter; closes
 * on mouse-leave, outside click, Escape, and when focus leaves the group.
 */
export default function NavDropdown({ label, items }: NavDropdownProps) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelId = useId();
  const location = useLocation();

  // Only wire hover-to-open where the device actually hovers; on touch a tap
  // would otherwise fire mouse-enter (open) then click (toggle) → appear inert.
  const [hoverCapable] = useState(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return true;
    return window.matchMedia('(hover: hover)').matches;
  });

  const groupActive = items.some((it) => isPathActive(location.pathname, it.href));

  // Close when clicking anywhere outside the group (covers click/touch users).
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [open]);

  return (
    <div
      ref={wrapRef}
      className="relative"
      onMouseEnter={() => hoverCapable && setOpen(true)}
      onMouseLeave={() => hoverCapable && setOpen(false)}
      onBlur={(e) => {
        if (!wrapRef.current?.contains(e.relatedTarget as Node)) setOpen(false);
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape' && open) {
          e.stopPropagation();
          setOpen(false);
          buttonRef.current?.focus();
        }
      }}
    >
      <button
        ref={buttonRef}
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium rounded-lg',
          'transition-colors duration-150',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500',
          groupActive || open
            ? 'bg-primary-50 text-primary-700'
            : 'text-muted-foreground hover:bg-muted hover:text-foreground'
        )}
      >
        {label}
        <ChevronDown
          className={cn('h-4 w-4 transition-transform duration-150', open && 'rotate-180')}
          aria-hidden="true"
        />
      </button>

      {open && (
        // `top-full` + `pt-2` keeps the panel touching the trigger so the pointer
        // never crosses an empty gap (which would fire mouse-leave and close it).
        <div id={panelId} className="absolute left-0 top-full z-50 w-64 pt-2">
          <div className="rounded-xl border border-border bg-card py-1.5 shadow-lg">
            {items.map((item) => {
              const itemActive = isPathActive(location.pathname, item.href);
              return (
                <NavLink
                  key={item.href}
                  to={item.href}
                  onClick={() => setOpen(false)}
                  className={cn(
                    'flex items-center gap-3 px-4 py-2.5 text-sm transition-colors',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary-500',
                    itemActive
                      ? 'bg-primary-50 text-primary-700'
                      : item.lead
                        ? 'text-primary-700 hover:bg-primary-50'
                        : 'text-foreground hover:bg-muted'
                  )}
                >
                  {item.icon && (
                    <span aria-hidden="true" className={cn('shrink-0', item.lead && 'text-primary-600')}>
                      {item.icon}
                    </span>
                  )}
                  <span className={cn('flex-1 truncate', item.lead && 'font-semibold')}>{item.label}</span>
                  {item.lead && <ChevronRight className="h-4 w-4 shrink-0 text-primary-500" aria-hidden="true" />}
                </NavLink>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
