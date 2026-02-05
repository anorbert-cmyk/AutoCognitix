/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554',
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
          50: '#fdf4ff',
          100: '#fae8ff',
          200: '#f5d0fe',
          300: '#f0abfc',
          400: '#e879f9',
          500: '#d946ef',
          600: '#c026d3',
          700: '#a21caf',
          800: '#86198f',
          900: '#701a75',
          950: '#4a044e',
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // Status colors for badges and indicators
        status: {
          success: {
            DEFAULT: 'rgb(var(--status-success) / <alpha-value>)',
            light: 'rgb(var(--status-success-light) / <alpha-value>)',
            dark: 'rgb(var(--status-success-dark) / <alpha-value>)',
          },
          warning: {
            DEFAULT: 'rgb(var(--status-warning) / <alpha-value>)',
            light: 'rgb(var(--status-warning-light) / <alpha-value>)',
            dark: 'rgb(var(--status-warning-dark) / <alpha-value>)',
          },
          pending: {
            DEFAULT: 'rgb(var(--status-pending) / <alpha-value>)',
            light: 'rgb(var(--status-pending-light) / <alpha-value>)',
            dark: 'rgb(var(--status-pending-dark) / <alpha-value>)',
          },
          error: {
            DEFAULT: 'rgb(var(--status-error) / <alpha-value>)',
            light: 'rgb(var(--status-error-light) / <alpha-value>)',
            dark: 'rgb(var(--status-error-dark) / <alpha-value>)',
          },
          info: {
            DEFAULT: 'rgb(var(--status-info) / <alpha-value>)',
            light: 'rgb(var(--status-info-light) / <alpha-value>)',
            dark: 'rgb(var(--status-info-dark) / <alpha-value>)',
          },
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      spacing: {
        '18': '4.5rem',
        '22': '5.5rem',
        'header': 'var(--header-height)',
        'header-mobile': 'var(--header-height-mobile)',
        'floating-bar': 'var(--floating-bar-height)',
      },
      minHeight: {
        'screen-minus-header': 'calc(100vh - var(--header-height))',
      },
      boxShadow: {
        'floating-bar': 'var(--floating-bar-shadow)',
      },
      zIndex: {
        'dropdown': 'var(--z-dropdown)',
        'sticky': 'var(--z-sticky)',
        'fixed': 'var(--z-fixed)',
        'modal-backdrop': 'var(--z-modal-backdrop)',
        'modal': 'var(--z-modal)',
        'popover': 'var(--z-popover)',
        'tooltip': 'var(--z-tooltip)',
        'toast': 'var(--z-toast)',
      },
      transitionDuration: {
        'fast': 'var(--duration-fast)',
        'normal': 'var(--duration-normal)',
        'slow': 'var(--duration-slow)',
      },
    },
  },
  plugins: [],
}
