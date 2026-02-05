import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    react({
      // Use automatic JSX runtime for smaller bundle
      jsxRuntime: 'automatic',
    }),
    // Conditionally add bundle analyzer
    mode === 'analyze' && (async () => {
      const { visualizer } = await import('rollup-plugin-visualizer')
      return visualizer({
        open: true,
        gzipSize: true,
        brotliSize: true,
        filename: 'dist/stats.html',
      })
    })(),
  ].filter(Boolean),
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@pages': path.resolve(__dirname, './src/pages'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@services': path.resolve(__dirname, './src/services'),
    },
  },
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    // Disable sourcemaps in production for smaller bundles
    sourcemap: process.env.NODE_ENV !== 'production',
    // Target modern browsers for smaller bundles
    target: 'es2020',
    // Minification settings
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: process.env.NODE_ENV === 'production',
        drop_debugger: true,
      },
    },
    // Chunk size warning threshold (kB)
    chunkSizeWarningLimit: 500,
    rollupOptions: {
      output: {
        // Optimized manual chunks for better caching
        manualChunks: {
          // Core React libraries - rarely change
          'vendor-react': ['react', 'react-dom'],
          // Routing - separate chunk
          'vendor-router': ['react-router-dom'],
          // Data fetching - separate chunk
          'vendor-query': ['@tanstack/react-query'],
          // Form handling
          'vendor-forms': ['react-hook-form', 'zod'],
          // UI utilities
          'vendor-ui': ['clsx', 'tailwind-merge', 'lucide-react'],
          // HTTP client
          'vendor-http': ['axios'],
        },
        // Asset file naming for better caching
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name?.split('.') || []
          const ext = info[info.length - 1]
          if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext)) {
            return `assets/images/[name]-[hash][extname]`
          }
          if (/woff2?|eot|ttf|otf/i.test(ext)) {
            return `assets/fonts/[name]-[hash][extname]`
          }
          return `assets/[name]-[hash][extname]`
        },
        chunkFileNames: 'assets/js/[name]-[hash].js',
        entryFileNames: 'assets/js/[name]-[hash].js',
      },
    },
  },
  // Optimize dependencies
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@tanstack/react-query',
      'axios',
    ],
    // Exclude large dependencies that don't benefit from pre-bundling
    exclude: [],
  },
  // Enable CSS code splitting
  css: {
    devSourcemap: true,
  },
}))
