# ============================================
# AutoCognitix Frontend - Development Dockerfile
# Multi-stage build with development and production targets
# ============================================

# ============================================
# Stage 1: Dependencies - Install node modules
# ============================================
FROM node:20-alpine AS deps

WORKDIR /app

# Install dependencies first for better layer caching
COPY package.json package-lock.json* ./

# Clean install for reproducible builds
RUN npm ci

# ============================================
# Stage 2: Development - Vite dev server
# ============================================
FROM node:20-alpine AS development

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy package files
COPY package.json package-lock.json* ./

# Environment variables
ENV NODE_ENV=development

# Expose Vite dev server port
EXPOSE 3000

# Development command with hot reload
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]

# ============================================
# Stage 3: Builder - Build production assets
# ============================================
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build arguments for environment variables at build time
ARG VITE_API_URL
ARG VITE_APP_VERSION

ENV VITE_API_URL=${VITE_API_URL:-/api}
ENV VITE_APP_VERSION=${VITE_APP_VERSION:-1.0.0}
ENV NODE_ENV=production

# Build the application
RUN npm run build

# ============================================
# Stage 4: Production - Nginx serving
# ============================================
FROM nginx:1.25-alpine AS production

# Set labels for container metadata
LABEL maintainer="AutoCognitix Team"
LABEL version="1.0"
LABEL description="AutoCognitix Frontend - Production"

# Install curl for health checks
RUN apk add --no-cache curl

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

# Start nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
