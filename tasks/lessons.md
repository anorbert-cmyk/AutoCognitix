# Lessons Learned - AutoCognitix

## Railway Deployment

### 2024-02-04 - Frontend Build Failures

**Problem:** Frontend build kept failing on Railway with various errors.

**Root Causes:**
1. `.gitignore` had `*.json` rule that excluded `tsconfig.json` and `package.json`
2. `serve` package was in devDependencies instead of dependencies
3. `vite-env.d.ts` was missing for `import.meta.env` types
4. TailwindCSS shadcn/ui custom classes (`border-border`, `bg-background`) not defined

**Solutions:**
- Add explicit exceptions in `.gitignore`: `!frontend/tsconfig.json`, `!frontend/package.json`
- Move `serve` to dependencies for production runtime
- Create `src/vite-env.d.ts` with Vite client types
- Add CSS variable-based colors to `tailwind.config.js`

**Prevention:**
- Always verify all config files are tracked in git before pushing
- Test `npm run build` locally before deployment
- Check for shadcn/ui specific TailwindCSS requirements

---

## Git Patterns

### Always check tracked files before deployment
```bash
git ls-files frontend/ | grep -E "\.(json|ts)$"
```

### Verify .gitignore exceptions work
```bash
git status --ignored
```
