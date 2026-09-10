# MEMORY.md — Durable Workspace Knowledge & Lessons Learned

Curated workspace patterns, gotchas, architecture choices, and lessons learned across tasks.

## Workspace Conventions
- **Docker Networking:** `personal-network` with **no host ports** (service-name addressing only: `personal-postgres:5432`, `personal-redis:6379`).
- **Dev Overrides:** Overrides live in `docker/<repo>/docker-compose.override.yml`. Repos are never modified for local dev.
- **Port Allocation:** Deployed services own 30xx/80xx; dev override services live strictly in 48xx.
- **Remote Access:** Tailscale serve on `nova` only; never funnel.

## Project Gotchas & Quick References
- `expense-tracker`: Laravel 12 + Inertia / Vue 3 + Postgres + Redis. Run tests via `php artisan test`.
- `horizonx`: Go API + containerized agent.
- `peek`: Python 3 / FastAPI + vanilla JS previewer.
