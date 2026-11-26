# Deploying NutriVision AI

Complete deployment guide: what runs where, how to put it online, and how to
verify, back up, and update it.

- **Recommended path:** one Linux VM running both containers with Docker Compose.
  Same origin for UI and API, one command to start, no CORS setup.
- **Split path:** frontend on a static host (Vercel/Netlify/Cloudflare Pages) and
  the API on a container host (Render/Railway/Fly/Koyeb). Useful when you want
  free hosting for the UI.
- **Demo path:** run Compose on your laptop and share it with a tunnel
  (Cloudflare Tunnel / ngrok). Not for a graded live demo you can't keep awake.

---

## 1. What gets deployed

| Part | Technology | Listens on | Ships as | Notes |
| --- | --- | --- | --- | --- |
| Frontend | React 19 build + nginx | 80 in container, `3000` host (Compose) | `frontend` Docker image (~103 MB) | nginx serves the SPA and proxies `/api` and `/uploads` to the backend service |
| Backend | FastAPI + Uvicorn | 8000 | `backend` Docker image (~2.9 GB) | CPU-only PyTorch; loads `backend/app/best_model.pth` at startup |
| Model | ResNet-50 multi-task checkpoint | – | inside the backend image | 100 MB `best_model.pth`; the API returns `503` if it is missing/invalid |
| Database | SQLite (default) or PostgreSQL | – | `/app/data/nutrivision.db` inside a Docker volume | Tables are created on startup; Alembic is also configured |
| Uploads | JPEG/PNG/WEBP files | – | `/app/data/uploads` inside a Docker volume | Served back at `/uploads/<file>` |

Traffic inside Compose:

```
Browser ──443/80──► nginx (frontend) ──/api, /uploads──► uvicorn (backend) ──► PyTorch
```

Only the frontend needs to be public. The browser never talks to port 8000
directly in the Compose setup, because nginx proxies those paths.

---

## 2. Pick your hosting

| Option | Best for | Effort | Cost |
| --- | --- | --- | --- |
| **A. One VM + Docker Compose** (recommended) | Everything in one place, custom domain, persistent data | Medium | ~$4–7/mo (Oracle Always Free works but is ARM64) |
| **B. Split: Vercel/Netlify + Render/Railway/Fly** | Free UI hosting, managed API | Medium | $0–7/mo; free API tiers are tight for PyTorch |
| **C. Laptop + Cloudflare Tunnel** | Quick demo to a mentor | Low | Free, but only while your laptop runs |

Minimum server size for the backend: **2 vCPU, 2 GB RAM, ~10 GB disk**.
PyTorch CPU inference wants 0.8–1.2 GB RAM; 512 MB free tiers will OOM on the
first model load. Building the image needs even more — see §4.6 if your VM is
small.

---

## 3. Before you start

1. **Put the code where the server can get it.** Push the repo to GitHub and
   clone it on the server, or copy it with `scp`.
   The repository carries the model files, so a plain clone is ~0.5 GB:
   `backend/app/best_model.pth` (100 MB), `backend/app/best_model_int8.pth`
   (96 MB), and the four artifacts under `ml/optimization/`. GitHub accepts all
   of them (each is below the 100 MB hard limit) but warns above 50 MB.
   See §7 if you want to slim this down.
2. **Generate a production secret:**
   ```bash
   openssl rand -hex 32
   ```
3. **Decide your database.** SQLite on a Docker volume is fine for a project
   demo. For anything longer-lived, provision PostgreSQL and set `DATABASE_URL`.
4. **Optional: a domain.** A subdomain like `nutri.example.com` pointed at your
   server's IP. HTTPS is required for camera-based uploads in most browsers, so
   plan for TLS from the start (§5).

---

## 4. Option A — one Linux VM with Docker Compose (recommended)

Tested with Ubuntu 22.04/24.04 and Docker Engine 28, Compose v2.

### 4.1 Prepare the server

```bash
# on the VM
sudo apt update && sudo apt upgrade -y
curl -fsSL https://get.docker.com | sudo sh        # installs Engine + Compose plugin
sudo usermod -aG docker "$USER"                    # log out/in afterwards
docker compose version                             # must print v2.x
```

Open only what you need in the cloud firewall/security group: **22 (SSH)**,
**80**, **443**. Do **not** open 8000. If you follow §4.4, 3000 stays
localhost-only as well.

### 4.2 Get the code and configure

```bash
git clone <your-repo-url> nutrivision
cd nutrivision
cp .env.example .env
nano .env
```

Fill it in (this exact file is what `docker compose` reads):

```ini
SECRET_KEY=<paste the openssl rand -hex 32 output>
APP_ENV=production
ACCESS_TOKEN_EXPIRE_MINUTES=60
CORS_ORIGINS=https://nutri.example.com
SEED_DEMO=false
MODEL_VARIANT=fp32
VITE_API_URL=
```

- `VITE_API_URL` stays **empty**: the SPA calls its own origin and nginx proxies.
- `SEED_DEMO=false` disables the `demo@nutrivision.ai` account. Create your own
  user through the signup page after deploying. If you *want* the demo account
  for a viva, keep `SEED_DEMO=true`.
- `CORS_ORIGINS` only matters for cross-origin calls (Option B). Keeping your
  public URL here is harmless.
- The compose file fails fast if `SECRET_KEY` is empty in production, which is
  intentional.

### 4.3 Build and start

```bash
docker compose up -d --build
docker compose ps          # backend should become "healthy"
docker compose logs -f backend
```

First start takes 10–30 s extra: PyTorch imports and the 100 MB checkpoint load
before `/api/health` turns healthy.

### 4.4 Verify from the server

```bash
curl -s http://127.0.0.1:3000/api/health | python3 -m json.tool
# expect "modelReady": true and "modelValidation": {"valid": true, ...}

curl -s -o /dev/null -w "spa:%{http_code}\n" http://127.0.0.1:3000/
curl -s -o /dev/null -w "api:%{http_code}\n" http://127.0.0.1:3000/api/health
```

**Close the backend port to the internet.** Edit `docker-compose.yml` and either
delete the backend `ports:` block or bind it to localhost:

```yaml
    ports:
      - "127.0.0.1:8000:8000"
```

Then `docker compose up -d` again. nginx reaches the backend over the Compose
network, so nothing public breaks.

### 4.5 HTTPS with Caddy (simplest option)

Run Caddy on the host, in front of the frontend container:

```bash
sudo apt install -y caddy
sudo nano /etc/caddy/Caddyfile
```

```
nutri.example.com {
    reverse_proxy 127.0.0.1:3000
}
```

```bash
sudo systemctl reload caddy
```

Caddy obtains and renews the Let's Encrypt certificate automatically. DNS for
`nutri.example.com` must point at the VM's public IP first. From then on use
`https://nutri.example.com` (and put it in `CORS_ORIGINS`, which becomes
unimportant for same-origin traffic but correct anyway).

Host nginx + Certbot or a Cloudflare proxy in front of port 80 works equally
well if you prefer them.

### 4.6 Building on a small VM (optional)

If the VM has under 2 GB RAM, the `pip install torch` layer can be killed by the
OOM killer. Two fixes:

**a) Add swap on the VM**

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

**b) Build on your laptop and ship the images** (fastest, avoids compiling on
the server):

```bash
# laptop, in the project root
docker compose build
docker login -u <dockerhub-user>
docker tag foodimagecalorieestimator-backend  <user>/nutrivision-backend:v1
docker tag foodimagecalorieestimator-frontend <user>/nutrivision-frontend:v1
docker push <user>/nutrivision-backend:v1
docker push <user>/nutrivision-frontend:v1
```

Then on the server use a compose override that pulls instead of builds:

```yaml
# docker-compose.images.yml
services:
  backend:
    image: <user>/nutrivision-backend:v1
    build: null
  frontend:
    image: <user>/nutrivision-frontend:v1
    build: null
```

```bash
docker compose -f docker-compose.yml -f docker-compose.images.yml up -d
```

### 4.7 Operations

```bash
docker compose logs -f --tail=100      # follow logs
docker compose restart backend         # restart just the API
docker compose down                    # stop containers (volumes survive)
docker compose up -d --build           # rebuild + restart after a git pull
```

Volumes `nutrivision_data` (SQLite DB) and `nutrivision_uploads` (images)
survive `down`, rebuilds, and reboots because both containers use
`restart: unless-stopped`.

**Backup** (run on the server, writes a tarball to your home dir):

```bash
cd ~
docker run --rm -v nutrivision_data:/data -v "$PWD":/backup alpine \
  tar czf /backup/nutrivision-data-$(date +%F).tar.gz -C /data .
docker run --rm -v nutrivision_uploads:/data -v "$PWD":/backup alpine \
  tar czf /backup/nutrivision-uploads-$(date +%F).tar.gz -C /data .
```

Restore is the same command with `tar xzf` into the volume.

**Update** (zero drama):

```bash
git pull
docker compose up -d --build
curl -fsS http://127.0.0.1:3000/api/health >/dev/null && echo OK
```

---

## 5. Option B — split hosting (static frontend + container backend)

Use this when you want the UI on free static hosting. The browser will call the
API across origins, so `CORS_ORIGINS` and `VITE_API_URL` must both be set.

### 5.1 Backend on Render / Railway / Fly

Build context is `backend/`, Dockerfile is `backend/Dockerfile`. The image
already honours a platform-provided `PORT`.

**Render**

1. New → Web Service → connect the repo.
2. Root Directory: `backend`. Runtime: Docker.
3. Health Check Path: `/api/health`. Start command: leave empty (image CMD).
4. Plan: at least **2 GB RAM** (512 MB will OOM on model load).
5. Add a **Disk**: name `data`, mount path `/app/data`, 1 GB. Free instances
   cannot attach disks, so SQLite/uploads would reset on every deploy.
6. Environment variables:
   ```ini
   APP_ENV=production
   SECRET_KEY=<openssl rand -hex 32>
   ACCESS_TOKEN_EXPIRE_MINUTES=60
   CORS_ORIGINS=https://<your-frontend-domain>
   SEED_DEMO=false
   DATA_DIR=/app/data
   UPLOADS_DIR=/app/data/uploads
   ```

**Railway:** New Project → Deploy from repo → Settings → Dockerfile path
`backend/Dockerfile`; add a Volume mounted at `/app/data`; same env vars; Railway
provides `PORT` automatically.

**Fly.io:**

```bash
cd backend
fly launch --dockerfile Dockerfile --no-deploy
fly volumes create data --size 1
# in fly.toml: [env] APP_ENV="production", DATA_DIR="/app/data", UPLOADS_DIR="/app/data/uploads"
#               [http_service] internal_port = 8000
fly secrets set SECRET_KEY=<secret> CORS_ORIGINS=https://<frontend-domain>
fly deploy
```

**Cloud Run / App Runner:** deploy the same image, memory 2 GiB, set the env
vars above, and mount storage only if you need persistence (otherwise history
resets when the instance recycles).

Note: the backend image is ~2.9 GB. First pull on small network links is slow;
subsequent deploys reuse layers.

### 5.2 Frontend on Vercel

1. Import the repo, **Root Directory: `frontend`**.
2. Framework preset: Vite. Build command `npm ci && npm run build`, output `dist`.
3. Environment variable: `VITE_API_URL=https://<backend-domain>` (no trailing
   slash). It is baked in at build time — changing it requires a redeploy.
4. SPA routing: add `frontend/vercel.json` before deploying:

   ```json
   { "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }] }
   ```

5. After the first deploy, copy the Vercel URL into the backend's
   `CORS_ORIGINS` and redeploy the backend.

**Netlify instead:** same root/build settings, add `frontend/public/_redirects`
containing `/*  /index.html  200`.

**Cloudflare Pages instead:** build command `npm run build`, output `dist`, add
the same `_redirects` file.

### 5.3 What changes vs. Option A

- Uploaded images resolve through `resolveAssetUrl()`, which prefixes
  `VITE_API_URL`, so history thumbnails keep working cross-origin.
- JWT is stored in `localStorage`; cross-origin use is fine as long as CORS is
  configured correctly (scheme + host, no trailing slash).
- You lose the single-origin setup: two deployments, two sets of logs, two URLs.

---

## 6. Environment variable reference

| Variable | Where | Production value | Purpose |
| --- | --- | --- | --- |
| `SECRET_KEY` | backend | random 64-hex | JWT signing. **Required** when `APP_ENV=production`; the app refuses to start without it |
| `APP_ENV` | backend | `production` | Enables the secret-key requirement |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | backend | `60` | Login session length |
| `CORS_ORIGINS` | backend | your frontend URL(s), comma-separated | Cross-origin allow-list; same-origin Compose needs only the public URL |
| `SEED_DEMO` | backend | `false` | Creates `demo@nutrivision.ai` / `demoPassword123` when true |
| `DATABASE_URL` | backend | unset (SQLite) or `postgresql://...` | Database. `postgres://` is auto-mapped to `postgresql://` |
| `DATA_DIR` | backend | `/app/data` | SQLite file location |
| `UPLOADS_DIR` | backend | `/app/data/uploads` | Upload storage |
| `MODEL_VARIANT` | backend | `fp32` or `int8` | `int8` loads `best_model_int8.pth` on CPU |
| `MODEL_WEIGHTS_PATH` | backend | unset | Point at a checkpoint outside the image |
| `PORT` | backend | platform-assigned | Honoured by the image (`${PORT:-8000}`) |
| `VITE_API_URL` | frontend build | empty (Compose) / backend URL (split) | API origin baked into the SPA |
| `VITE_DEV_BACKEND_URL` | frontend dev | – | Vite dev-proxy target; not used in production |

---

## 7. Handling the large model files

The checkpoint is committed, so Docker builds work from a plain clone. If the
repository size bothers you, three options:

1. **Git LFS** (keeps history, adds fetch cost on clone):
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git add backend/app/best_model.pth backend/app/best_model_int8.pth
   git commit -m "Track model checkpoints with Git LFS"
   ```
   Render/Railway/Fly and most CI systems fetch LFS automatically.
2. **Artifact download at deploy time.** Upload the checkpoint to S3/R2/release
   assets, then add a download step before `docker build`, or set
   `MODEL_WEIGHTS_PATH` to a path your startup script fetches.
3. **Drop unused variants.** Only `backend/app/best_model.pth` (and
   `best_model_int8.pth` if you use `MODEL_VARIANT=int8`) is needed at runtime.
   The four files in `ml/optimization/` are benchmark artifacts;
   `ml/optimization/benchmark_results.md` documents what they were used for.

Whatever you do, do not deploy without a valid checkpoint: the API deliberately
returns `503` and reports the reason in `/api/health`.

---

## 8. Security checklist

- [ ] `SECRET_KEY` is random and only in `.env` / platform secrets, never in git.
- [ ] `APP_ENV=production` so a missing key fails fast.
- [ ] `SEED_DEMO=false` (unless a demo account is an explicit requirement).
- [ ] HTTPS everywhere; camera/upload APIs require a secure context.
- [ ] Only 80/443 (and 22) are publicly reachable; 8000 is loopback-only.
- [ ] `CORS_ORIGINS` lists exactly your frontend origin(s).
- [ ] `.env` is gitignored (it is) and excluded from images (it is, via
      `.dockerignore`).
- [ ] Backups of `nutrivision_data` and `nutrivision_uploads` run on a schedule.
- [ ] Keep Docker and base images updated (`docker compose build --pull`).

---

## 9. Post-deploy verification checklist

Run these from your machine against the public URL:

```bash
# 1. Model is loaded and valid
curl -s https://nutri.example.com/api/health | python -m json.tool
#    "modelReady": true, "modelValidation": {"valid": true}

# 2. Signup works (replace the email)
curl -s -X POST https://nutri.example.com/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"me@example.com","password":"Str0ngPassword!","username":"me"}'
#    -> {"success": true, "token": "..."}

# 3. Prediction works (use any jpg)
TOKEN="<token from step 2>"
curl -s -X POST https://nutri.example.com/api/predict \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@sushi.jpg"
#    -> foodClass, confidenceScore, estimatedCalories, protein, carbohydrates, fat

# 4. Saved history round-trips
curl -s https://nutri.example.com/api/history -H "Authorization: Bearer $TOKEN"
```

In the browser: signup → upload → result page → save to history → dashboard
shows the entry → profile goals save → mobile width (≤ 390 px) has no horizontal
scroll. On split hosting, also watch the DevTools console for CORS errors.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `/api/health` shows `modelReady: false` with error codes | Checkpoint missing, wrong class count, missing `target_stats`/`val_metrics`, or default BatchNorm stats | Re-copy `backend/app/best_model.pth`; the error codes name the failed check |
| `/api/predict` returns `503` | Same as above (by design) | Fix the checkpoint; the UI shows "Model unavailable" |
| 502 from the frontend | Backend container unhealthy or still starting | `docker compose ps`, `docker compose logs backend`; wait for `healthy` |
| CORS error in the console (split hosting) | `CORS_ORIGINS` missing the frontend origin | Set exact scheme + host, no trailing slash; redeploy |
| Images/thumbnails 404 after redeploy | Uploads volume not mounted | Mount a persistent disk at `/app/data` |
| History empty after redeploy | SQLite on ephemeral storage | Attach a disk/volume, or move to PostgreSQL |
| Backend restarts with OOM | Host under 2 GB RAM | Upgrade plan; one Uvicorn worker is correct for CPU inference |
| Build dies while installing torch | OOM during pip on a 1 GB VM | Build on your laptop and push the image (§4.6) |
| `docker compose` complains about `SECRET_KEY` | `.env` missing/empty | `cp .env.example .env`, set the key |
| First request after idle is slow | Container restart / cold start | Expected on free tiers; keep the health check enabled |
| Camera upload blocked in browser | No HTTPS | Put Caddy/Cloudflare in front (§4.5) |

---

## 11. Costs at a glance

| Component | Typical option | Cost |
| --- | --- | --- |
| VM (Option A) | Hetzner CX22 / DigitalOcean 2 GB / Lightsail 2 GB | ~$4–7/mo |
| VM free tier | Oracle Cloud Always Free (ARM Ampere) | $0; build is slower, boot volume is fine |
| Backend (Option B) | Render Standard 2 GB + 1 GB disk | ~$7–9/mo |
| Backend free tier | Railway trial credit / Fly shared-cpu-1x | $0–5/mo, tight memory |
| Frontend | Vercel / Netlify / Cloudflare Pages | $0 |
| Domain (optional) | any registrar | ~$10/yr |
| HTTPS | Caddy + Let's Encrypt | $0 |

---

## 12. Command cheat sheet

```bash
# local sanity build before deploying anywhere
docker compose build && docker compose up -d && curl -s localhost:3000/api/health

# server lifecycle
docker compose up -d --build
docker compose ps
docker compose logs -f backend
docker compose restart backend
docker compose down

# health / smoke
curl -s localhost:3000/api/health | python3 -m json.tool
curl -s -o /dev/null -w "%{http_code}\n" localhost:3000/

# secrets
openssl rand -hex 32
```
