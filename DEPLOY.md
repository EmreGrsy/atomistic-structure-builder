# Deployment

The app is a single Streamlit process with no database; state lives in the
browser session.

## Free hosting (Streamlit Community Cloud)

The repo is set up for the free tier: `packages.txt` installs the packmol
binary from apt and `requirements.txt` covers everything else (moltemplate
is pip). Steps:

1. Push this repo to GitHub.
2. On https://share.streamlit.io sign in with GitHub and create a new app
   from the repo, main file `app.py`, Python 3.11.
3. In the app settings under Secrets paste the content of your
   `.streamlit/secrets.toml` (the OPENAI_API_KEY line). Secrets never live
   in the repo.

Free tier caveats: modest CPU and memory (large builds run slower and the
packmol timeout matters), the app sleeps after inactivity and wakes on the
next visit, and the machine's disk is ephemeral, so nothing durable can be
stored on it.

## Own server

- The conda environment from `environment.yml` (packmol and moltemplate.sh
  must be on PATH).
- `.streamlit/secrets.toml` with the OpenAI key (never committed).
- Headless launch:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  streamlit run app.py --server.port 8501 \
  --server.fileWatcherType none --server.headless true
```

- A reverse proxy (nginx or similar) with TLS in front of the port, with
  websocket forwarding enabled (Streamlit needs it).

## Connecting the repo to GitHub

```bash
gh auth login                 # or create the repo in the web UI
gh repo create atomistic-structure-builder --private --source . --push
```

Without the gh CLI: create an empty repo on github.com, then

```bash
git remote add origin git@github.com:YOURUSER/atomistic-structure-builder.git
git push -u origin main
```

`.gitignore` already excludes `.streamlit/secrets.toml` and all generated
data; nothing under `data/` is tracked.

## User continuity

Session state is in memory today: a refresh starts fresh. The zero cost
options, in increasing effort:

1. **Spec download and upload.** The build spec is a small JSON; a download
   button plus an upload box lets a user save their system and continue any
   time, on any deployment, with no storage at all.
2. **Server side session files keyed by a code.** Works on an own server;
   NOT durable on Community Cloud (ephemeral disk).
3. **A free tier database** (Supabase or Neon) keyed by a user token, for
   real accounts later.
