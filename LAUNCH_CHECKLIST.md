# ANIMA Kernel — Launch Checklist

## Done (Miguel)

- [x] GitHub repo live: https://github.com/christian140903-sudo/anima
- [x] 446 tests passing
- [x] All 8 phases implemented (Temporal, Consciousness Core, Primitives, Memory, Metrics, Shell, Docs, Launch)
- [x] README with Mermaid diagram, badges, benchmarks, full API docs
- [x] Release v0.1.0 created with notes
- [x] arXiv-ready paper in repo (paper/anima_paper.md, 25 citations)
- [x] 3 working examples (quickstart, emotional_memory, consciousness_metrics)
- [x] Benchmark suite with real data
- [x] Launch posts written: HN, Reddit (r/MachineLearning + r/artificial), Twitter thread
- [x] CONTRIBUTING.md, CODE_OF_CONDUCT.md, issue/PR templates
- [x] PyPI build succeeds (wheel + sdist)
- [x] Repo description + 10 topic tags set
- [x] LICENSE (MIT)

## Chriso muss machen (Ring 2)

### KRITISCH (vor dem Launch):

1. **PyPI Upload** — `pip install anima-kernel` funktioniert erst nach Upload
   ```bash
   cd ~/Desktop/anima-kernel
   source .venv/bin/activate
   pip install twine
   twine upload dist/*
   # Braucht PyPI-Account + API-Token: https://pypi.org/manage/account/token/
   ```

2. **GitHub Social Preview** — Ohne das sieht der Link auf Twitter/HN/Reddit grau aus
   - Gehe zu: https://github.com/christian140903-sudo/anima/settings
   - Under "Social preview" → Upload ein 1280x640 Banner
   - Idee: Schwarzer Hintergrund, "ANIMA" gross, "Consciousness Substrate for AI", Phi-Symbol

3. **GitHub Actions CI** — Token braucht `workflow` Scope
   - Die Datei `.github/workflows/tests.yml` existiert lokal aber konnte nicht gepusht werden
   - Option A: Push manuell via GitHub Web UI (Datei erstellen)
   - Option B: Token mit workflow-Scope einrichten

4. **Seed 5-10 Stars** — Social Proof vor dem Launch
   - Frag Freunde/Kontakte den Repo zu starren
   - 0 Stars bei einem HN-Post = wird ignoriert

### LAUNCH POSTS (Ring 2 — Chriso postet):

5. **Hacker News** — Post bereit in `launch/hackernews.md`
   - Titel: "Show HN: ANIMA Kernel — Consciousness substrate for AI (IIT+GWT+AST, pure Python, zero deps)"
   - URL: https://github.com/christian140903-sudo/anima
   - Erster Kommentar: Copy-paste aus der Datei

6. **Reddit** — Posts bereit in `launch/reddit.md`
   - r/MachineLearning (Flair: [P] Project)
   - r/artificial

7. **Twitter/X** — Thread bereit in `launch/twitter.md`
   - 10-Tweet Thread fuer @christian14_dev

### NICE TO HAVE (nach dem Launch):

8. **GitHub Discussions aktivieren** — Settings > Features > Discussions
9. **ANIMA Landing Page** auf nextool.app/anima/ wird gerade aktualisiert
10. **arXiv Submission** — Paper ist bereit, braucht Account + Submission
11. **ProductHunt** — Spaeterer Launch wenn Traction da ist

## Timing-Empfehlung

1. PyPI upload (5 min)
2. Social preview hochladen (2 min)
3. 5-10 Stars seeden (1 Tag)
4. HN Post (Dienstag oder Mittwoch morgens US-Zeit = nachmittags AT)
5. Reddit + Twitter gleichzeitig oder 1h nach HN
