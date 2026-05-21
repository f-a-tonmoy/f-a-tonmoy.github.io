# Fahim Ahamed — Portfolio

Personal portfolio site for **Fahim Ahamed** — data scientist with a background in applied AI research.

🌐 **Live**: [f-a-tonmoy.github.io](https://f-a-tonmoy.github.io)

---

## About

Data scientist with a background in applied AI research — turning data into insights, predictive models, and decisions across healthcare, NLP, computer vision, and beyond.

- **7+ peer-reviewed AI research papers** (SAGE, Springer, IEEE)
- **80+ articles** on data science, ML, and responsible AI
- Specialties: explainable AI, medical imaging, NLP, adversarial robustness
- Based in NYC · U.S. Permanent Resident · No sponsorship required

📄 **[Resume](assets/Resume%20-%20Fahim%20Ahamed.pdf)** &nbsp;·&nbsp; 💼 **[LinkedIn](https://linkedin.com/in/f-a-tonmoy)** &nbsp;·&nbsp; 🔬 **[ORCID](https://orcid.org/0009-0006-2638-6521)** &nbsp;·&nbsp; ✉️ **[Email](mailto:f.a.tonmoy00@gmail.com)**

---

## Pages

| | |
|---|---|
| 🏠 [Home](https://f-a-tonmoy.github.io) | Overview, recent work, articles |
| 💼 [Experience](https://f-a-tonmoy.github.io/experience.html) | Timeline of roles + education |
| 🔬 [Research](https://f-a-tonmoy.github.io/research.html) | Published papers + manuscripts in pipeline |
| 🛠️ [Projects](https://f-a-tonmoy.github.io/projects.html) | Applied ML, deep learning, software engineering |
| ✍️ [Articles](https://f-a-tonmoy.github.io/writing.html) | 80+ technical articles, filterable + searchable |

---

## Stack

Pure static site — **HTML · CSS · vanilla JavaScript**. No build step. Deployed via GitHub Pages.

```
.
├── index.html / experience.html / research.html / projects.html / writing.html
├── 404.html               # branded not-found page
├── styles.css             # single stylesheet
├── articles.js            # article data
├── assets/                # thumbnails, school logos, resume
├── compress-thumbnails.py # one-shot image optimization
└── deploy.sh / deploy.bat # one-shot push to GitHub Pages
```

---

## Local development

```bash
# clone and serve
python -m http.server 8000
# visit http://localhost:8000
```

## Deploy

```bash
./deploy.sh "your commit message"
```

GitHub Pages auto-rebuilds on push to `main`. Live in ~30–60s.

---

© 2026 Fahim Ahamed
