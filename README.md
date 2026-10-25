# Fahim Ahamed — Portfolio

Personal portfolio site for **Fahim Ahamed** — data scientist with a background in applied AI research.

**Live**: [f-a-tonmoy.github.io](https://f-a-tonmoy.github.io)

---

## About

Data scientist with a background in applied AI research — turning data into insights, predictive models, and decisions across healthcare, NLP, computer vision, and beyond.

- **4+ peer-reviewed AI research papers** (SAGE, Springer, IEEE), 2 more accepted or under review
- **85+ articles** on data science, ML, and responsible AI
- Specialties: explainable AI, computer vision, NLP, adversarial robustness
- Based in NYC · U.S. Permanent Resident · No sponsorship needed

**[Resume](assets/Resume%20-%20Fahim%20Ahamed.pdf)** &nbsp;·&nbsp; **[LinkedIn](https://linkedin.com/in/f-a-tonmoy)** &nbsp;·&nbsp; **[ORCID](https://orcid.org/0009-0006-2638-6521)** &nbsp;·&nbsp; **[Email](mailto:f.a.tonmoy00@gmail.com)**

---

## Pages

| | |
|---|---|
| [Home](https://f-a-tonmoy.github.io) | Overview, current roles, featured work, selected writing |
| [Experience](https://f-a-tonmoy.github.io/experience.html) | Timeline of roles + education |
| [Research](https://f-a-tonmoy.github.io/research.html) | Published papers + manuscripts in pipeline |
| [Projects](https://f-a-tonmoy.github.io/projects.html) | Treemap overview, then ML, deep learning, public health analytics, cybersecurity, recommendation systems, software engineering |
| [Data Stories](https://f-a-tonmoy.github.io/data-stories.html) | Long-form analyses of public datasets, published as interactive articles |
| [Writing](https://f-a-tonmoy.github.io/writing.html) | 85+ technical articles, filterable + searchable |

---

## Design

The UI is a "Data-Noir" system: a near-black base with a plotting-grid background, one emerald accent, square 3px corners, and two typefaces, **Space Grotesk** for text and **JetBrains Mono** for labels, dates, and metadata.

- **Light and dark themes.** First visit follows the system setting; the toggle overrides it and the choice is remembered.
- **Theme-matched project art.** Every project thumbnail comes in a light and a dark version, and the page swaps between them with the theme.
- **Projects treemap.** An at-a-glance board of all ten projects, sized by how many each group holds; clicking a tile jumps to the project and draws a trace around its card.
- **Click to enlarge** on project thumbnails and data-story images (native `<dialog>`, keyboard accessible).
- **Contact.** Email links copy the address on desktop and open the mail app on touch devices.
- **No third-party requests on load.** Fonts are self-hosted, and reduced-motion is respected throughout.

---

## Stack

Pure static site — **HTML · CSS · vanilla JavaScript**. No build step. Deployed via GitHub Pages.

```
.
├── index.html / experience.html / research.html / projects.html
│   data-stories.html / writing.html
├── 404.html       # branded not-found page
├── styles.css     # single stylesheet
├── site.js        # shared chrome injection + interactivity
├── articles.js    # article data + the shared article-card renderer
├── check-articles.js  # `node check-articles.js`: catches bad article data before a push
├── .nojekyll      # tells GitHub Pages to skip Jekyll (pure-HTML site)
└── assets/
    ├── project-thumbnails/{light,dark}/   # one set per theme
    ├── data-stories/  article-thumbnails/  school-logos/
    ├── webfonts/      # Space Grotesk + JetBrains Mono, Latin subset, woff2
    └── avatar, favicon, OG image, resume
```

---

## Local development

```bash
# clone and serve
python -m http.server 8000
# visit http://localhost:8000

# before pushing, after editing articles.js
node check-articles.js
```

## Deploy

```bash
git add .
git commit -m "your commit message"
git push
```

GitHub Pages auto-rebuilds on push to `main`. Live in ~30–60s.

---

© 2026 Fahim Ahamed
