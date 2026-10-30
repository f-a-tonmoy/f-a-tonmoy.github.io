// Runs on every push (.github/workflows/check-articles.yml). Locally:  node tools/check-articles.js
// Reports every problem in assets/data/articles.js that would otherwise misrender silently: a wrong
// image path shows a grey placeholder, a category with no tab only appears under "All".
const fs = require('fs');
const path = require('path');

const dir = process.argv[2] || path.join(__dirname, '..'); // repo root
const read = (f) => fs.readFileSync(path.join(dir, f), 'utf8');
const articles = new Function(read('assets/data/articles.js') + '; return articles;')();
const tabs = [...read('pages/writing.html').matchAll(/data-filter="([^"]+)"/g)].map((m) => m[1]).filter((t) => t !== 'All');

const problems = [];
const seen = { url: new Map(), title: new Map(), featured: new Map(), category: new Map() };
const dup = (kind, key, label) => {
  if (seen[kind].has(key)) problems.push(`${kind} used twice: ${key} (${seen[kind].get(key)} and ${label})`);
  else seen[kind].set(key, label);
};

articles.forEach((a, i) => {
  const label = `#${i + 1} "${String(a.title).slice(0, 40)}"`;
  for (const field of ['title', 'url', 'image', 'category']) {
    if (typeof a[field] !== 'string' || !a[field].trim()) problems.push(`${label}: missing ${field}`);
  }
  dup('url', a.url, label);
  dup('title', a.title, label);
  if (a.image && !fs.existsSync(path.join(dir, a.image))) problems.push(`${label}: image not found: ${a.image}`);
  const dark = a.image && a.image.replace('/light/', '/dark/');
  if (dark && dark !== a.image && !fs.existsSync(path.join(dir, dark))) problems.push(`${label}: dark version not found: ${dark}`);
  if (!tabs.includes(a.category)) problems.push(`${label}: category "${a.category}" has no tab on pages/writing.html`);
  if (a.featured) {
    dup('featured', String(a.featured), label);
    dup('category', a.category, label); // the homepage shows one featured piece per category
  }
});

const order = [...seen.featured.keys()].map(Number).sort((a, b) => a - b).join(',');
if (order !== tabs.map((_, i) => i + 1).join(',')) {
  problems.push(`featured must be numbered 1..${tabs.length} (one per tab), found: ${order || 'none'}`);
}

if (problems.length) {
  console.error(problems.map((p) => '  - ' + p).join('\n'));
  console.error(`\n${problems.length} problem(s) in ${articles.length} articles.`);
  process.exit(1);
}
console.log(`OK: ${articles.length} articles, ${tabs.length} tabs, featured 1..${tabs.length}.`);
