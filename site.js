/* Shared site chrome — markup + behavior loaded on every page.
   - Injects the header nav, footer, theme-toggle, and back-to-top.
   - Wires up all interactive handlers (nav drawer, scroll-reveal, theme persist, etc.).
   - Each handler is guarded by element existence so unrelated pages no-op. */
(function () {
  // ============================================================
  //  1. Inject chrome markup
  // ============================================================

  var path = window.location.pathname;
  var currentPage = (path.split('/').pop() || 'index.html').toLowerCase();

  function ariaCurrent(name) {
    return currentPage === name + '.html' ? ' aria-current="page"' : '';
  }

  function headerHTML() {
    return ''
      + '<a class="brand" href="/" aria-label="Home">'
      +   '<svg class="brand-icon" viewBox="0 0 24 24" aria-hidden="true">'
      +     '<path d="M3.75 10.5 12 3.75l8.25 6.75v8.25a1.5 1.5 0 0 1-1.5 1.5h-4.5v-5.5h-4.5v5.5h-4.5a1.5 1.5 0 0 1-1.5-1.5V10.5Z" />'
      +   '</svg>'
      + '</a>'
      + '<nav class="nav" aria-label="Primary navigation">'
      +   '<a href="/experience.html"' + ariaCurrent('experience') + '>Experience</a>'
      +   '<a href="/research.html"'   + ariaCurrent('research')   + '>Research</a>'
      +   '<a href="/projects.html"'   + ariaCurrent('projects')   + '>Projects</a>'
      +   '<a href="/writing.html"'    + ariaCurrent('writing')    + '>Writing</a>'
      +   '<a class="nav-mobile-action" href="/assets/Resume%20-%20Fahim%20Ahamed.pdf" target="_blank" rel="noopener">Resume</a>'
      +   '<a class="nav-mobile-action" href="https://linkedin.com/in/f-a-tonmoy" target="_blank" rel="noopener">Contact</a>'
      + '</nav>'
      + '<div class="header-actions">'
      +   '<a class="header-cta" href="/assets/Resume%20-%20Fahim%20Ahamed.pdf" target="_blank" rel="noopener">Resume</a>'
      +   '<a class="header-cta" href="https://linkedin.com/in/f-a-tonmoy" target="_blank" rel="noopener">Contact</a>'
      + '</div>'
      + '<button class="nav-toggle" aria-label="Toggle navigation" aria-expanded="false">'
      +   '<span></span><span></span><span></span>'
      + '</button>';
  }

  function footerHTML() {
    return ''
      + '<div class="footer-inner">'
      +   '<span class="footer-copy">&copy; 2026 Fahim Ahamed</span>'
      +   '<div class="footer-links">'
      +     '<a href="https://github.com/f-a-tonmoy" target="_blank" rel="noopener">GitHub</a>'
      +     '<span class="email-action">'
      +       '<a href="mailto:f.a.tonmoy00@gmail.com">Email</a>'
      +       '<button class="copy-email" type="button" data-email="f.a.tonmoy00@gmail.com" aria-label="Copy email address" title="Copy email">'
      +         '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H8V7h11v14z"/></svg>'
      +       '</button>'
      +     '</span>'
      +   '</div>'
      + '</div>';
  }

  function iconButton(cls, label, svg, title) {
    var btn = document.createElement('button');
    btn.className = cls;
    btn.type = 'button';
    btn.setAttribute('aria-label', label);
    if (title) btn.title = title;
    btn.innerHTML = svg;
    return btn;
  }

  var THEME_SVG = ''
    + '<svg class="theme-icon-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"/></svg>'
    + '<svg class="theme-icon-sun" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm0-6a1 1 0 0 1 1 1v2a1 1 0 1 1-2 0V2a1 1 0 0 1 1-1zm0 18a1 1 0 0 1 1 1v2a1 1 0 1 1-2 0v-2a1 1 0 0 1 1-1zm11-7a1 1 0 0 1-1 1h-2a1 1 0 1 1 0-2h2a1 1 0 0 1 1 1zM4 12a1 1 0 0 1-1 1H1a1 1 0 1 1 0-2h2a1 1 0 0 1 1 1zm15.07-7.07a1 1 0 0 1 0 1.41l-1.41 1.42a1 1 0 1 1-1.42-1.42l1.42-1.41a1 1 0 0 1 1.41 0zM7.76 16.24a1 1 0 0 1 0 1.41l-1.41 1.42a1 1 0 1 1-1.42-1.42l1.42-1.41a1 1 0 0 1 1.41 0zm11.31 2.83a1 1 0 0 1-1.41 0l-1.42-1.41a1 1 0 1 1 1.42-1.42l1.41 1.42a1 1 0 0 1 0 1.41zM7.76 7.76a1 1 0 0 1-1.41 0L4.93 6.35a1 1 0 0 1 1.42-1.42l1.41 1.42a1 1 0 0 1 0 1.41z"/></svg>';

  var TOP_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l-8 8h5v8h6v-8h5z"/></svg>';

  // Populate header (children.length ignores whitespace text nodes between tags)
  var header = document.querySelector('.site-header');
  if (header && header.children.length === 0) header.innerHTML = headerHTML();

  // Populate footer
  var footer = document.querySelector('.site-footer');
  if (footer && footer.children.length === 0) footer.innerHTML = footerHTML();

  // Append floating buttons (idempotent — skips if already present in HTML)
  if (!document.querySelector('.theme-toggle')) {
    document.body.appendChild(iconButton('theme-toggle', 'Toggle dark mode', THEME_SVG, 'Toggle theme'));
  }
  if (!document.querySelector('.back-to-top')) {
    document.body.appendChild(iconButton('back-to-top', 'Back to top', TOP_SVG));
  }

  // ============================================================
  //  2. Wire up handlers (run AFTER injection so elements exist)
  // ============================================================

  // Mobile nav drawer toggle
  var navToggle = document.querySelector('.nav-toggle');
  var nav = document.querySelector('.nav');
  if (navToggle && nav) {
    navToggle.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      navToggle.classList.toggle('open', open);
      navToggle.setAttribute('aria-expanded', String(open));
    });
  }

  // Back-to-top button (visible after 500px scroll)
  var btt = document.querySelector('.back-to-top');
  if (btt) {
    var onScroll = function () {
      btt.classList.toggle('visible', window.scrollY > 500);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    btt.addEventListener('click', function () {
      var prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      window.scrollTo({ top: 0, behavior: prefersReduced ? 'auto' : 'smooth' });
    });
  }

  // Theme toggle (persists to localStorage; theme-init script in <head> handles the cold-load)
  // Uses the View Transitions API for a circular reveal animation from the click point.
  // Gracefully falls back to instant toggle on unsupported browsers and reduced-motion users.
  var themeBtn = document.querySelector('.theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', function (e) {
      var current = document.documentElement.getAttribute('data-theme');
      var systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      var nextDark;
      if (current === 'dark') nextDark = false;
      else if (current === 'light') nextDark = true;
      else nextDark = !systemDark;
      var next = nextDark ? 'dark' : 'light';

      var apply = function () {
        document.documentElement.setAttribute('data-theme', next);
        try { localStorage.setItem('theme', next); } catch (err) {}
      };

      var prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

      // Instant fallback when View Transitions API isn't supported or user wants reduced motion
      if (!document.startViewTransition || prefersReduced) {
        apply();
        return;
      }

      // Capture click point for the circular-reveal origin; compute end radius so
      // the circle is guaranteed to cover the whole viewport from that point.
      var x = e.clientX;
      var y = e.clientY;
      var endRadius = Math.hypot(
        Math.max(x, window.innerWidth - x),
        Math.max(y, window.innerHeight - y)
      );

      document.documentElement.style.setProperty('--theme-anim-x', x + 'px');
      document.documentElement.style.setProperty('--theme-anim-y', y + 'px');
      document.documentElement.style.setProperty('--theme-anim-r', endRadius + 'px');

      // Newer API (Chrome 125+, Safari 18.2+) accepts {update, types}; older just takes a fn
      var vt;
      try {
        vt = document.startViewTransition({ update: apply, types: ['theme'] });
      } catch (err) {
        vt = document.startViewTransition(apply);
      }

      // Drive the reveal from here rather than leaning on the CSS keyframes: the
      // custom properties above do not reach ::view-transition-new(root), so those
      // keyframes fall back to `50% 50%` and the circle opens from the middle of
      // the snapshot instead of the button. Passing literal pixels avoids the
      // substitution entirely. A script animation outranks the CSS one, so the
      // keyframes stay as a fallback where pseudoElement animation is unsupported.
      if (vt && vt.ready && document.documentElement.animate) {
        vt.ready.then(function () {
          document.documentElement.animate(
            {
              clipPath: [
                'circle(0px at ' + x + 'px ' + y + 'px)',
                'circle(' + endRadius + 'px at ' + x + 'px ' + y + 'px)'
              ]
            },
            {
              duration: 500,
              easing: 'cubic-bezier(0.22, 0.61, 0.36, 1)',
              pseudoElement: '::view-transition-new(root)'
            }
          );
        }).catch(function () {});
      }
    });
  }

  // Intercept clicks on nav links pointing to the CURRENT page — prevents the redundant
  // reload (which would trigger an unwanted page-transition flicker). Instead, scroll to
  // top + brief pulse on the link as an acknowledgement.
  function isSamePage(href) {
    var norm = function (p) { return p.replace(/\/index\.html$/, '/').replace(/(.)\/$/, '$1'); };
    try {
      var url = new URL(href, window.location.href);
      return url.origin === window.location.origin
        && norm(url.pathname) === norm(window.location.pathname);
    } catch (err) {
      return false;
    }
  }

  document.querySelectorAll('.brand, .nav > a').forEach(function (link) {
    link.addEventListener('click', function (e) {
      if (link.target === '_blank') return; // external / new-tab links unaffected
      if (!isSamePage(link.href)) return;
      var url = new URL(link.href, window.location.href);
      if (url.hash) return; // anchor link — let the browser jump normally

      e.preventDefault();
      var prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      window.scrollTo({ top: 0, behavior: prefersReduced ? 'auto' : 'smooth' });

      // Brief pulse to confirm the click registered
      link.classList.remove('nav-pulse'); // re-trigger animation on rapid clicks
      void link.offsetWidth;
      link.classList.add('nav-pulse');
      setTimeout(function () { link.classList.remove('nav-pulse'); }, 400);
    });
  });

  // Scroll-reveal (no-op when reduced-motion is set; CSS handles fallback)
  var revealEls = document.querySelectorAll('.reveal');
  if (revealEls.length && 'IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    revealEls.forEach(function (el) { io.observe(el); });
  } else if (revealEls.length) {
    revealEls.forEach(function (el) { el.classList.add('is-visible'); });
  }

  // ============================================================
  //  3. Interactivity polish
  // ============================================================

  var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var canHover = window.matchMedia('(hover: hover)').matches;

  // --- Copy-email button + toast -----------------------------------
  var copyBtn = document.querySelector('.copy-email');
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      var email = copyBtn.dataset.email || '';
      var done = function (ok) {
        showToast(ok ? 'Email copied to clipboard' : 'Couldn’t copy — try selecting');
      };
      navigator.clipboard.writeText(email).then(function () { done(true); }, function () { done(false); });
    });
  }

  function showToast(msg) {
    var toast = document.querySelector('.toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.className = 'toast';
      toast.setAttribute('role', 'status');
      toast.setAttribute('aria-live', 'polite');
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    // Force reflow so the transition runs even if the toast was just created
    void toast.offsetWidth;
    toast.classList.add('visible');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(function () {
      toast.classList.remove('visible');
    }, 1800);
  }

  // --- Missing thumbnail -> labelled placeholder -------------------
  // Delegated in the capture phase because error events don't bubble. The sweep
  // afterwards catches any image that already failed before this script ran.
  function thumbFailed(img) {
    var holder = img.closest('.project-thumb');
    if (!holder || holder.classList.contains('is-empty')) return;
    var heading = img.closest('.project') && img.closest('.project').querySelector('h3');
    holder.className = 'project-thumb is-empty';
    holder.setAttribute('data-fallback', heading ? heading.textContent.trim() : 'Project');
    ['role', 'tabindex', 'aria-label'].forEach(function (a) { holder.removeAttribute(a); });
    img.remove();
  }

  document.addEventListener('error', function (e) {
    if (e.target.tagName === 'IMG' && e.target.closest('.project-thumb')) thumbFailed(e.target);
  }, true);

  document.querySelectorAll('.project-thumb img').forEach(function (img) {
    if (img.complete && !img.naturalWidth) thumbFailed(img);
  });

  // --- Landing trace on the card jumped to from the index map ------
  // The path starts at the middle of the left edge rather than a corner, so the
  // line closes on a straight run instead of meeting itself mid-curve.
  var SVGNS = 'http://www.w3.org/2000/svg';

  function traceCard(card) {
    // clear any trace anywhere, not just this card's: jumping between cards
    // would otherwise leave the previous one behind
    document.querySelectorAll('.trace-svg').forEach(function (n) { n.remove(); });

    var box = card.getBoundingClientRect();
    var pad = 10, r = 14;
    var w = box.width + pad * 2, h = box.height + pad * 2, cy = h / 2;

    var svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('class', 'trace-svg');
    svg.setAttribute('width', w);
    svg.setAttribute('height', h);
    svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    svg.setAttribute('aria-hidden', 'true');

    var path = document.createElementNS(SVGNS, 'path');
    path.setAttribute('pathLength', '100'); // lets one dasharray fit every card size
    path.setAttribute('d',
      'M0 ' + cy + 'L0 ' + r +
      'A' + r + ' ' + r + ' 0 0 1 ' + r + ' 0' +
      'L' + (w - r) + ' 0' +
      'A' + r + ' ' + r + ' 0 0 1 ' + w + ' ' + r +
      'L' + w + ' ' + (h - r) +
      'A' + r + ' ' + r + ' 0 0 1 ' + (w - r) + ' ' + h +
      'L' + r + ' ' + h +
      'A' + r + ' ' + r + ' 0 0 1 0 ' + (h - r) + 'Z');

    svg.appendChild(path);
    card.appendChild(svg);
    // draw + fade run 1s total; drop the node just after, so nothing lingers
    setTimeout(function () { svg.remove(); }, 1200);
  }

  // Smooth-scroll duration depends on how far the jump is, so a fixed delay
  // either fires early on long jumps or dawdles on short ones. Wait for the
  // scroll position to actually stop moving instead.
  // Polled on a timer rather than requestAnimationFrame: rAF is suspended while
  // the document is hidden, which would strand the trace on a background tab.
  function afterScrollSettles(fn) {
    var last = window.scrollY, still = 0, ticks = 0;
    var timer = setInterval(function () {
      var y = window.scrollY;
      still = (y === last) ? still + 1 : 0;
      last = y;
      if (still >= 3 || ++ticks > 200) { // ~60ms at rest, 4s hard cap
        clearInterval(timer);
        fn();
      }
    }, 20);
  }

  function traceHashTarget() {
    var el = location.hash.length > 1 && document.querySelector(location.hash);
    if (!el || !el.classList.contains('project')) return;
    afterScrollSettles(function () { traceCard(el); });
  }

  window.addEventListener('hashchange', traceHashTarget);
  traceHashTarget();

  // --- Thumbnail lightbox -----------------------------------------
  // Native <dialog> supplies the focus trap, Esc-to-close, and backdrop, so
  // there's nothing here but wiring. Placeholders hold no <img>, so they're
  // skipped automatically.
  var zoomThumbs = document.querySelectorAll('.project-thumb img');
  if (zoomThumbs.length) {
    var lightbox = document.createElement('dialog');
    lightbox.className = 'lightbox';
    lightbox.innerHTML = '<img alt="" /><button class="lightbox-close" type="button" aria-label="Close">&times;</button>';
    document.body.appendChild(lightbox);
    var lightboxImg = lightbox.querySelector('img');

    zoomThumbs.forEach(function (img) {
      var holder = img.closest('.project-thumb');
      holder.classList.add('is-zoomable');
      holder.setAttribute('role', 'button');
      holder.setAttribute('tabindex', '0');
      holder.setAttribute('aria-label', 'View larger image');

      var openLightbox = function () {
        lightboxImg.src = img.currentSrc || img.src;
        lightbox.showModal();
      };

      holder.addEventListener('click', openLightbox);
      holder.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          openLightbox();
        }
      });
    });

    // Close on backdrop click (target is the dialog itself) or the X button
    lightbox.addEventListener('click', function (e) {
      if (e.target === lightbox || e.target.closest('.lightbox-close')) lightbox.close();
    });
  }

  // --- Cursor-aware glow on cards ---------------------------------
  // Only wires up on hover-capable, no-reduced-motion devices
  if (canHover && !prefersReducedMotion) {
    var glowCards = document.querySelectorAll('.stats > div, .card, .education-grid > article, .profile-panel, .article-card, .timeline-card, .contact');
    glowCards.forEach(function (el) {
      // Measure once on entry. Reading the rect inside mousemove forces a
      // synchronous layout on every event — up to 120 a second while hovering,
      // and the articles page has 89 of these cards.
      var rect = null;
      el.addEventListener('mouseenter', function () { rect = el.getBoundingClientRect(); });
      el.addEventListener('mousemove', function (e) {
        if (!rect) rect = el.getBoundingClientRect();
        el.style.setProperty('--mx', (e.clientX - rect.left) + 'px');
        el.style.setProperty('--my', (e.clientY - rect.top) + 'px');
      });
      el.addEventListener('mouseleave', function () { rect = null; });
    });

    // --- Magnetic primary/secondary buttons ------------------------
    var magnetButtons = document.querySelectorAll('.button');
    magnetButtons.forEach(function (btn) {
      var bRect = null;
      btn.addEventListener('mouseenter', function () { bRect = btn.getBoundingClientRect(); });
      btn.addEventListener('mousemove', function (e) {
        var rect = bRect || (bRect = btn.getBoundingClientRect());
        var dx = e.clientX - rect.left - rect.width / 2;
        var dy = e.clientY - rect.top - rect.height / 2;
        // Pull strength ~25% of cursor distance from button center
        btn.style.setProperty('--mag-x', (dx * 0.25).toFixed(1) + 'px');
        btn.style.setProperty('--mag-y', (dy * 0.25).toFixed(1) + 'px');
      });
      btn.addEventListener('mouseleave', function () {
        bRect = null;
        btn.style.setProperty('--mag-x', '0px');
        btn.style.setProperty('--mag-y', '0px');
      });
    });
  }
})();
