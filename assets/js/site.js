/* Shared site chrome — markup + behavior loaded on every page.
   - Injects the header nav, footer, theme-toggle, and back-to-top.
   - Wires up all interactive handlers (nav drawer, scroll-reveal, theme persist, etc.).
   - Each handler is guarded by element existence so unrelated pages no-op. */
(function () {
  // ============================================================
  //  1. Inject chrome markup
  // ============================================================

  var path = window.location.pathname;
  var currentPage = (function () {
    if (/\/experience\.html$/i.test(path))   return 'experience';
    if (/\/research\.html$/i.test(path))     return 'research';
    if (/\/projects\.html$/i.test(path))     return 'projects';
    if (/\/data-stories\.html$/i.test(path)) return 'data-stories';
    if (/\/writing\.html$/i.test(path))      return 'writing';
    return null; // home, 404, anything else
  })();

  function ariaCurrent(name) {
    return currentPage === name ? ' aria-current="page"' : '';
  }

  function headerHTML() {
    return ''
      + '<div class="header-inner">'
      + '<a class="brand" href="/" aria-label="Home">'
      // One continuous stroke: apex, left roof, left wall, floor in to the
      // doorway, up and over the opening, back out along the floor, right wall,
      // closed to the apex. The door is a notch out of the floor, not a shape.
      // Reference proportions: roof 36% / walls 64% of height, doorway 33% of
      // the house width and 64% of the wall height.
      +   '<svg class="brand-icon" viewBox="0 0 24 24" aria-hidden="true">'
      +     '<path d="M4 10.4 12 4.2l8 6.2V19a1.6 1.6 0 0 1-1.6 1.6H15v-5.8H9v5.8H5.6A1.6 1.6 0 0 1 4 19z" />'
      +   '</svg>'
      + '</a>'
      + '<nav class="nav" aria-label="Primary navigation">'
      +   '<a href="/pages/experience.html"' + ariaCurrent('experience') + '>Experience</a>'
      +   '<a href="/pages/research.html"'   + ariaCurrent('research')   + '>Research</a>'
      +   '<a href="/pages/data-stories.html"' + ariaCurrent('data-stories') + '>Data Stories</a>'
      +   '<a href="/pages/projects.html"'   + ariaCurrent('projects')   + '>Projects</a>'
      +   '<a href="/pages/writing.html"'    + ariaCurrent('writing')    + '>Writing</a>'
      +   '<a class="nav-mobile-action" href="/assets/fahim-ahamed-resume.pdf" target="_blank" rel="noopener">Resume</a>'
      +   '<a class="nav-mobile-action" href="mailto:f.a.tonmoy00@gmail.com">Contact</a>'
      + '</nav>'
      + '<div class="header-actions">'
      // Document and envelope drawn on the same 24-unit grid and 2-unit stroke as the theme
      // toggle, so the header icons read at one weight. On a pointer device the envelope copies the address.
      +   '<a class="header-cta header-cta-icon" href="/assets/fahim-ahamed-resume.pdf" target="_blank" rel="noopener" aria-label="View resume" data-tip="View resume">'
      +     '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.2 3.4H7a2 2 0 0 0-2 2v13.2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.2z" /><path d="M14.2 3.4v4.8H19" /><circle cx="12" cy="12.4" r="1.9" /><path d="M8.7 17.8a3.3 3.3 0 0 1 6.6 0" /></svg>'
      +   '</a>'
      +   '<a class="header-cta header-cta-icon" href="mailto:f.a.tonmoy00@gmail.com" aria-label="Copy email address" data-tip="Copy email address">'
      +     '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3.2" y="5.2" width="17.6" height="13.6" rx="2.2" /><path d="m3.8 7.4 7.3 5.3a1.6 1.6 0 0 0 1.8 0l7.3-5.3" /></svg>'
      +   '</a>'
      + '</div>'
      + '<button class="nav-toggle" aria-label="Toggle navigation" aria-expanded="false">'
      +   '<span></span><span></span><span></span>'
      + '</button>'
      + '</div>';
  }

  function footerHTML() {
    return ''
      + '<div class="footer-inner">'
      +   '<span class="footer-copy">&copy; 2026 Fahim Ahamed</span>'
      +   '<div class="footer-links">'
      +     '<a href="https://github.com/f-a-tonmoy" target="_blank" rel="noopener">GitHub</a>'
      +     '<span class="email-action">'
      +       '<a href="mailto:f.a.tonmoy00@gmail.com">Email</a>'
      +       '<button class="copy-email" type="button" data-email="f.a.tonmoy00@gmail.com" aria-label="Copy email address" data-tip="Copy email">'
      +         '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8.4" y="8.4" width="12.2" height="12.2" rx="2.2" /><path d="M15.6 8.4V5.6a2.2 2.2 0 0 0-2.2-2.2H5.6a2.2 2.2 0 0 0-2.2 2.2v7.8a2.2 2.2 0 0 0 2.2 2.2h2.8" /></svg>'
      +       '</button>'
      +     '</span>'
      +   '</div>'
      + '</div>';
  }

  function makeThemeToggle() {
    var btn = document.createElement('button');
    btn.className = 'theme-toggle';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Toggle dark mode');
    btn.dataset.tip = 'Toggle theme';
    btn.innerHTML = ''
      + '<svg class="theme-icon-moon" viewBox="0 0 24 24" aria-hidden="true">'
      +   '<path d="M20.2 13.95A8.2 8.2 0 1 1 10.05 3.8a6.5 6.5 0 0 0 10.15 10.15z"/>'
      + '</svg>'
      + '<svg class="theme-icon-sun" viewBox="0 0 24 24" aria-hidden="true">'
      +   '<circle cx="12" cy="12" r="3.8"/>'
      +   '<path d="M12 3v1.9M12 19.1V21M21 12h-1.9M4.9 12H3"/>'
      +   '<path d="M18.4 5.6l-1.35 1.35M6.95 17.05 5.6 18.4M18.4 18.4l-1.35-1.35M6.95 6.95 5.6 5.6"/>'
      + '</svg>';
    return btn;
  }

  function makeBackToTop() {
    var btn = document.createElement('button');
    btn.className = 'back-to-top';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Back to top');
    btn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true">'
      + '<path d="M12 19V5.6"/>'
      + '<path d="m6.2 11.4 5.8-5.8 5.8 5.8"/>'
      + '</svg>';
    return btn;
  }

  // Populate header (children.length ignores whitespace text nodes between tags)
  var header = document.querySelector('.site-header');
  if (header && header.children.length === 0) header.innerHTML = headerHTML();

  // Populate footer
  var footer = document.querySelector('.site-footer');
  if (footer && footer.children.length === 0) footer.innerHTML = footerHTML();

  // Append floating buttons (idempotent — skips if already present in HTML)
  if (!document.querySelector('.theme-toggle')) document.body.appendChild(makeThemeToggle());
  if (!document.querySelector('.back-to-top')) document.body.appendChild(makeBackToTop());

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

  // Scroll state: back-to-top visibility (after 500px) and the header's lifted
  // edge (after 40px). One listener drives both.
  var btt = document.querySelector('.back-to-top');
  if (btt || header) {
    var onScroll = function () {
      if (btt) btt.classList.toggle('visible', window.scrollY > 500);
      if (header) header.classList.toggle('is-scrolled', window.scrollY > 40);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  if (btt) {
    btt.addEventListener('click', function () {
      var prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      window.scrollTo({ top: 0, behavior: prefersReduced ? 'auto' : 'smooth' });
    });
  }

  // Theme toggle (persists to localStorage; theme-init script in <head> handles the cold-load)
  // Uses the View Transitions API for a top-to-bottom scanline wipe (see theme-scan in styles.css).
  // Gracefully falls back to instant toggle on unsupported browsers and reduced-motion users.
  var themeBtn = document.querySelector('.theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
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
        syncThumbArt(); // baked art can't follow tokens; swap it with the theme
      };

      var prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

      // Instant fallback when View Transitions API isn't supported or user wants reduced motion
      if (!document.startViewTransition || prefersReduced) {
        apply();
        return;
      }

      // Newer API (Chrome 125+, Safari 18.2+) accepts {update, types}; older just takes a fn
      try {
        document.startViewTransition({ update: apply, types: ['theme'] });
      } catch (err) {
        document.startViewTransition(apply);
      }
    });
  }

  // Intercept clicks on nav links pointing to the CURRENT page — prevents the redundant
  // reload (which would trigger an unwanted page-transition flicker). Instead, scroll to
  // top + brief pulse on the link as an acknowledgement.
  function isSamePage(href) {
    if (!href) return false;
    try {
      var url = new URL(href, window.location.href);
      if (url.origin !== window.location.origin) return false;
      var normalize = function (p) {
        if (!p || p === '/' || p === '/index.html') return '/';
        return p.replace(/\/$/, '');
      };
      return normalize(url.pathname) === normalize(window.location.pathname);
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
  // mailto: is the right affordance on a phone (taps straight into the mail app)
  // but can be a dead click on a desktop with no registered client. So touch
  // navigates and pointer devices copy instead. canHover is the proxy for "has a
  // real pointer" -- a width query would misfire on a narrow desktop window.
  var copyTargets = document.querySelectorAll('.copy-email, a[href^="mailto:"]');
  Array.prototype.forEach.call(copyTargets, function (el) {
    el.addEventListener('click', function (ev) {
      // The footer copy button has no href, so it copies on every device.
      if (!canHover && el.hasAttribute('href')) return;
      ev.preventDefault();
      var email = el.dataset.email || (el.getAttribute('href') || '').replace('mailto:', '');
      if (!navigator.clipboard) { showToast('', el, email); return; }
      navigator.clipboard.writeText(email).then(
        function () { showToast('Email copied to clipboard', el); },
        function () { showToast('Copy failed: ', el, email); }
      );
    });
  });

  // --- Themed tooltips ---------------------------------------------
  // [data-tip] stands in for title=, whose native bubble ignores the theme. One fixed
  // element, so a tip escapes overflow:hidden parents and is clamped to the viewport.
  // Hover devices only: on touch, a tap already does the thing the tip would describe.
  if (canHover) {
    var tip = document.createElement('div');
    tip.className = 'tip';
    tip.setAttribute('aria-hidden', 'true'); // the controls carry aria-labels already
    document.body.appendChild(tip);

    var showTip = function (el) {
      tip.textContent = el.dataset.tip;
      var r = el.getBoundingClientRect();
      var y = r.bottom + 8;
      if (y + tip.offsetHeight > document.documentElement.clientHeight - 8) y = r.top - tip.offsetHeight - 8;
      var half = tip.offsetWidth / 2;
      var cx = Math.min(Math.max((r.left + r.right) / 2, half + 8), document.documentElement.clientWidth - half - 8);
      tip.style.left = Math.round(cx - half) + 'px';
      tip.style.top = Math.round(y) + 'px';
      tip.classList.add('visible');
    };
    var hideTip = function () { tip.classList.remove('visible'); };

    document.addEventListener('mouseover', function (e) {
      var el = e.target.closest('[data-tip]');
      if (el) showTip(el);
    });
    document.addEventListener('mouseout', function (e) {
      var el = e.target.closest('[data-tip]');
      if (el && !el.contains(e.relatedTarget)) hideTip();
    });
    document.addEventListener('focusin', function (e) {
      var el = e.target.closest('[data-tip]');
      if (el && el.matches(':focus-visible')) showTip(el);
    });
    document.addEventListener('focusout', hideTip);
    // a click hands over to the toast or a new tab; scrolling detaches the tip from its control
    document.addEventListener('click', hideTip);
    window.addEventListener('scroll', hideTip, { passive: true });
  }

  // Pass `anchor` to park the toast against the control that triggered it. The
  // default bottom-of-window spot is a whole viewport away from a click in the
  // header, so a visitor reads "nothing happened" before they ever look down.
  function showToast(msg, anchor, address) {
    var toast = document.querySelector('.toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.className = 'toast';
      toast.setAttribute('role', 'status');
      toast.setAttribute('aria-live', 'polite');
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    // an address shown for copying by hand stays in its own case; the toast sets the rest in caps
    if (address) {
      var addr = document.createElement('span');
      addr.className = 'toast-address';
      addr.textContent = address;
      toast.appendChild(addr);
    }
    toast.classList.toggle('anchored', !!anchor);
    // Force reflow so the transition runs even if the toast was just created,
    // and so offsetHeight below measures the new text.
    void toast.offsetWidth;
    if (anchor) {
      var r = anchor.getBoundingClientRect();
      var y = r.bottom + 10;
      // clientWidth/Height, not innerWidth/Height: those include the scrollbar,
      // which position:fixed does not, and the toast would sit a scrollbar-width off.
      var vw = document.documentElement.clientWidth;
      if (y + toast.offsetHeight > document.documentElement.clientHeight - 8) y = r.top - toast.offsetHeight - 10;
      // Centred on the trigger, then clamped: the header CTAs sit close to the
      // viewport edge, where a toast wider than its trigger would overflow.
      var half = toast.offsetWidth / 2;
      var cx = Math.min(Math.max((r.left + r.right) / 2, half + 8), vw - half - 8);
      toast.style.setProperty('--toast-x', Math.round(cx) + 'px');
      toast.style.setProperty('--toast-y', Math.round(y) + 'px');
    }
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
    var card = img.closest('.project');
    var heading = card && card.querySelector('h3');
    holder.className = 'project-thumb is-empty';
    holder.setAttribute('data-fallback', heading ? heading.textContent.trim() : 'Project');
    img.remove();
  }

  // A thumbnail that fails after the lightbox wired it must stop being a button, or
  // it stays a tab stop that opens an empty dialog. Covers the story cards too,
  // whose own inline onerror only swaps in the placeholder.
  function unzoom(holder) {
    holder.classList.remove('is-zoomable');
    ['role', 'tabindex', 'aria-label'].forEach(function (a) { holder.removeAttribute(a); });
  }

  document.addEventListener('error', function (e) {
    if (e.target.tagName !== 'IMG') return;
    var holder = e.target.closest('.project-thumb, .story-media');
    if (!holder) return;
    unzoom(holder);
    if (holder.classList.contains('project-thumb')) thumbFailed(e.target);
  }, true);

  document.querySelectorAll('.project-thumb img').forEach(function (img) {
    if (img.complete && !img.naturalWidth) thumbFailed(img);
  });

  // --- Theme-matched thumbnail art --------------------------------
  // The project thumbnails are baked raster charts, so they cannot follow CSS
  // tokens. make_thumbnails.py emits a light/ and a dark/ set; swap the folder
  // segment to match the active theme. Every thumbnail is loading="lazy" and
  // below the fold, so this runs before any of them are fetched: no flash and
  // no wasted request.
  function currentTheme() {
    var set = document.documentElement.getAttribute('data-theme');
    if (set === 'light' || set === 'dark') return set;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function syncThumbArt() {
    var dark = currentTheme() === 'dark';
    var from = dark ? '/light/' : '/dark/';
    var to = dark ? '/dark/' : '/light/';
    document.querySelectorAll('.project-thumb img, .map-tile img, .article-thumb img, .story-media img').forEach(function (img) {
      // srcset too: the project cards list a 600px and a 1200px copy
      ['src', 'srcset'].forEach(function (attr) {
        var val = img.getAttribute(attr) || '';
        if (val.indexOf(from) !== -1) img.setAttribute(attr, val.split(from).join(to));
      });
    });
  }

  syncThumbArt();

  // follow the OS when the visitor has expressed no preference of their own
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
    if (!document.documentElement.getAttribute('data-theme')) syncThumbArt();
  });

  // --- Landing pulse on the card jumped to from the index map ------
  function pulseCard(card) {
    // clear it anywhere first: jumping between cards would otherwise leave the previous one lit
    document.querySelectorAll('.project.is-landed').forEach(function (n) { n.classList.remove('is-landed'); });
    void card.offsetWidth; // restart the animation when the same card is jumped to twice
    card.classList.add('is-landed');
    clearTimeout(card._landTimer);
    // one 0.6s pulse; the timer also ends the still ring shown under reduced motion
    card._landTimer = setTimeout(function () { card.classList.remove('is-landed'); }, 700);
  }

  // Smooth-scroll duration depends on how far the jump is, so a fixed delay
  // either fires early on long jumps or dawdles on short ones. Wait for the
  // scroll position to actually stop moving instead.
  // Polled on a timer rather than requestAnimationFrame: rAF is suspended while
  // the document is hidden, which would strand the pulse on a background tab.
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

  function pulseHashTarget() {
    // getElementById, not querySelector: a fragment like #1 is not a valid selector,
    // and the throw would stop this whole script (lightbox, glow, buttons never wire).
    var id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (err) { return; } // malformed %-escape
    var el = id && document.getElementById(id);
    if (!el || !el.classList.contains('project')) return;
    afterScrollSettles(function () { pulseCard(el); });
  }

  window.addEventListener('hashchange', pulseHashTarget);
  pulseHashTarget();

  // --- Thumbnail lightbox -----------------------------------------
  // Native <dialog> supplies the focus trap, Esc-to-close, and backdrop, so
  // there's nothing here but wiring. Placeholders hold no <img>, so they're
  // skipped automatically.
  // NOT .map-tile: those are navigation into the card (and fire the pulse).
  var zoomThumbs = document.querySelectorAll('.project-thumb img, .story-media:not(a) img'); // a linked story-media navigates instead
  if (zoomThumbs.length) {
    var lightbox = document.createElement('dialog');
    lightbox.className = 'lightbox';
    lightbox.innerHTML = '<img alt="" /><button class="lightbox-close" type="button" aria-label="Close">'
      + '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6.5 6.5 11 11M17.5 6.5l-11 11" /></svg></button>';
    document.body.appendChild(lightbox);
    var lightboxImg = lightbox.querySelector('img');

    zoomThumbs.forEach(function (img) {
      var holder = img.closest('.project-thumb, .story-media');
      holder.classList.add('is-zoomable');
      holder.setAttribute('role', 'button');
      holder.setAttribute('tabindex', '0');
      // name the card, or every thumbnail on the page is announced identically
      var card = holder.closest('.project, .story');
      var heading = card && card.querySelector('h3, h2');
      holder.setAttribute('aria-label', 'View larger image' + (heading ? ': ' + heading.textContent.trim() : ''));

      var openLightbox = function () {
        if (!img.isConnected) return; // the image failed and was replaced by a placeholder
        lightboxImg.src = img.src; // src is the full-size copy; currentSrc may be the 600px one srcset chose
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
      el.addEventListener('mousemove', function (e) {
        var rect = el.getBoundingClientRect();
        el.style.setProperty('--mx', (e.clientX - rect.left) + 'px');
        el.style.setProperty('--my', (e.clientY - rect.top) + 'px');
      });
    });

    // --- Magnetic primary/secondary buttons ------------------------
    var magnetButtons = document.querySelectorAll('.button');
    magnetButtons.forEach(function (btn) {
      btn.addEventListener('mousemove', function (e) {
        var rect = btn.getBoundingClientRect();
        var dx = e.clientX - rect.left - rect.width / 2;
        var dy = e.clientY - rect.top - rect.height / 2;
        // Pull strength ~25% of cursor distance from button center
        btn.style.setProperty('--mag-x', (dx * 0.25).toFixed(1) + 'px');
        btn.style.setProperty('--mag-y', (dy * 0.25).toFixed(1) + 'px');
      });
      btn.addEventListener('mouseleave', function () {
        btn.style.setProperty('--mag-x', '0px');
        btn.style.setProperty('--mag-y', '0px');
      });
    });
  }
})();
