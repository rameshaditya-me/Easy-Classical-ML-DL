(function (global) {
  const REPO_ROOT_PREFIX = /^src\/pages\//;

  function syncHighlightTheme(theme) {
    const light = document.getElementById("hljs-light");
    const dark = document.getElementById("hljs-dark");
    if (!light || !dark) return;
    if (theme === "dark") {
      dark.media = "all";
      light.media = "none";
    } else {
      light.media = "all";
      dark.media = "none";
    }
  }

  function initTheme() {
    const toggle = document.getElementById("theme-toggle");
    if (!toggle) return;

    const root = document.documentElement;

    function setTheme(theme) {
      if (theme === "dark") {
        root.dataset.theme = "dark";
        toggle.setAttribute("aria-label", "Switch to light mode");
      } else {
        delete root.dataset.theme;
        toggle.setAttribute("aria-label", "Switch to dark mode");
      }
      syncHighlightTheme(theme);
      localStorage.setItem("theme", theme);
    }

    syncHighlightTheme(root.dataset.theme === "dark" ? "dark" : "light");

    toggle.addEventListener("click", () => {
      setTheme(root.dataset.theme === "dark" ? "light" : "dark");
    });

    toggle.setAttribute(
      "aria-label",
      root.dataset.theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
    );
  }

  function normalizePagePath(path) {
    const cleaned = path.replace(/^\//, "").replace(REPO_ROOT_PREFIX, "pages/");
    if (!cleaned.startsWith("pages/") && !cleaned.startsWith("index.md")) {
      return cleaned.startsWith("pages/") ? cleaned : `pages/${cleaned}`;
    }
    return cleaned;
  }

  function pageUrl(mdPath) {
    const src = normalizePagePath(mdPath);
    return `page.html?src=${encodeURIComponent(src)}`;
  }

  function resolveAssetPath(basePath, assetPath) {
    if (/^(https?:|\/|#|mailto:)/.test(assetPath)) return assetPath;
    const baseDir = basePath.includes("/") ? basePath.slice(0, basePath.lastIndexOf("/")) : "";
    const parts = `${baseDir}/${assetPath.replace(/^\.\//, "")}`.split("/");
    const resolved = [];
    for (const part of parts) {
      if (!part || part === ".") continue;
      if (part === "..") resolved.pop();
      else resolved.push(part);
    }
    return resolved.join("/");
  }

  function configureMarked() {
    marked.setOptions({ gfm: true, breaks: false });
    if (typeof hljs === "undefined") return;

    marked.use({
      renderer: {
        code({ text, lang }) {
          const language = lang && hljs.getLanguage(lang) ? lang : "plaintext";
          const highlighted = hljs.highlight(text, { language }).value;
          return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`;
        },
      },
    });
  }

  function enhanceContent(container, basePath) {
    container.querySelectorAll("a[href]").forEach((link) => {
      const href = link.getAttribute("href");
      if (!href || href.startsWith("#")) return;

      if (/^https?:\/\//.test(href)) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        return;
      }

      if (href.endsWith(".md")) {
        link.href = pageUrl(resolveAssetPath(basePath, href));
        return;
      }

      if (!/^https?:\/\//.test(href)) {
        link.href = resolveAssetPath(basePath, href);
      }
    });

    container.querySelectorAll("img[src]").forEach((img) => {
      const src = img.getAttribute("src");
      if (src && !/^(https?:|\/)/.test(src)) {
        img.src = resolveAssetPath(basePath, src);
        img.loading = "lazy";
      }
    });
  }

  function renderMarkdown(md, container, basePath) {
    container.innerHTML = marked.parse(md);
    enhanceContent(container, basePath);
    const title = container.querySelector("h1");
    if (title) document.title = `${title.textContent} · Notebooks on Learning`;
  }

  function loadMarkdown(path, container) {
    const src = normalizePagePath(path);
    container.innerHTML = `<p class="content-loading">Loading…</p>`;

    return fetch(src)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load ${src} (${res.status})`);
        return res.text();
      })
      .then((md) => {
        renderMarkdown(md, container, src);
      })
      .catch((err) => {
        container.innerHTML = `<p class="content-error">Could not load content: ${err.message}</p>`;
      });
  }

  function bootIndex() {
    const content = document.getElementById("content");
    if (!content) return;
    loadMarkdown("index.md", content);
  }

  function bootPage() {
    const content = document.getElementById("content");
    if (!content) return;
    const params = new URLSearchParams(window.location.search);
    const src = params.get("src");
    if (!src) {
      content.innerHTML = `<p class="content-error">No page specified.</p>`;
      return;
    }
    loadMarkdown(src, content);
  }

  global.Site = {
    initTheme,
    configureMarked,
    bootIndex,
    bootPage,
    loadMarkdown,
    renderMarkdown,
    pageUrl,
  };
})(window);
