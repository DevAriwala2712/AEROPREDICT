(() => {
  const STORAGE_KEY = "aeropredict-theme";

  function getPreferredTheme() {
    const savedTheme = localStorage.getItem(STORAGE_KEY);
    if (savedTheme === "light" || savedTheme === "dark") {
      return savedTheme;
    }

    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }

  function updateToggleLabels(theme) {
    document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
      const label = button.querySelector("[data-theme-toggle-label]");
      const icon = button.querySelector("[data-theme-toggle-icon]");
      if (label) {
        label.textContent = theme === "light" ? "Light Mode" : "Dark Mode";
      }
      if (icon) {
        icon.textContent = theme === "light" ? "dark_mode" : "light_mode";
      }
      button.setAttribute("aria-pressed", String(theme === "light"));
    });
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem(STORAGE_KEY, theme);
    updateToggleLabels(theme);
    window.dispatchEvent(new CustomEvent("aeropredict:theme", { detail: { theme } }));
  }

  function toggleTheme() {
    applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
  }

  function ensureToggle() {
    if (document.querySelector("[data-theme-toggle]")) {
      return;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "theme-toggle-fab";
    button.setAttribute("data-theme-toggle", "true");
    button.setAttribute("aria-label", "Toggle light mode");
    button.innerHTML = '<span class="material-symbols-outlined text-base" data-theme-toggle-icon>light_mode</span><span class="theme-toggle-label" data-theme-toggle-label>Light Mode</span>';
    button.addEventListener("click", toggleTheme);
    document.body.appendChild(button);
  }

  function initTheme() {
    applyTheme(getPreferredTheme());
    document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
      button.addEventListener("click", toggleTheme);
    });
    ensureToggle();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTheme, { once: true });
  } else {
    initTheme();
  }

  window.AeroPredictTheme = { applyTheme, toggleTheme };
})();