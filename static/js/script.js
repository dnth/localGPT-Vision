// Theme toggle and small utilities

function toggleTheme() {
  const el = document.documentElement;
  const current = el.getAttribute('data-theme') || 'light';
  const next = current === 'light' ? 'dark' : 'light';
  el.setAttribute('data-theme', next);
  try {
    localStorage.setItem('theme', next);
  } catch (e) {
    // ignore storage errors
  }
}

// Optional: expose for other scripts if needed
window.toggleTheme = toggleTheme;