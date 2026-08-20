(() => {
  const stored = localStorage.getItem('utsm-dashboard-mode');
  const mode = stored === 'advanced' ? 'advanced' : 'simple';
  document.documentElement.dataset.mode = mode;

  const current = document.body.dataset.page || 'home';
  const links = [
    ['home', '/', 'Home'], ['live', '/live', 'Live'], ['runs', '/runs', 'Runs'],
    ['import', '/import', 'Import'], ['dyno', '/dyno', 'Dyno']
  ];
  const header = document.createElement('header');
  header.className = 'shell';
  const inner = document.createElement('div');
  inner.className = 'shell-inner';
  const wordmark = document.createElement('a');
  wordmark.className = 'wordmark'; wordmark.href = '/'; wordmark.setAttribute('aria-label', 'UTSM dashboard home');
  const mark = document.createElement('span'); mark.className = 'wordmark-mark'; mark.textContent = 'U';
  const name = document.createElement('span'); name.textContent = 'UTSM Dashboard';
  wordmark.append(mark, name);
  const menu = document.createElement('button');
  menu.className = 'mobile-nav'; menu.type = 'button'; menu.textContent = 'Menu'; menu.setAttribute('aria-expanded', 'false'); menu.setAttribute('aria-label', 'Open navigation');
  const nav = document.createElement('nav'); nav.className = 'global-nav'; nav.setAttribute('aria-label', 'Main navigation');
  links.forEach(([id, href, label]) => {
    const link = document.createElement('a'); link.href = href; link.textContent = label;
    if (id === current) link.setAttribute('aria-current', 'page');
    nav.append(link);
  });
  const tools = document.createElement('div'); tools.className = 'shell-tools';
  const switcher = document.createElement('div'); switcher.className = 'mode-switch'; switcher.setAttribute('role', 'group'); switcher.setAttribute('aria-label', 'Dashboard detail level');
  ['simple', 'advanced'].forEach(value => {
    const button = document.createElement('button'); button.type = 'button'; button.dataset.modeChoice = value;
    button.textContent = value[0].toUpperCase() + value.slice(1); button.setAttribute('aria-pressed', String(value === mode));
    button.addEventListener('click', () => {
      document.documentElement.dataset.mode = value; localStorage.setItem('utsm-dashboard-mode', value);
      switcher.querySelectorAll('button').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      document.dispatchEvent(new CustomEvent('utsm-mode-change', {detail: {mode: value}}));
    });
    switcher.append(button);
  });
  tools.append(switcher); inner.append(wordmark, menu, nav, tools); header.append(inner); document.body.prepend(header);
  menu.addEventListener('click', () => {
    const open = nav.classList.toggle('open'); menu.setAttribute('aria-expanded', String(open)); menu.textContent = open ? 'Close' : 'Menu';
  });
  nav.addEventListener('click', () => { nav.classList.remove('open'); menu.setAttribute('aria-expanded', 'false'); menu.textContent = 'Menu'; });
  window.utsm = {
    formatBytes(value) { if (value < 1024) return `${value} B`; if (value < 1048576) return `${(value / 1024).toFixed(1)} KB`; return `${(value / 1048576).toFixed(1)} MB`; },
    formatDistance(value) { return value == null ? 'Not available' : value >= 1000 ? `${(value / 1000).toFixed(2)} km` : `${Number(value).toFixed(0)} m`; },
    formatDate(value) { if (!value) return 'Date unknown'; const parsed = new Date(`${value}T00:00:00Z`); return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString(undefined, {year:'numeric', month:'short', day:'numeric', timeZone:'UTC'}); },
    node(tag, className, text) { const item = document.createElement(tag); if (className) item.className = className; if (text != null) item.textContent = text; return item; }
  };
})();
