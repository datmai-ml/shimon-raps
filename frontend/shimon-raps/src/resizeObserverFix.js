// Only suppress the error overlay, don't modify ResizeObserver
const error = console.error;
console.error = (...args) => {
  if (args[0]?.includes?.('ResizeObserver')) return;
  error(...args);
};

window.addEventListener('error', (e) => {
  if (e.message?.includes('ResizeObserver')) {
    e.stopImmediatePropagation();
    e.stopPropagation();
    e.preventDefault();
  }
}, true);

window.addEventListener('unhandledrejection', (e) => {
  if (e.reason?.message?.includes('ResizeObserver')) {
    e.stopImmediatePropagation();
  }
}, true);