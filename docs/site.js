const navigation = [...document.querySelectorAll('nav a[href^="#"]')];
const destinations = navigation.map(link => document.getElementById(link.hash.slice(1)));
const masthead = document.querySelector('.masthead');

function updateNavigation() {
  const scrollOffset = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop);
  const boundary = Math.max(scrollOffset + 4, masthead.getBoundingClientRect().bottom + 24);
  let active = -1;

  destinations.forEach((section, index) => {
    if (section.getBoundingClientRect().top <= boundary) active = index;
  });

  navigation.forEach((link, index) => {
    if (index === active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}

let pending = false;
function scheduleNavigationUpdate() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    updateNavigation();
    pending = false;
  });
}

window.addEventListener('scroll', scheduleNavigationUpdate, { passive: true });
window.addEventListener('resize', scheduleNavigationUpdate);
window.addEventListener('hashchange', scheduleNavigationUpdate);
updateNavigation();
