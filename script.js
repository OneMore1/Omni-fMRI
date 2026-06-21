const navLinks = Array.from(document.querySelectorAll(".site-nav a"));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

document.querySelectorAll('[aria-disabled="true"]').forEach((link) => {
  link.addEventListener("click", (event) => event.preventDefault());
});

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

    if (!visible) return;

    navLinks.forEach((link) => {
      link.classList.toggle(
        "is-active",
        link.getAttribute("href") === `#${visible.target.id}`
      );
    });
  },
  { rootMargin: "-25% 0px -55% 0px", threshold: [0.15, 0.35, 0.55] }
);

sections.forEach((section) => observer.observe(section));

const lightbox = document.querySelector(".lightbox");
const lightboxImage = lightbox.querySelector("img");
const lightboxCaption = lightbox.querySelector("figcaption");
const closeButton = document.querySelector(".lightbox-close");

function openLightbox(button) {
  lightboxImage.src = button.dataset.image;
  lightboxImage.alt = button.querySelector("img")?.alt || button.dataset.title;
  lightboxCaption.textContent = button.dataset.title;
  lightbox.classList.add("is-open");
  lightbox.setAttribute("aria-hidden", "false");
  closeButton.focus();
}

function closeLightbox() {
  lightbox.classList.remove("is-open");
  lightbox.setAttribute("aria-hidden", "true");
  lightboxImage.src = "";
}

document
  .querySelectorAll("[data-image]")
  .forEach((button) => button.addEventListener("click", () => openLightbox(button)));

closeButton.addEventListener("click", closeLightbox);

lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && lightbox.classList.contains("is-open")) {
    closeLightbox();
  }
});
