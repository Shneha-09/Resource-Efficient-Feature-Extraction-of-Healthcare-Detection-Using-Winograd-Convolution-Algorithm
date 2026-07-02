console.log("✅ script.js is correctly connected!");

document.addEventListener("DOMContentLoaded", () => {
  // Smooth scrolling for internal links
  const links = document.querySelectorAll('a[href^="#"]');
  links.forEach(link => {
    link.addEventListener("click", function (e) {
      const targetId = this.getAttribute("href");
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        e.preventDefault();
        targetElement.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  // Optional: Close any open cards when clicking outside
  document.addEventListener('click', (e) => {
    const openCards = document.querySelectorAll('.card.open');
    openCards.forEach(card => {
      if (!card.contains(e.target)) {
        card.classList.remove('open');
      }
    });
  });
});

// Card toggle function
function toggleCard(card) {
  card.classList.toggle('open');
}
