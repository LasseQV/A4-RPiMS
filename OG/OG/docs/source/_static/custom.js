document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("a.reference.external[href*='github.com']").forEach(function (el) {
    if (el.textContent.trim() === "[source]") {
      el.textContent = "[GitHub]";
    }
  });
});