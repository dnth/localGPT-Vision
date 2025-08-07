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
// Image modal functionality
function setupImageModal() {
    const images = document.querySelectorAll('.img-thumbnail');
    images.forEach(img => {
        img.addEventListener('click', function() {
            // Create modal
            const modal = document.createElement('div');
            modal.style.position = 'fixed';
            modal.style.top = '0';
            modal.style.left = '0';
            modal.style.width = '100%';
            modal.style.height = '100%';
            modal.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
            modal.style.display = 'flex';
            modal.style.alignItems = 'center';
            modal.style.justifyContent = 'center';
            modal.style.zIndex = '1000';
            modal.style.cursor = 'zoom-out';
            
            // Create modal image
            const modalImg = document.createElement('img');
            modalImg.src = this.src;
            modalImg.style.maxWidth = '90%';
            modalImg.style.maxHeight = '90%';
            modalImg.style.objectFit = 'contain';
            
            // Close modal on click
            modal.addEventListener('click', function() {
                document.body.removeChild(modal);
            });
            
            // Add image to modal
            modal.appendChild(modalImg);
            
            // Add modal to document
            document.body.appendChild(modal);
        });
    });
}

// Initialize image modal when page loads
document.addEventListener('DOMContentLoaded', setupImageModal);
window.toggleTheme = toggleTheme;