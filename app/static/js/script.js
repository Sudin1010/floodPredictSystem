// Flood Prediction System - JavaScript
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-password-toggle]').forEach((button) => {
        const showLabel = button.getAttribute('aria-label') || 'Show password';
        const hideLabel = showLabel.replace('Show', 'Hide');

        button.addEventListener('click', () => {
            const targetId = button.getAttribute('data-target');
            const input = document.getElementById(targetId);

            if (!input) {
                return;
            }

            const isHidden = input.type === 'password';
            input.type = isHidden ? 'text' : 'password';
            button.setAttribute('aria-label', isHidden ? hideLabel : showLabel);
            button.classList.toggle('is-visible', isHidden);

            const hiddenLabel = button.querySelector('.visually-hidden');
            if (hiddenLabel) {
                hiddenLabel.textContent = isHidden ? hideLabel : showLabel;
            }
        });
    });
});
