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

    const setValidationMessage = (input, message) => {
        const messageElement = document.querySelector(`[data-message-for="${input.id}"]`);
        if (messageElement) {
            messageElement.textContent = message;
        }
    };

    const clearField = (input) => {
        input.classList.remove('is-invalid', 'is-valid');
        setValidationMessage(input, '');
    };

    const markField = (input, isValid, message = '') => {
        input.classList.toggle('is-invalid', !isValid);
        input.classList.toggle('is-valid', isValid && input.value.trim() !== '');
        setValidationMessage(input, message);
    };

    const validateRequiredField = (input) => {
        if (input.value.trim() === '') {
            markField(input, false, input.dataset.authRequired || 'This field is required.');
            return false;
        }

        markField(input, true);
        return true;
    };

    const usernamePattern = /^[a-zA-Z0-9_]{3,20}$/;
    const emailPattern = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
    const passwordPattern = /^(?=.*[A-Za-z])(?=.*\d).{8,}$/;

    const validateField = (form, input) => {
        if (!validateRequiredField(input)) {
            return false;
        }

        const value = input.value.trim();

        if (input.matches('[data-auth-username]') && !usernamePattern.test(value)) {
            markField(input, false, input.dataset.authUsername);
            return false;
        }

        if (input.matches('[data-auth-username-or-email]') && !usernamePattern.test(value) && !emailPattern.test(value)) {
            markField(input, false, input.dataset.authUsernameOrEmail);
            return false;
        }

        if (input.matches('[data-auth-email]') && !emailPattern.test(value)) {
            markField(input, false, input.dataset.authEmail || 'Please enter a valid email address.');
            return false;
        }

        if (input.matches('[data-auth-password]') && !passwordPattern.test(input.value)) {
            markField(input, false, input.dataset.authPassword);
            return false;
        }

        if (form.dataset.authForm === 'register' && input.id === 'confirm_password') {
            const password = form.querySelector('#password');
            if (password && input.value !== password.value) {
                markField(input, false, 'Passwords do not match.');
                return false;
            }
        }

        markField(input, true);
        return true;
    };

    const validateAuthForm = (form) => {
        let isValid = true;

        form.querySelectorAll('[data-auth-required]').forEach((input) => {
            if (!validateField(form, input)) {
                isValid = false;
            }
        });

        return isValid;
    };

    const setAuthLoadingState = (form) => {
        const submitButton = form.querySelector('button[type="submit"]');

        if (!submitButton) {
            return;
        }

        const loadingText = form.dataset.authForm === 'register' ? 'Creating Account...' : 'Logging in...';

        if (!submitButton.dataset.originalText) {
            submitButton.dataset.originalText = submitButton.textContent.trim();
        }

        submitButton.disabled = true;
        submitButton.setAttribute('aria-busy', 'true');
        submitButton.classList.add('auth-btn-loading');
        submitButton.innerHTML = `<span class="auth-btn-spinner" aria-hidden="true"></span><span>${loadingText}</span>`;
    };

    document.querySelectorAll('[data-auth-form]').forEach((form) => {
        form.querySelectorAll('[data-auth-required]').forEach((input) => {
            input.addEventListener('input', () => {
                clearField(input);
            });
        });

        form.addEventListener('submit', (event) => {
            if (form.dataset.submitting === 'true') {
                event.preventDefault();
                return;
            }

            if (!validateAuthForm(form)) {
                event.preventDefault();
                return;
            }

            form.dataset.submitting = 'true';
            setAuthLoadingState(form);
        });
    });
});
