document.addEventListener('DOMContentLoaded', function() {
    // Toggle spouse section based on marital status
    const maritalStatus = document.getElementById('maritalStatus');
    const spouseSection = document.getElementById('spouseSection');
    const spouseQualifications = document.getElementById('spouseQualifications');
    
    maritalStatus.addEventListener('change', function() {
        if (this.value === 'spouse') {
            spouseSection.style.display = 'block';
        } else {
            spouseSection.style.display = 'none';
            spouseQualifications.style.display = 'none';
        }
    });
    
    // Toggle spouse qualifications based on accompanying status
    const spouseAccompanying = document.querySelector('[name="spouse_accompanying"]');
    if (spouseAccompanying) {
        spouseAccompanying.addEventListener('change', function() {
            if (maritalStatus.value === 'spouse' && this.value === 'true') {
                spouseQualifications.style.display = 'block';
            } else {
                spouseQualifications.style.display = 'none';
            }
        });
    }
    
    // Toggle second language section
    const secondLangToggle = document.getElementById('secondLangToggle');
    const secondLangSection = document.getElementById('secondLangSection');
    const secondLangInputs = document.querySelectorAll('#secondLangSection input[type="number"]');
    
    function handleSecondLanguageToggle() {
        if (secondLangToggle.value === 'yes') {
            secondLangSection.style.display = 'block';
            secondLangInputs.forEach(input => {
                input.disabled = false;
                if (input.value === '0') input.value = '5';
            });
        } else {
            secondLangSection.style.display = 'none';
            secondLangInputs.forEach(input => {
                input.disabled = true;
                input.value = '0';
            });
        }
    }
    
    secondLangToggle.addEventListener('change', handleSecondLanguageToggle);
    
    // Initialize form state
    handleSecondLanguageToggle();
    
    // Form validation
    const form = document.getElementById('crsForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            // Validate age
            const ageInput = document.querySelector('[name="age"]');
            if (ageInput && (ageInput.value < 17 || ageInput.value > 100)) {
                alert('Please enter a valid age between 17 and 100');
                e.preventDefault();
                return false;
            }
            
            // Validate first language scores
            const firstLangInputs = document.querySelectorAll('input[name^="first_lang_"]');
            for (const input of firstLangInputs) {
                if (!input.value || input.value < 1 || input.value > 12) {
                    alert('Please enter valid CLB levels (1-12) for first language');
                    e.preventDefault();
                    return false;
                }
            }
            
            // Validate second language scores if enabled
            if (secondLangToggle.value === 'yes') {
                for (const input of secondLangInputs) {
                    if (!input.value || input.value < 1 || input.value > 12) {
                        alert('Please enter valid CLB levels (1-12) for second language');
                        e.preventDefault();
                        return false;
                    }
                }
            }
            
            return true;
        });
    }
});