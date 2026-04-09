document.addEventListener('DOMContentLoaded', () => {
    const newsInput = document.getElementById('newsInput');
    const predictBtn = document.getElementById('predictBtn');
    const resultSection = document.getElementById('resultSection');
    const resultCard = document.getElementById('resultCard');
    const predictionTitle = document.getElementById('predictionTitle');
    const resultMessage = document.getElementById('resultMessage');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const loader = document.getElementById('loader');
    const btnText = document.querySelector('.btn-text');

    predictBtn.addEventListener('click', async () => {
        const text = newsInput.value.trim();
        
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }

        // UI Loading State
        predictBtn.disabled = true;
        loader.style.display = 'block';
        btnText.style.opacity = '0';
        resultSection.classList.add('hidden');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Update UI with results
            displayResult(data.prediction, data.confidence);
        } catch (error) {
            console.error('Error:', error);
            alert('Analysis failed: ' + error.message);
        } finally {
            // Restore UI State
            predictBtn.disabled = false;
            loader.style.display = 'none';
            btnText.style.opacity = '1';
        }
    });

    function displayResult(prediction, confidence) {
        resultSection.classList.remove('hidden');
        
        // Reset classes
        resultCard.classList.remove('news-fake', 'news-real');
        
        const isFake = prediction.toLowerCase() === 'fake';
        
        if (isFake) {
            resultCard.classList.add('news-fake');
            predictionTitle.textContent = '🚨 FAKE NEWS DETECTED';
            resultMessage.textContent = 'Our AI analysis indicates a high probability that this content is fabricated or misleading.';
        } else {
            resultCard.classList.add('news-real');
            predictionTitle.textContent = '✅ REAL NEWS';
            resultMessage.textContent = 'The patterns and source characteristics analyzed suggest this information is likely authentic.';
        }

        // Animate values
        animateValue(confidenceValue, 0, Math.round(confidence), 1000);
        
        // Animate progress bar
        setTimeout(() => {
            confidenceBar.style.width = confidence + '%';
        }, 100);
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
