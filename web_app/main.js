// Identification for Javascript
const videoElement = document.getElementById('webcam');
const startButton = document.getElementById('startButton');
const snapButton = document.getElementById('snapButton');
const canvas = document.getElementById('canvas');
const capturedImage = document.getElementById('capturedImage');
const stopButton = document.getElementById('stopButton');

let stream;

// Start Webcam
startButton.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        startButton.disabled = true;
        stopButton.disabled = false;
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
});

// Take Picture and display Snapped Image
snapButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    capturedImage.src = canvas.toDataURL('image/png');
    capturedImage.style.display = 'block';
});

// Display Error if failed (Generated from Chat GPT)
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
}

// Stop Webcam
stopButton.addEventListener('click', () => {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
        startButton.disabled = false;
        stopButton.disabled = true;
    }
});