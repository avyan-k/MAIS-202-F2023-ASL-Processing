// Identification for Javascript
const videoElement = document.getElementById('webcam');
const startButton = document.getElementById('startButton');
const snapButton = document.getElementById('snapButton');
const canvas = document.getElementById('canvas');
const capturedImage = document.getElementById('capturedImage');
const stopButton = document.getElementById('stopButton');
const generatedText = document.getElementById('translated')

let stream;

// Start Webcam
startButton.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        startButton.disabled = true;
        stopButton.disabled = false;
        snapButton.disabled = false;
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
});

// Take Picture and download Snapped Image
snapButton.addEventListener('click', function () {
    // Draw the current video frame onto the canvas
    const context = canvas.getContext('2d');
    // var text = document.createTextNode("A");
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
    // Get pixel data from the canvas
    let imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    let pixelArray=imageData.data;

    let pixelDataArray = Array.from(pixelArray);
    fetch('/receive', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({imageData: pixelDataArray})
    })
    .then(response => response.json())
    .then(data => {
        // Display the random letter received from the backend
        generatedText.innerText = data.letter;
    })
    .catch((error) => {
        console.error('Error:', error);
    });

  });

// Display Error if failed
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
        snapButton.disabled = true;
    }
});