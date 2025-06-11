let classifier;
let mobilenetModel;
let webcamElement;
let webcamStream;

const EMOJIS = ["👍", "✌️", "✊"];
const MENSAJES = ["Buen trabajo 👍", "Muy bien ✌️", "¡Fuerza! ✊"];

async function init() {
  classifier = knnClassifier.create();
  mobilenetModel = await mobilenet.load();

  webcamElement = document.getElementById("webcam");
  webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcamElement.srcObject = webcamStream;

  await new Promise((resolve) => (webcamElement.onloadedmetadata = resolve));

  predictLoop();
}

async function capture() {
  return tf.tidy(() => {
    const webcamTensor = tf.browser.fromPixels(webcamElement);
    return tf.image
      .resizeBilinear(webcamTensor, [224, 224])
      .toFloat()
      .div(127)
      .sub(1)
      .expandDims();
  });
}

async function addExample(classId) {
  const img = await capture();
  const activation = mobilenetModel.infer(img, true);
  classifier.addExample(activation, classId);
  img.dispose();
}

async function predictLoop() {
  if (classifier.getNumClasses() > 0) {
    const img = await capture();
    const activation = mobilenetModel.infer(img, true);
    const result = await classifier.predictClass(activation);
    img.dispose();

    const emoji = EMOJIS[result.label];
    const mensaje = MENSAJES[result.label];
    const confianza = result.confidences[result.label].toFixed(2);

    mostrarResultado(`${emoji}`, `${mensaje} (Confianza: ${confianza})`);
  }
  setTimeout(predictLoop, 500);
}

function mostrarResultado(emoji, texto) {
  document.getElementById("result").textContent = emoji;
  document.getElementById("message").textContent = texto;
}

window.addEventListener("load", init);
