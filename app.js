const webcam = document.getElementById('wc');
const nameUser = document.getElementById('name');

var users = {}

/**
   * Compute the euclidean distance between two tensors.
   * @param {TensorxD} tensor1 An input Tensor.
   * @param {TensorxD} tensor2 An input Tensor.
   */

function euclidean_distance(tensor1,tensor2){
  return tf.sqrt(tf.sum(tf.square(tf.sub(tensor1,tensor2))))
}

/**
   * Using resnet_v2_50 as backbone.
   * @param {Tensor4D} img An input img Tensor.
   */
async function encoding(img) {
  const resnet = await tf.loadGraphModel("https://tfhub.dev/google/tfjs-model/imagenet/resnet_v2_50/feature_vector/3/default/1", { fromTFHub: true }) 
  return resnet.execute(img)
}

/**
   * Launch the webcam.
   * @param {HTMLElement} webcam refrence of the webcam.
   */

async function launchWebcam(webcam) {
  return new Promise((resolve, reject) => {
    navigator.getUserMedia = navigator.getUserMedia ||
        navigator.webkitGetUserMedia || navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia(
          {video: {width: 224, height: 224}},
          stream => {
            webcam.srcObject = stream;
            webcam.addEventListener('loadeddata', async () => {
              resolve();
            }, false);
          },
          error => {
            reject(error);
          });
    } else {
      reject();
    }
  });
}

/**
   * Transforming an image input to a tensor.
   * @param {HTMLElement} webcam refrence of the webcam.
   */

function imageToTensor(webcam) {
  return tf.tidy(() => {
    const webcamImage = tf.browser.fromPixels(webcam);
    const batchedImage = webcamImage.expandDims(0);
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
  });
}


function startCapturing(webcam) {

  if (nameUser.value===''){alert("There is no username entered");}
  else{
    const img = imageToTensor(webcam);
    const result = encoding(img);
    users[nameUser.value] = result;
    
  }
}


launchWebcam(webcam)