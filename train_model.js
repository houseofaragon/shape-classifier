/*
    This file allows you to train your own model using the ml5.js library.
    
    This example uses 64x64 pixel images of circles, squares, and triangles.
    generated in p5js.

    1. We initialize a convolutional neural network
    and add the images to the model as training data.
    2. The data is normalized to values between 0 and 1.
    3. And then we train the model for 50 epochs
    4. Once the model is trained, we save the model.

*/
let shapeClassifier;
let canvas;
let inputImage;
let clearButton;
let guess;
let confidence;

let circles = [];
let squares = [];
let triangles = [];

// makes sure the images are 128 x 128 pixels
function preload() {
  for (let i = 0; i < 50; i++) {
    circles[i] = loadImage(`data/data_circle-${i}.png`);
    squares[i] = loadImage(`data/data_square-${i}.png`);
    triangles[i] = loadImage(`data/data_triangle-${i}.png`);
  }
}

function setup() {
  canvas = createCanvas(64, 64);
  // background(255);
  // image(circles[10], 0, 0, width, height);

  let options = {
    // [width, height, numClasses = channels (RGBA)]
    inputs: [64, 64, 4],
    // Convolutional Neural Network
    task: "imageClassification", 
    // triggers visualization of the loss function
    debug: true
  }

  // https://www.youtube.com/watch?v=hWurN0XhzLY&t=0s
  shapeClassifier = ml5.neuralNetwork(options);

  // Add data
  for (let i = 0; i < circles.length; i++) {
    shapeClassifier.addData({ image: circles[i] }, { label: "circle" });
    shapeClassifier.addData({ image: triangles[i] }, { label: "triangle" });
    shapeClassifier.addData({ image: squares[i] }, { label: "square" });
  }

  // normalize data - analyses data by looking at the minimum
  // and maximum ranges of pixel values and then
  // normalize all the inputs to numbers between 0 and 1
  // https://www.youtube.com/watch?v=UaKab6h9Z0I&t=0s
  shapeClassifier.normalizeData();

  // send all images through the neural network 50 times
  shapeClassifier.train({ epochs: 50 }, finishedTraining);
}

function finishedTraining() {
  console.log("finished training");
  shapeClassifier.save();
}