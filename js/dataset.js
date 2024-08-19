import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs';

const datasetPath = '../dataset';

/**
 * Reads a file from the dataset directory.
 * @param {string} filename - The name of the file to read.
 * @returns {Buffer} - The contents of the file as a Buffer.
 */
function readFile(filename) {
  return fs.readFileSync(path.join(datasetPath, filename));
}

/**
 * Parses the image data from the MNIST dataset file.
 * @param {Buffer} data - The buffer containing the image data.
 * @returns {number[][]} - An array of images, where each image is represented as a 2D array of pixel values.
 */
function parseImages(data) {
  const magic = data.readUInt32BE(0);
  const numImages = data.readUInt32BE(4);
  const numRows = data.readUInt32BE(8);
  const numCols = data.readUInt32BE(12);

  const images = [];
  let offset = 16;

  for (let i = 0; i < numImages; i++) {
    const image = [];
    for (let r = 0; r < numRows; r++) {
      const row = [];
      for (let c = 0; c < numCols; c++) {
        row.push(data.readUInt8(offset++));
      }
      image.push(row);
    }
    images.push(image);
  }

  return images;
}

/**
 * Parses the label data from the MNIST dataset file.
 * @param {Buffer} data - The buffer containing the label data.
 * @returns {number[]} - An array of labels.
 */
function parseLabels(data) {
  const magic = data.readUInt32BE(0);
  const numLabels = data.readUInt32BE(4);

  const labels = [];
  let offset = 8;

  for (let i = 0; i < numLabels; i++) {
    labels.push(data.readUInt8(offset++));
  }

  return labels;
}

// Load training images
const trainImagesData = readFile('train-images.idx3-ubyte');
const trainImages = parseImages(trainImagesData);

// Load training labels
const trainLabelsData = readFile('train-labels.idx1-ubyte');
const trainLabels = parseLabels(trainLabelsData);

// Load test images
const testImagesData = readFile('t10k-images.idx3-ubyte');
const testImages = parseImages(testImagesData);

// Load test labels
const testLabelsData = readFile('t10k-labels.idx1-ubyte');
const testLabels = parseLabels(testLabelsData);

console.log('Training images:', trainImages.length);
console.log('Training labels:', trainLabels.length);
console.log('Test images:', testImages.length);
console.log('Test labels:', testLabels.length);

// Convert arrays to tensors
trainImages = tf.tensor(trainImages);
testImages  = tf.tensor(testImages);
trainLabels = tf.tensor(trainLabels);
testLabels  = tf.tensor(testLabels);

// Export the loaded dataset
export { trainImages, trainLabels, testImages, testLabels };