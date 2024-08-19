import { trainImages, trainLabels, testImages, testLabels } from './dataset.js';
import * as tf from '@tensorflow/tfjs-node';
import { performance } from 'perf_hooks';

console.log(`TensorFlow.js: ${tf.version.tfjs}`);

// ================================
// ======== BENCHMARKING ==========
// ================================

/**
 * Decorator function to measure the execution time of a given function.
 * Prints the function name and the time taken for execution in milliseconds.
 * @param {Function} func - The function to benchmark.
 * @param {...*} args - The arguments to pass to the function.
 * @returns {*} - The result of the function.
 */
function benchmark(func, ...args) {
  console.log(`Starting ${func.name}...`);
  const startTime = performance.now();
  const result = func(...args);
  const finish = () => {
    const endTime = performance.now();
    console.log(`${func.name} completed in ${(endTime - startTime).toFixed(2)} milliseconds.`);
  };
  if (result.then) {
    return result.then(output => {
      finish();
      return output;
    });
  }
  finish();
  return result;
}

// ================================
// ======== PREPROCESSING =========
// ================================

/**
 * Utility function to check the range of values across the entire dataset.
 * Prints the minimum and maximum values of the images.
 * @param {tf.Tensor} images - The tensor containing the images.
 * @param {string} text - The text to display along with the normalization values.
 */
function normalizePeek(images, text = "?") {
  let minSync = images.min().dataSync();
  let maxSync = images.max().dataSync();
  console.log(`${text} normalization min:${minSync[0]} / max:${maxSync[0]}`);
}

/**
 * Preprocesses the training and test data.
 * Normalizes the pixel values and reshapes the images.
 * @param {tf.Tensor} trainImages - The tensor containing the training images.
 * @param {tf.Tensor} testImages - The tensor containing the test images.
 * @param {tf.Tensor} trainLabels - The tensor containing the training labels.
 * @param {tf.Tensor} testLabels - The tensor containing the test labels.
 * @returns {[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]} - An array containing the preprocessed data.
 */
function preprocessData(trainImages, testImages, trainLabels, testLabels) {
  // Normalize pixel values
  normalizePeek(trainImages, "Train Before");
  trainImages = trainImages.div(255);
  normalizePeek(trainImages, "Train After");

  normalizePeek(testImages, "Test Before");
  testImages = testImages.div(255);
  normalizePeek(testImages, "Test After");

  // Reshape the images
  trainImages = trainImages.reshape([-1, 28, 28, 1]);
  testImages = testImages.reshape([-1, 28, 28, 1]);

  return [trainImages, testImages, trainLabels, testLabels];
}

// ================================
// ======== TRAINING ==============
// ================================

/**
 * Creates the CNN model architecture.
 * @returns {tf.Sequential} - The created model.
 */
function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    filters: 30,
    kernelSize: [3, 3],
    activation: 'relu',
    inputShape: [28, 28, 1]
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

/**
 * Main function that orchestrates the entire flow of the application.
 * Preprocesses the data, creates the model, trains the model, and evaluates its performance.
 * @returns {Promise<{training: {images: tf.Tensor, labels: tf.Tensor}, testing: {images: tf.Tensor, labels: tf.Tensor}, model: tf.Sequential}>} - An object containing the preprocessed data and the trained model.
 */
async function app() {
  // Preprocess the data
  const [
    preprocessedTrainImages,
    preprocessedTestImages,
    preprocessedTrainLabels,
    preprocessedTestLabels
  ] = benchmark(preprocessData, trainImages, testImages, trainLabels, testLabels);

  // Create the model
  const model = benchmark(createModel);

  let startTime, batchStartTime;

  // Train the model
  await benchmark(
    model.fit.bind(model),
    preprocessedTrainImages,
    preprocessedTrainLabels,
    {
      epochs: 5,
      batchSize: 32,
      shuffle: true,
      stepsPerEpoch: Math.floor(preprocessedTrainImages.shape[0] / 32),
      validationSteps: Math.floor(preprocessedTestImages.shape[0] / 32),

      callbacks: {
        onTrainBegin: (logs) => {
          console.log("Starting training...");
        },
        onBatchEnd: (batch, logs) => {
          const batchNumber = batch + 1;
          const totalBatches = Math.floor(preprocessedTrainImages.shape[0] / 32);
          const batchEndTime = Date.now();
          const elapsedTime = batchEndTime - startTime;
          const avgBatchTime = elapsedTime / batchNumber;
          const numberOfBatchesRemaining = totalBatches - batchNumber;
          const timeRemaining = numberOfBatchesRemaining ? readableTime(numberOfBatchesRemaining * avgBatchTime)
            : readableTime(elapsedTime);
          const progressBar = getProgressBar(batchNumber, totalBatches);

          process.stdout.clearLine();
          process.stdout.cursorTo(0);
          process.stdout.write(
            `${String(batchNumber).padStart(totalBatches.toString().length, ' ')}/${totalBatches} ${progressBar} ${timeRemaining} ${Math.round(avgBatchTime)}ms/step - accuracy: ${logs.acc.toFixed(4)} - loss: ${logs.loss.toFixed(4)}`
          );
        },
        onEpochBegin: (epoch, logs) => {
          console.log(); // Move to the next line after an epoch ends
          console.log(`Epoch ${epoch + 1}/5`);
          startTime = Date.now();
        },
        onTrainEnd: (logs) => {
          console.log(); // Move to the next line after training ends
          console.log("Training completed!");
          console.log(`Final Training loss: ${logs.loss.toFixed(4)}, accuracy: ${logs.acc.toFixed(4)}`);
        }
      }
    }
  );

  // Evaluate the model on test data
  const evaluateModel = async () => {
    const [testLoss, testAccuracy] = await model.evaluate(preprocessedTestImages, preprocessedTestLabels);
    const loss = testLoss.dataSync();
    const accuracy = testAccuracy.dataSync();
    console.log("Test Loss:", loss[0]);
    console.log("Test Accuracy:", accuracy[0]);
  };

  await benchmark(evaluateModel);

  return {
    training: {
      images: preprocessedTrainImages,
      labels: preprocessedTrainLabels
    },
    testing: {
      images: preprocessedTestImages,
      labels: preprocessedTestLabels
    },
    model
  };
}

const preprocessed = await benchmark(app);

// ================================
// ======== TESTING ===============
// ================================

/**
 * Tests the trained model on a specific image from the test set.
 * Prints the predicted label and the true label for the image.
 * @param {Object} preprocessed - The preprocessed data and the trained model.
 * @param {number} index - The index of the image to test.
 * @returns {Promise<void>}
 */
async function testImage(preprocessed, index) {
  const testImages = preprocessed.testing.images;
  const testLabels = preprocessed.testing.labels;
  const { model } = preprocessed;

  // Get the image at the selected index
  const testImage = testImages.slice([index], [1]);

  console.log("Making predictions on the selected image...");
  const predictions = await model.predict(testImage);
  console.log(predictions);

  // Get the predicted label
  const predictedLabel = predictions.argMax(-1).dataSync()[0];

  // Get the true label
  const trueLabel = testLabels.slice([index], [1]).dataSync()[0];

  console.log(`Predicted Label: ${predictedLabel}, Image Label: ${trueLabel}, INDEX: ${index}`);
}

// Select an index of the image to test
const testImageIndex = 1869;

// Test the image at the selected index
benchmark(testImage, preprocessed, testImageIndex);

// ================================
// ======== UTILITY ===============
// ================================

/**
 * Converts milliseconds to a readable time format (hours, minutes, seconds).
 * @param {number} ms - The time in milliseconds.
 * @returns {string} - The formatted time string.
 */
function readableTime(ms) {
  // Convert milliseconds to total seconds
  const totalSeconds = Math.floor(ms / 1000);

  // Calculate hours, minutes, and seconds
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  // Format the time as a readable string
  const hoursStr = hours > 0 ? `${hours}h` : "";
  const minutesStr = minutes > 0 ? `${minutes > 9 ? minutes : "0" + minutes}m` : "";
  const secondsStr = `${seconds > 9 ? seconds : "0" + seconds}s`;

  return `${hoursStr}${minutesStr}${secondsStr}`;
}

/**
 * Generates a progress bar string based on the completed and total steps.
 * @param {number} completed - The number of completed steps.
 * @param {number} total - The total number of steps.
 * @returns {string} - The progress bar string.
 */
function getProgressBar(completed, total) {
  const barLength = 25;
  const progress = Math.round((completed / total) * barLength);
  const bar = '━'.repeat(progress) + ' '.repeat(barLength - progress);
  return bar;
}