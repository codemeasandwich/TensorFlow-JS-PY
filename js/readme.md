# Comparative Study of JavaScript and Python for Image Classification using TensorFlow

This research project offers a comparative study of JavaScript (TensorFlow.js for Node.js) and Python (TensorFlow) for image classification tasks using the MNIST dataset (a set of 70,000 handwritten digits). The study aims to address the lack of academic research directly comparing these two approaches and provide insights into the feasibility and practicality of using JavaScript for machine learning development.

## Project Structure

- `main.js`: The main JavaScript script that loads the MNIST dataset, preprocesses the data, creates and trains the CNN model, and evaluates its performance.
- `dataset.js`: The JavaScript module responsible for loading and parsing the MNIST dataset files.
- `dataset/`: The directory containing the MNIST dataset files.


## Getting Started

To run the JavaScript script:

1. Install the required dependencies:
   ```
   npm install
   ```

2. Place the MNIST dataset files in the `dataset/` directory.

3. Run the `main.js` script:
   ```
   node main.js
   ```

The script will load the MNIST dataset, preprocess the data, create and train the CNN model, and evaluate its performance. It will also test the model on a specific image index and display the predicted label.