// require("./abs");

const express = require("express");
const bodyParser = require("body-parser");
var cors = require("cors");
const {
  writeToJSON,
  appendToJSON,
  readFromJSON,
  prepData,
} = require("./utils.js");

// const writeToJSON = require("./utils.js").writeToJSON;
// const appendToJSON = require("./utils.js").appendToJSON;
// const readFromJSON = require("./utils.js").readFromJSON;
// const prepData = require("./utils.js").prepData;
const fs = require("fs-extra");
const csv1 = require("csv-parser");
const fft = require("fft-js").fft;
const fftUtil = require("fft-js").util;
// const tfvis = require("@tensorflow/tfjs-vis");

const Papa = require("papaparse");

//TODO = require tensorflow and firebase
const tf = require("@tensorflow/tfjs-node-gpu");
// import * as tf from "@tensorflow/tfjs-node-gpu";
var admin = require("firebase-admin");

console.log(tf.getBackend());

const logdir = "logs";
const summaryWriter = tf.node.summaryFileWriter(logdir);

// Fetch the service account key JSON file contents
var serviceAccount = require("./isistr-db-firebase-adminsdk-gzboz-fecbf1a908.json");
// const { tensor } = require("@tensorflow/tfjs-node");
// const {
//   computeOutShape,
// } = require("@tensorflow/tfjs-core/dist/ops/segment_util.js");

// Initialize the app with a service account, granting admin privileges
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  // The database URL depends on the location of the database
  databaseURL: "https://isistr-db-default-rtdb.firebaseio.com/",
});

// As an admin, the app has access to read and write all data, regardless of Security Rules
var db = admin.database();
var ref = db.ref("restricted_access/secret_document");
ref.once("value", function (snapshot) {
  console.log(snapshot.val());
});

function getDataFromDB(datasetName) {
  switch (datasetName) {
    case "results":
      var ref = db.ref("app/results");
      break;
    case "toneAI_Results":
      var ref = db.ref("app/toneAI_Results");
      break;
    case "toneAI_Results1":
      var ref = db.ref("app/toneAI_Results1");
      break;
    case "results_new/toneAI_begin":
      var ref = db.ref("app/results_new/toneAI_begin");
      break;
    default:
      var ref = db.ref("app/results");
      break;
  }

  ref.once("value", function (snapshot) {
    // console.log(snapshot.val());
    const data = snapshot.val();
    // console.log the  first  object of  the  object
    // console.log(data.YellowColor);
    if (datasetName.includes("toneAI")) {
      trainAudioModel_Dynamic(data, datasetName);
    } else {
      trainModel(data);
    }
    return data;
  });
}

async function trainAudioModel_Dynamic(data, identifier) {
  console.log(data, "data");
  console.log(identifier, "identifier");

  let resultsArray = [];

  Object.keys(data).forEach((key) => {
    for (let i = 0; i < data[key].length; i++) {
      resultsArray.push(data[key][i]);
    }
  });

  const numClasses = 4;

  if (!Array.isArray(resultsArray)) {
    console.error("Results is not an array.");
  }

  function applyFFT(inputTensor) {
    const fftOutput = [];

    for (let i = 0; i < 12; i++) {
      const signal = inputTensor.slice([i], [1]);
      const signalLength = signal.shape[0];
      const signalLengthTensor = tf.scalar(signalLength);

      // Apply FFT to the signal
      const signalFFT = tf.spectral.rfft(signal);

      // Compute the power spectrum of the signal
      const powerSpectrum = tf.abs(signalFFT).square().div(signalLengthTensor);

      // Convert power spectrum to a regular array
      const powerSpectrumData = powerSpectrum.dataSync();

      // Store the power spectrum for this signal in the output array
      fftOutput.push(powerSpectrumData);

      // Dispose of the tensors we created
      signal.dispose();
    }

    return fftOutput;
  }

  const newResultsArray = resultsArray.map(([result, label], index) => {
    const tempArray = [];

    //  NEED TO FLATTEN FFT data array prior to passing to the model to train on
    // let result = [];
    // for (var electrode in arr) {
    //   console.log(arr[electrode], "electrode");
    //   for (var entry in electrode) {
    //     console.log(electrode[entry], "entry");
    //   }
    // }

    // loop the result array and split and create 4 new arrays of 12 each
    // resultsArray.length
    if (index < resultsArray.length) {
      for (let j = 0; j < result.length; j += 12) {
        const newResult = tf.tensor1d(result.slice(j, j + 12), "float32");
        const fftResult = applyFFT(newResult);
        console.log("Electrode processed");
        tempArray.push(fftResult);
      }
    }

    return [tempArray, label];
  });

  pushDataToDB(newResultsArray, "/fftData/");

  log("Finished processing data.");

  const newData = newResultsArray.map(([tempArray, label]) => {
    const arr = [];
    for (let i = 0; i < tempArray.length; i++) {
      const subArr = tempArray[i].map((item) => [...item]);
      while (subArr.length < 12) {
        subArr.push(new Array(12).fill(0));
      }
      arr.push(subArr);
    }
    return [arr, label];
  });

  const dataset = tf.data.array(newData);

  dataset.forEachAsync((element) => {
    console.log(element.shape); // output: [3, 4]
    return Promise.resolve(); // required to avoid warning message
  }, this);

  console.log(newData[1], "newData");

  // Shuffle the data and split into training, validation, and test sets
  const numExamples = newResultsArray.length;
  const numTrainExamples = Math.floor(numExamples * 0.7);
  const numValExamples = Math.floor(numExamples * 0.15);
  const numTestExamples = numExamples - numTrainExamples - numValExamples;
  const batchSize = 32;

  // // Reshape input dataset to match the input shape of the model
  // const reshapedData = tf.tensor4d(dataset, [numExamples, 4, 12, 12]);

  const trainDataset = dataset.take(numTrainExamples).batch(batchSize);
  const valDataset = dataset
    .skip(numTrainExamples)
    .take(numValExamples)
    .batch(batchSize);
  const testDataset = dataset
    .skip(numTrainExamples + numValExamples)
    .batch(batchSize);

  const model = tf.sequential({
    layers: [
      tf.layers.dense({
        units: 128,
        activation: "relu",
        inputShape: [4, 12, 1],
      }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 64, activation: "relu" }),
      tf.layers.dense({ units: numClasses, activation: "softmax" }),
    ],
  });

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Set up TensorBoard callback
  const tensorBoardCallback = tf.node.tensorBoard(logdir, {
    updateFreq: "epoch",
    histogramFreq: 1,
  });

  // Train the model on the reshaped data
  await model.fit(trainDataset, {
    epochs: 10,
    validationData: valDataset,
    callbacks: [tensorBoardCallback],
  });

  // Evaluate the model on the test dataset
  const evalOutput = model.evaluate(testDataset);

  // Log the evaluation accuracy
  console.log(`Test Accuracy: ${(await evalOutput[1].data())[0]}`);

  // Train the model on the reshaped data
  // trainDataset
  // await model.fit(trainDataset, {
  //   epochs: 25,
  //   validationData: valDataset,
  //   callbacks: tfvis.show.fitCallbacks(
  //     { name: "Training Performance" },
  //     ["loss", "val_loss", "acc", "val_acc"],
  //     { callbacks: ["onEpochEnd"] }
  //   ),
  // });

  // Save the model
  await model.save("file://./model");
}

getDataFromDB("results_new/toneAI_begin");
