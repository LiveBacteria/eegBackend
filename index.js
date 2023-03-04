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

const app = express();
const port = 3232;

// const server = require("http").createServer();
// const io = require("socket.io")(server);
// io.on("connection", (client) => {
//   client.on("event", (data) => {
//     /* … */
//   });
//   client.on("disconnect", () => {
//     /* … */
//   });
// });
// server.listen(6464);

app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());

// create  webhook to send training data to client
// socket.on("trainingData", (data) => {
//   console.log(data);
//   res.send(200, { data: data });
// });

app.get("/", cors(), (req, res) => {
  // res.send("Hello World!");
  //redirect to the client at localhost:3000
  res.redirect("http://localhost:3000");
});

app.get("/test", (req, res) => {
  console.log("Got body:", req.body);
  writeToJSON(req.body);
  res.sendStatus(200);
});

app.post("/test", (req, res) => {
  console.log("Got body:", req.body);
  writeToJSON(req.body);
  res.sendStatus(200);
});

app.post("/api/eeg", (req, res) => {
  // console.log("Got body:", req.body);
  //   appendToJSON(req.body);
  res.sendStatus(200);
});

app.get("/api/trainNewAIModel", (req, res) => {
  console.log("Got body:", req.body);

  res.sendStatus(200);
});

app.get("/api/data.csv", (req, res) => {
  console.log("Got body:", req.body);
  //   appendToJSON(req.body);
  res.sendFile("./data.csv", { root: __dirname });
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});

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

let resultsData;

app.get("/api/modelTrain", (req, res) => {
  //   console.log("Got body:", req.body);
  resultsData = getDataFromDB();
  res.send(200, { data: "Training started" });
});

app.get("/api/modelTrain_Audio", (req, res) => {
  //   console.log("Got body:", req.body);
  resultsData = getDataFromDB("results_new/toneAI_begin");
  res.send(200, { data: "Training started" });
});

app.post("/api/newClientDataPush", (req, res) => {
  //   console.log("Got body:", req.body);
  // push data to firebase
  pushDataToDB(req.body);
  res.send(200, { data: "Data pushed to DB" });
});

app.post("/api/toneAIPredict", async (req, res) => {
  console.log("Got body:", req.body);
  // predict data
  let preproccessedData = preprocessSample(req.body);
  let prediction = await predict(preproccessedData);
  console.log(prediction);
  res.send(200, { data: prediction });
});

const devMode = true;

function log(data) {
  if (devMode) {
    console.log(data);
  }
}

// const io = require("socket.io")(3000, {
//   cors: {
//     origin: "http://localhost:3000",
//     methods: ["GET", "POST"],
//   },
// });

// Load the trained model from disk
// const model = await tf.loadLayersModel("file://path/to/model.json");

// io.on("connection", (socket) => {
//   console.log("Connected");

//   socket.on("sampleData", (data) => {
//     // Preprocess the sample data
//     const sample = preprocessSample(data);

//     // Make a prediction using the model
//     const prediction = model.predict(sample);

//     // Convert the prediction to a JSON object
//     const predictionJSON = prediction.arraySync();

//     // Send the prediction back to the client
//     socket.emit("prediction", predictionJSON);
//   });
// });

function preprocessSample(sampleData) {
  // Convert the sample data to a tensor
  const tensor = tf.tensor2d([sampleData], [1, 48]);

  // Normalize the tensor
  const normalizedTensor = tensor.div(2048);

  return normalizedTensor;
}

// retry at EEG Model Training:

function beginEEGTraining() {
  // get data from firebase
  jsonData = getDataFromDB();

  // Extract the samples and labels arrays from the json data
  let samplesArray = [];
  let labelsArray = [];
  jsonData.forEach((data) => {
    const key = Object.keys(data)[0];
    samplesArray.push(data[key].eeg.samples);
    labelsArray.push(data[key].promptName);
  });

  // Normalize the samples array
  const samplesNormalized = samplesArray.map((samples) => {
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    return samples.map((val) => (val - min) / (max - min));
  });

  // One-hot encode the labels array
  const labelsEncoded = labelsArray.map((label) => {
    const labels = new Set(labelsArray);
    const encoded = Array.from(labels).reduce((encoding, l) => {
      encoding[l] = label === l ? 1 : 0;
      return encoding;
    }, {});
    return encoded;
  });

  console.log(samplesNormalized);
  console.log(labelsEncoded);

  // Assume that samplesNormalized and labelsEncoded are the preprocessed data obtained from previous step
  const testSplit = 0.2;
  const samplesCount = samplesNormalized.length;
  const testCount = Math.floor(samplesCount * testSplit);
  const trainCount = samplesCount - testCount;

  const samplesNormalizedShuffled = shuffle(samplesNormalized);
  const labelsEncodedShuffled = shuffle(labelsEncoded);

  const X_train = samplesNormalizedShuffled.slice(0, trainCount);
  const y_train = labelsEncodedShuffled.slice(0, trainCount);
  const X_test = samplesNormalizedShuffled.slice(trainCount);
  const y_test = labelsEncodedShuffled.slice(trainCount);

  console.log(X_train);
  console.log(y_train);
  console.log(X_test);
  console.log(y_test);

  function shuffle(array) {
    let currentIndex = array.length,
      temporaryValue,
      randomIndex;
    while (0 !== currentIndex) {
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex -= 1;
      temporaryValue = array[currentIndex];
      array[currentIndex] = array[randomIndex];
      array[randomIndex] = temporaryValue;
    }
    return array;
  }

  // Assume that X_train, y_train, X_test, and y_test are the preprocessed and split data
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [X_train[0].length],
      units: 16,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: Object.keys(y_train[0]).length,
      activation: "softmax",
    })
  );
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const batchSize = 32;
  const epochs = 50;

  async function trainModel() {
    const history = await model.fit(
      tf.tensor2d(X_train),
      tf.tensor2d(y_train),
      { batchSize, epochs }
    );
    console.log(history.history.acc);
  }

  async function testModel() {
    const result = model.evaluate(tf.tensor2d(X_test), tf.tensor2d(y_test));
    console.log(result[1].dataSync());
  }

  trainModel();
  testModel();
}

// Import TensorFlow.js
// const tf = require("@tensorflow/tfjs");
// require("@tensorflow/tfjs-node");
// require tfjs node gpu
require("@tensorflow/tfjs-node-gpu");
const path = require("path");

async function predict(newSample) {
  console.log(path.join(__dirname, "toneAI_model_highEpoch"));
  const modelSavePath = path.join(__dirname, "toneAI_model_highEpoch"); // Path to saved model

  // Load the saved model
  // const model = await tf.loadLayersModel(modelSavePath);
  const model = await tf.loadLayersModel(
    "file://./toneAI_model_highEpoch/model.json"
  );

  // Make a prediction on the new sample
  const prediction = model.predict(newSample);
  // Extract the index of the highest probability
  // console.log(prediction);
  const index = prediction.argMax(-1).dataSync()[0];
  console.log(index);
  let label;
  // Get the label corresponding to the index
  switch (index) {
    case 0:
      label = "airTone_440";
      break;
    case 1:
      label = "earthTone_45";
      break;
    case 2:
      label = "fireTone_880";
      break;
    case 3:
      label = "waterTone_220";
      break;
    default:
      label = -1;
      break;
  }
  console.log(`Predicted label: ${label}, index: ${index}`);
  return label;
}

app.post("/api/predictEEG", async (req, res) => {
  //   console.log("Got body:", req.body);
  const prediction1 = await predict(req.body);
  res.send(prediction1);
});

app.get("/api/trainEEG", (req, res) => {
  // console.log("Got body:", req.body);
  //   appendToJSON(req.body);
  beginEEGTraining();
  res.sendStatus(200);
});

// train the model data from groupDataByElectrode
// {
//     electrode,
//     labelName: promptName,
//     samples,
//     }

// create a function that uses the best algorithm for training the model of eeg data

async function trainModel(data) {
  const { formattedData, lengthOfDatasets } = formatData(data);
  // Use sigmoid activation function for binary classification

  let csv = Papa.unparse(formattedData, {
    header: true,
    complete: function (results) {
      fs.writeFile("./data.csv", results, function (err) {
        if (err) {
          return console.log(err);
        }

        console.log("The file was saved!");
      });
    },
  });

  // Load the EEG data from the CSV file
  let eegData = [];
  let headers = [];
  fs.createReadStream("./data.csv")
    .pipe(csv1())
    .on("headers", (row) => {
      headers = row;
    })
    .on("data", (row) => {
      eegData.push(row);
    })
    .on("end", () => {
      // Convert the EEG data to the frequency domain using the FFT
      const fftData = fft(eegData.map((row) => Object.values(row).map(Number)));

      // Save the FFT data to a new CSV file
      const fftDataString =
        headers.join(",") +
        "\n" +
        fftData.map((row) => row.join(",")).join("\n");
      fs.writeFileSync("./fft_data.csv", fftDataString);
    });

  // // save csv to local file
  // fs.writeFile("./data.csv", csv, function (err) {
  //   if (err) {
  //     return console.log(err);
  //   }

  //   console.log("The file was saved!");
  // });

  // get all column names from the data
  const columnNames = Object.keys(formattedData[0]);

  // console.log(formattedData[0]);
  // console.log(columnNames);

  // get the label column name
  const labelColumnName = "promptName";

  // get the features column names
  const featuresColumnNames = columnNames.filter(
    (columnName) => columnName !== labelColumnName
  );

  console.log(featuresColumnNames.length, "featuresColumnNames");

  // create a model
  const model = tf.sequential();

  // add a  input layer with 10 neurons
  model.add(
    tf.layers.dense({
      inputShape: [featuresColumnNames.length],
      activation: "relu",
      units: 36,
    })
  );

  tf.layers.dense({
    activation: "sigmoid",
    units: 4,
  });

  // add a  output layer with 1 neuron
  model.add(
    tf.layers.dense({
      activation: "sigmoid",
      units: 4,
    })
  );

  model.add(
    tf.layers.dense({
      activation: "sigmoid",
      units: 1,
    })
  );

  // compile the model for eeg data training using categorical crossentropy loss
  model.compile({
    // loss: "absoluteDifference",
    // loss: "categoricalCrossentropy",
    // loss: "meanSquaredError",
    // loss: "meanAbsoluteError",
    // loss: "meanAbsolutePercentageError",
    // loss: "meanSquaredLogarithmicError",
    // loss: "squaredHinge",
    // loss: "hinge",
    // loss: "categoricalHinge",
    loss: "logcosh",
    // loss: "huberLoss",
    // loss: "cosineProximity",
    metrics: ["accuracy", "mse"],
    // set an optimizer that is not tf.train.adam or sgd because they don't work with categorical crossentropy
    // optimizer: tf.train.adagrad(0.6),
    optimizer: tf.train.adam(0.6),
  });

  // convert the data to a form we can use for training
  function convertToTensors() {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(formattedData);

      // Step 2. Convert data to Tensor
      const inputs = formattedData.map((d) =>
        featuresColumnNames.map((name) => d[name])
      );

      console.log(inputs, "inputs");

      const labels = formattedData.map((d) => d[labelColumnName]);

      const inputTensor = tf.tensor2d(inputs, [
        inputs.length,
        inputs[0].length,
      ]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor
        .sub(inputMin)
        .div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor
        .sub(labelMin)
        .div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      };
    });
  }

  // convert the data to a form we can use for training
  const { inputs, labels } = convertToTensors(formattedData);

  // train the model
  await model.fit(inputs, labels, {
    batchSize: 32,
    epochs: 1000,
    // callbacks: tf.node.tensorBoard("/tmp/tfjs_logs"),
  });

  // evaluate the model
  const evalOutput = model.evaluate(inputs, labels);
  console.log(`Accuracy: ${(evalOutput[1].dataSync()[0] * 100).toFixed(1)}%`);

  // make some predictions using the model and compare them to the
  // labels
  const preds = model.predict(inputs).dataSync();
  labels.dataSync().forEach((val, i) => {
    console.log(`Label: ${val}, Prediction: ${preds[i]}`);
  });

  // save the model
  await model.save("file://model");
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

async function trainAudioMo_del_Dynamic(data, identifier) {
  console.log(data, "data");
  console.log(identifier, "identifier");

  let resultsArray = [];

  Object.keys(data).forEach((key) => {
    for (let i = 0; i < data[key].length; i++) {
      resultsArray.push(data[key][i]);
    }
  });

  // console.log(resultsArray);

  const numClasses = 4;

  if (!Array.isArray(resultsArray)) {
    console.error("Results is not an array.");
  }

  function applyFFT(inputTensor) {
    // console.log("inputArray", inputTensor, inputTensor.shape);
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
    // log("finished, 301");
    return fftOutput;
  }

  const newResultsArray = [];

  for (let i = 0; i < resultsArray.length; i++) {
    const result = resultsArray[i][0];
    const label = resultsArray[i][1];
    const tempArray = [];

    // loop the result array and split and create 4 new arrays of 12 each
    for (let j = 0; j < result.length; j += 12) {
      // const newResult = result.slice(j, j + 12);
      const newResult = tf.tensor1d(result.slice(j, j + 12), "float32");
      const fftResult = applyFFT(newResult);
      // console.log(fftResult, "fftResult");
      tempArray.push(fftResult);
    }
    // log("finished, 620");
    log([tempArray, label], "tempArray");
    newResultsArray.push([tempArray, label]);
  }

  console.log(newResultsArray, "newResultsArray");

  const dataset = tf.data.array(newResultsArray);

  // Shuffle the data and split into training, validation, and test sets
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 128, activation: "relu", inputShape: [12, 4] }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 64, activation: "relu" }),
      tf.layers.dense({ units: 3, activation: "softmax" }),
    ],
  });

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Train the model on the training set
  const epochs = 10;
  const batchSize = 32;

  const trainBatch = trainDataset.batch(batchSize);
  const valBatch = valDataset.batch(batchSize);
  const testBatch = testDataset.batch(batchSize);

  await model.fit(
    dataset.shuffle(inputData[0].length).batch(batchSize).take(numTrainSamples),
    {
      epochs: numEpochs,
      validationData: dataset
        .skip(numTrainSamples)
        .batch(batchSize)
        .take(numValSamples),
    }
  );
  console.log("Finished training the model");
  console.log(history);
  console.log("Evaluating model on test data...");

  const result = model.evaluate(testBatch);
  console.log(`Test loss: ${result[0]}, Test accuracy: ${result[1]}`);

  // Train the model on the training set
  // await model.fit(trainDataset.batch(batchSize), {
  // epochs,
  // validationData: valDataset.batch(batchSize),
  // });

  // const [testLoss, testAcc] = model.evaluate(testDataset.batch(batchSize));
  // console.log(
  // `Test loss: ${testLoss.toFixed(4)}, Test accuracy: ${testAcc.toFixed(4)}`
  // );

  // const [testLoss, testAcc] = model.evaluate(testDataset.batch(batchSize));
  // console.log(
  // `Test loss: ${testLoss.toFixed(4)}, Test accuracy: ${testAcc.toFixed(4)}`
  // );

  const modelSavePath = "file://toneAI_model_500Epoch"; // Path to save the model

  // await model.save(modelSavePath);
  console.log("Model saved successfully!");

  // Use the model to make predictions
  // const predictions = model.predict(x);

  // Convert the predictions to labels
  // const predictedLabels = Array.from(tf.argMax(predictions, 1).dataSync());
}

// backup:
async function trainAudioModel(data) {
  console.warn(
    "trainAudioModel is deprecated. Use trainAudioModel_Dnyamic instead."
  );
  console.log(data, "data");
  const resultsArray = prepData(data);
  console.log(resultsArray);

  const numClasses = 4;

  if (!Array.isArray(resultsArray)) {
    console.error("Results is not an array.");
  }

  // // Define the model architecture
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [48], units: 64, activation: "relu" })
  );
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));

  // Compile the model
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Overly complex model below
  // Define the model architecture
  // const model = tf.sequential();
  // model.add(
  //   tf.layers.dense({
  //     inputShape: [48],
  //     units: 128,
  //     activation: "relu",
  //     kernel_regularizer: tf.regularizers.l2({ l2: 0.005 }),
  //   })
  // );
  // model.add(tf.layers.dropout({ rate: 0.5 }));
  // model.add(
  //   tf.layers.dense({
  //     units: 64,
  //     activation: "relu",
  //     kernel_regularizer: tf.regularizers.l2({ l2: 0.05 }),
  //   })
  // );
  // model.add(tf.layers.dropout({ rate: 0.5 }));
  // model.add(
  //   tf.layers.dense({
  //     units: numClasses,
  //     activation: "softmax",
  //   })
  // );

  // // Compile the model
  // model.compile({
  //   optimizer: "adam",
  //   loss: "categoricalCrossentropy",
  //   metrics: ["accuracy"],
  // });

  const features = [];
  const labels = [];

  // shuffle the data
  let shuffled = resultsArray.sort(() => 0.5 - Math.random());

  // Prepare the data
  for (let i = 0; i < resultsArray.length; i++) {
    features.push(resultsArray[i][0]);
    labels.push(resultsArray[i][1]);
  }

  const fLength = features.length;

  console.log(features, "features");
  console.log(labels, "labels");

  const x = tf.tensor2d(features, [fLength, 48]);
  const y = tf.oneHot(tf.tensor1d(labels, "int32"), numClasses);

  const history = await model.fit(x, y, { epochs: 500 });

  const modelSavePath = "file://toneAI_model_rework"; // Path to save the model

  await model.save(modelSavePath);
  console.log("Model saved successfully!");

  // Use the model to make predictions
  const predictions = model.predict(x);

  // Convert the predictions to labels
  const predictedLabels = Array.from(tf.argMax(predictions, 1).dataSync());
}

function formatData(data) {
  let dataArr = [];
  let groupObj = {};

  // for each test Object.keys(data).length)
  for (let i = 0; i < Object.keys(data).length - 1; i++) {
    console.log(Object.keys(data)[3]);
    // for each index of the test
    for (const property in data[Object.keys(data)[i]]) {
      let promptName =
        data[Object.keys(data)[i]][property][
          Object.keys(data[Object.keys(data)[i]][property])[0]
        ].promptName;
      const { electrode, index, samples } =
        data[Object.keys(data)[i]][property][
          Object.keys(data[Object.keys(data)[i]][property])[0]
        ].eeg;

      // flatten the data
      let feature = {};
      //   feature.electrode = electrode;
      //   feature.index = index;
      samples.forEach((sample, i) => {
        feature["sample" + electrode + i] = samples[i];
        if (groupObj[index] == undefined) {
          groupObj[index] = {};
        }

        if (groupObj[index]["sample" + electrode + i] == undefined) {
          groupObj[index]["sample" + electrode + i] = [];
        }
        // normalize the data
        normedData = (samples[i] + 1000) / 2000;
        // normedData = samples[i];

        if (normedData === NaN || normedData === undefined) {
          console.log("normedData", normedData);
        }

        groupObj[index]["sample" + electrode + i] = normedData;
        // console.log("sample" + i, samples[i]);
      });

      let promptNameNew;

      switch (promptName) {
        case "BlueColor":
          // promptNameNew = [1, 0, 0, 0];
          // promptNameNew = 0.0;
          // promptNameNew = "BlueColor";
          promptNameNew = 0;
          break;
        case "GreenColor":
          // promptNameNew = [0, 1, 0, 0];
          // promptNameNew = 0.33;
          // promptNameNew = "GreenColor";
          promptNameNew = 1;
          break;
        case "RedColor":
          // promptNameNew = [0, 0, 1, 0];
          // promptNameNew = 0.66;
          // promptNameNew = "RedColor";
          promptNameNew = 2;
          break;
        case "YellowColor":
          // promptNameNew = [0, 0, 0, 1];
          // promptNameNew = 1.0;
          // promptNameNew = "YellowColor";
          promptNameNew = 3;
          break;
      }

      //   if (promptName == "YellowColor") {
      //     console.log("promptName", promptName);
      //     console.log("promptNameNew", promptNameNew);
      //   }

      //   console.log(parseInt(promptNameNew));

      promptNameNew = parseInt(promptNameNew);

      groupObj[index]["promptName"] = promptNameNew;

      //   let newNumbName = promptName.match(/\d/g);

      //   newNumbName = newNumbName.join("");

      // console.log(feature, "feature");

      // May need to add index as a key for further training fidelity
      let tempObj = {
        ...feature,
        promptName: promptNameNew,
        // promptName,
      };

      dataArr.push(tempObj);
    }
    console.log(dataArr);
  }

  let dataArrNew = [];

  for (const property in groupObj) {
    dataArrNew.push(groupObj[property]);
  }

  console.log(dataArrNew.length, "formattedData.length");

  return {
    formattedData: dataArrNew,
    //   lengthOfValues: Object.keys(dataArr[0].features.flat()).length,
    lengthOfDatasets: dataArr.length,
  };
}

async function getDataFromDB_new(path) {
  var ref = db.ref("app/" + path);

  let data = await ref.once("value", function (snapshot) {
    // console.log(snapshot.val());
    const result = snapshot.val();
    // console.log the  first  object of  the  object
    // console.log(data.YellowColor);
    return result;
  });
  return data;
}

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

// push data to firebase
function pushDataToDB(data, path) {
  let ref = db.ref("app/");
  if (path) {
    ref = db.ref("app/" + path);
    ref.push(data);
  } else {
    console.warn.log("No path provided");
  }
}

const WebSocket = require("ws");

const wss = new WebSocket.Server({ port: 8080 });

wss.on("connection", (ws) => {
  console.log("Client connected");

  ws.on("message", (message) => {
    console.log(`Received message: ${message}`);

    // Handle the message and send a response if necessary
    const response = handleMessage(message);
    if (response) {
      ws.send(response);
    }
  });

  ws.on("close", () => {
    console.log("Client disconnected");
  });
});

function handleMessage(message) {
  // Handle the message and return a response
  return "Response";
}

// /api/recieveEEGData that takes in the data from the EEG along with the promptName and saves it to the database
app.post("/api/recieveEEGData", (req, res) => {
  const { data, promptName } = req.body;

  console.log("data", data);
  console.log("promptName", promptName);

  //TODO = save data to firebase
});

//TODO = configure firebase
const firebaseConfig = {
  apiKey: "AIzaSyAOFLMQQmAEo27A-q3Lq5xcUcFuy-4kPXs",
  authDomain: "isistr-db.firebaseapp.com",
  projectId: "isistr-db",
  storageBucket: "isistr-db.appspot.com",
  messagingSenderId: "147532042641",
  appId: "1:147532042641:web:be99fbf7e44343ec858a30",
  databaseURL: "https://isistr-db-default-rtdb.firebaseio.com/",
};

gotData = (data) => {
  const testData = data.val();
  const keys = Object.keys(testData);

  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    // const testLabel = testData[k].testLabel;
    // const testDate = testData[k].testDate;
    // const testIndex = testData[k].testIndex;
    // const electrode = testData[k].electrode;
    // data = testData[k].data;
  }
  console.log(data);
  //   sortTestData(data);
};

errData = (err) => {
  console.log("Error!");
  console.log(err);
};
