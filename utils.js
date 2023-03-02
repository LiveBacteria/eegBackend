const fs = require("fs-extra");

//  Export a function that writes data to a JSON file
function writeToJSON(data) {
  //   const jsonNew = JSON.stringify(data, null, 2);
  fs.writeJson("./data.json", data, (err) => {
    if (err) return console.error(err);
    console.log("success!");
  });
}

//  Export a function that appends data to a JSON file
function appendToJSON(data) {
  const json = fs.readJsonSync("./data.json");
  const jsonNew = JSON.stringify(json.concat(data), null, 2);
  fs.writeFileSync("./data.json", jsonNew);
}

function prepData(data) {
  console.log(data);
  const results = {};

  const determinePromptNameID = (promptName) => {
    switch (promptName) {
      case "airTone_440":
        promptName = 0;
        break;
      case "earthTone_45":
        promptName = 1;
        break;
      case "fireTone_880":
        promptName = 2;
        break;
      case "waterTone_220":
        promptName = 3;
        break;
      default:
        promptName = 0;
        break;
    }
    return promptName;
  };

  // iterate through each level until data reached, data is an object
  Object.keys(data).forEach((test) => {
    Object.keys(data[test]).forEach((excess) => {
      Object.keys(data[test][excess]).forEach((datum) => {
        datum = data[test][excess][datum];
        // iterate through each EEG reading in the data
        // for (const { eeg, promptName } of datum) {
        const { promptName } = datum;
        const { electrode, index, samples } = datum["eeg"];

        // create an object with default values if the index does not exist
        if (!results.hasOwnProperty(index)) {
          results[index] = { promptName: 0, samples: Array(48).fill(0) };
        }

        // update the prompt name for the index
        results[index].promptName = determinePromptNameID(promptName);

        // update the EEG readings for the electrode and index
        const startIndex = electrode * 12;
        for (let i = 0; i < 12; i++) {
          results[index].samples[startIndex + i] = samples[i];
        }
      });
    });
  });

  console.log(results);
  const resultsArray = Object.values(results).map(({ samples, promptName }) => [
    samples,
    promptName,
  ]);
  return resultsArray;
}

// Export a function that collects data while an eeg is being recorded
// function collectData(data) {

//  Export a function that reads data from a JSON file
function readFromJSON() {
  const json = Deno.readTextFileSync("data.json");
  return JSON.parse(json);
}

module.exports = { readFromJSON, writeToJSON, appendToJSON, prepData };
