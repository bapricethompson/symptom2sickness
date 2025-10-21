const fs = require("fs");
const csv = require("csv-parser");

const inputFile = "./data.csv";
const outputFile = "./data.json";

const results = [];

fs.createReadStream(inputFile)
  .pipe(csv())
  .on("data", (row) => {
    results.push(row);
  })
  .on("end", () => {
    fs.writeFileSync(outputFile, JSON.stringify(results, null, 2));
    console.log(`CSV converted to JSON! Saved as ${outputFile}`);
  });
