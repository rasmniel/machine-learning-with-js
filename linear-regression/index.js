const fs = require('fs');
const readLine = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
});

const SimpleLinearRegression = require('ml-regression-simple-linear');
let SLR;

const
    x = [], // Input
    y = []; // Output

const field = 'TV';
const result = 'Sales';

const json = JSON.parse(fs.readFileSync('./linear-regression/advertising.json', 'UTF-8'));
// Initialize data into the two arrays.
json.forEach((row) => {
    // console.log(row[field] + ', ' + row[result]);
    x.push(row[field]);
    y.push(row[result]);
});

function learn() {
    SLR = new SimpleLinearRegression(x, y);
    console.log(SLR.toString(5));
    readLine.question('Enter input X for prediction > ', (answer) => {
        const parsedAnswer = parseFloat(answer);
        const prediction = SLR.predict(parsedAnswer).toFixed(2);
        console.log(`At X (${field}) = ${parsedAnswer}, y (${result}) = ${prediction}`);
        learn();
    });
}

learn();