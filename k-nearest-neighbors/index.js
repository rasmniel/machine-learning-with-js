const fs = require('fs');
const prompt = require('prompt');

const kNearestNeighbors = require('ml-knn');
let kNN;

const
    xSet = [], // Input (array of value arrays for each index)
    ySet = []; // Output (all type numbers corresponding to X indices)

const json = fs.readFileSync('./k-nearest-neighbors/iris.json', 'UTF-8');

// Shuffle to avoid sorted data.
const data = DurstenfeldShuffle(JSON.parse(json));
const separationSize = 0.7 * data.length;
const types = getTypes(data);
populateSets(data);

function DurstenfeldShuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}

/**
 * There are three different types of Iris flowers
 * that this dataset classifies.
 *
 * 0. Iris Setosa (Iris-setosa)
 * 1. Iris Versicolor (Iris-versicolor)
 * 2. Iris Virginica (Iris-virginica)
 */
function getTypes(data) {
    // Create array of unique types.
    let types = new Set();
    data.forEach((row) => {
        types.add(row.type);
    });
    return [...types];
}

function populateSets(data) {
    data.forEach((row) => {
        let rowArray = Object.keys(row).map(key => row[key]);
        rowArray = rowArray.filter(element => typeof element === 'number');
        const typeNumber = types.indexOf(row.type);

        xSet.push(rowArray);
        ySet.push(typeNumber);
    });
}

// Separate set into training and test by separationSize
// Separation size 0.7 = first 70% of data will be training, the latter 30% will be test.
function separate() {
    // Training set is first part of data set.
    const trainingSetX = xSet.slice(0, separationSize);
    const trainingSetY = ySet.slice(0, separationSize);
    train(trainingSetX, trainingSetY);

    // Test (verification) set is latter part of data set.
    const testSetX = xSet.slice(separationSize);
    const testSetY = ySet.slice(separationSize);
    test(testSetX, testSetY);
}

function train(x, y) {
    // Create kNN algorithm with training set and neighbor-count (k).
    kNN = new kNearestNeighbors(x, y, {k: 7});
}

function test(x, y) {
    const prediction = kNN.predict(x);
    const misclassifications = prediction.filter((element, index) => element !== y[index]);
    console.log(`Test Set Size = ${x.length} and number of misclassifications = ${misclassifications.length}`);
}

function learn() {
    separate();
    let input = [];
    prompt.start();
    prompt.get(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'], (error, result) => {
        if (error) return;
        for (const key in result) if (result.hasOwnProperty(key))
            input.push(+result[key]);
        // input = [1.7, 2.5, 0.5, 3.4]; // = 2?
        const prediction = kNN.predict(input);
        console.log(`With ${input}, type =  ${prediction}`);
    });
}

learn();