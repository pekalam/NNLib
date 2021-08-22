# NNLib

Neural network library written in C#.

## Features:
- Implementation of MLP neural network
- Gradient descent (with momentum) and Levenberg-Marquardt algorithms
- Various normalization algorithms
- Support of CSV files as separate package (high extensibility)

# Usage
```csharp
// load and divide data from CSV file
var csv = CsvFacade.LoadSets(@"C:\Users\Marek\Desktop\sin.csv",
    new RandomDataSetDivider(),
    new DataSetDivisionOptions { TrainingSetPercent = 80, ValidationSetPercent = 10, TestSetPercent = 10, });

// perform min-max data normalization
var normalization = new MinMaxNormalization(-1, 1);
await normalization.FitAndTransform(csv.sets);

// create 1 x 5 x 1 MLP neural network
var mlp = MLPNetwork.Create(
    1, (5, new SigmoidActivationFunction()), (1, new LinearActivationFunction())
);

// use gradient descent algorithm
var algorithm = new GradientDescentAlgorithm(new GradientDescentParams
{
    LearningRate = 0.001,
    Randomize = true,
    Momentum = 0.01,
});
// use quadratic loss function
var lossFunction = new QuadraticLossFunction();

// train neural network until error > 0.001
var trainer = new MLPTrainer(mlp, csv.sets, algorithm, lossFunction);
while (trainer.Error > 0.001)
{
    trainer.DoEpoch();
    Console.WriteLine($"Error: {trainer.Error}");
    Console.WriteLine($"Validation error: {trainer.RunValidation()}");
}
```