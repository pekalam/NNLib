using System;
using System.Threading.Tasks;
using NNLib;
using NNLib.ActivationFunction;
using NNLib.Csv;
using NNLib.Data;
using NNLib.LossFunction;
using NNLib.MLP;
using NNLib.Training.GradientDescent;

namespace Example
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var csv = CsvFacade.LoadSets(@"C:\Users\Marek\Desktop\sin.csv", new RandomDataSetDivider(),
                new DataSetDivisionOptions {TrainingSetPercent = 80, ValidationSetPercent = 10, TestSetPercent = 10,});

            var norma = new RobutstNormalization();
            await norma.FitAndTransform(csv.sets);

            var mlp = MLPNetwork.Create(
                1, (5, new SigmoidActivationFunction()), (1, new LinearActivationFunction())
            );

            var algorithm = new GradientDescentAlgorithm(new GradientDescentParams
            {
                LearningRate = 0.001, Randomize = true, Momentum = .1,
            });
            var lossFunction = new QuadraticLossFunction();
            var trainer = new MLPTrainer(mlp, csv.sets, algorithm, lossFunction);

            while (trainer.Error > 0.001)
            {
                trainer.DoEpoch();
                Console.WriteLine(trainer.ProgressString());
                Console.WriteLine($"Validation error: {trainer.RunValidation()}");
            }
        }
    }
}