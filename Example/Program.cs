using System;
using NNLib.ActivationFunction;
using NNLib.Csv;
using NNLib.LossFunction;
using NNLib.MLP;
using NNLib.Training.GradientDescent;

namespace Example
{
    class Program
    {
        static void Main(string[] args)
        {
            var data = CsvFacade.LoadSets(@"C:\Users\Marek\Desktop\bigsin.csv").sets;

            var mlp = MLPNetwork.Create(1,
                (30, new SigmoidActivationFunction()), (1, new LinearActivationFunction())
            );

            var algorithm = new GradientDescentAlgorithm(new GradientDescentParams
            {
                LearningRate = 0.0001,
            });
            var lossFunction = new QuadraticLossFunction();
            var trainer = new MLPTrainer(mlp, data, algorithm, lossFunction);

            while (trainer.Error > 0.001)
            {
                trainer.DoEpoch();
                Console.WriteLine(trainer.ProgressString());
            }
        }
    }
}
