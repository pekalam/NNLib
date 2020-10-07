using NNLib;
using System;
using System.Diagnostics;
using MathNet.Numerics;
using NNLib.Common;
using NNLib.Csv;
using NNLib.Tests;

namespace TrainingTest
{
    class Program
    {
        static void Main(string[] args)
        {
            // Control.UseNativeMKL();
            //Control.UseManaged();

            var lossFunction = new QuadraticLossFunction();
            var net = new MLPNetwork(new PerceptronLayer(1, 10, new TanHActivationFunction()),
                new PerceptronLayer(10, 10, new TanHActivationFunction()),
                new PerceptronLayer(10, 1, new LinearActivationFunction()));
            var trainer = new MLPTrainer(net,
                new SupervisedTrainingSets(CsvFacade.LoadSets("E:\\sin.csv").sets.TrainingSet),
                new GradientDescentAlgorithm(new GradientDescentParams()
                {
                    LearningRate = 0.001, Momentum = 0.0, BatchSize = 37,
                }),
                lossFunction
            );


            int i = 0;
            var s = Stopwatch.StartNew();
            while (trainer.Error > 0.01 && trainer.Epochs != 10_000)
            {
                trainer.DoEpoch();
                i++;
                if (i == 1000)
                {
                    Console.WriteLine($"{trainer.Epochs} {trainer.Error}");
                    i = 0;
                }
            }

            s.Stop();

            Console.WriteLine(trainer.Error + " " + s.Elapsed);

            Console.ReadKey();
        }
    }
}