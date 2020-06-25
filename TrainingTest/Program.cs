using NNLib;
using System;
using NNLib.Common;
using NNLib.Tests;

namespace TrainingTest
{
    class Program
    {
        static void Main(string[] args)
        {
            var lossFunction = new QuadraticLossFunction();
            var net = new MLPNetwork(new PerceptronLayer(2, 50, new SigmoidActivationFunction()),
                new PerceptronLayer(50, 50, new SigmoidActivationFunction()),
                new PerceptronLayer(50, 1, new SigmoidActivationFunction()));
            var trainer = new MLPTrainer(net, 
                new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                new GradientDescentAlgorithm(new GradientDescentParams()
                {
                    LearningRate = 0.1, Momentum = 0.2,
                }), 
                lossFunction, 
                new BatchParams() {BatchSize = 1}
                );


            while (trainer.Error > 0.01)
            {
                trainer.DoEpoch();
            }

            Console.WriteLine(trainer.Error);

            Console.ReadKey();
        }
    }
}