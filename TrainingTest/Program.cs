using System;
using NNLib;
using NNLib.ActivationFunction;
using UnitTests;

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
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                new GradientDescentParams()
                {
                    BatchSize = 1, LearningRate = 0.1, Momentum = 0.2,
                }, lossFunction);


            while (trainer.Error > 0.01)
            {
                trainer.DoEpoch();
            }

            Console.WriteLine(trainer.Error);

            Console.ReadKey();
        }
    }
}
