using NNLib;
using NNLib.ActivationFunction;
using System;
using Xunit;
using Xunit.Abstractions;

namespace UnitTests
{
    public class TrainerPerformanceTests : TrainerTestBase
    {
        private readonly ITestOutputHelper output;

        public TrainerPerformanceTests(ITestOutputHelper output)
        {
            this.output = output;
        }

        private void TestAndGate(GradientDescentParams parameters, ILossFunction lossFunction, TimeSpan timeout)
        {
            var net = CreateNetwork(2, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                parameters, lossFunction);

            VerifyTrainingError(0.01, trainer, output, timeout);
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_online_GD()
        {
            TestAndGate(new GradientDescentParams()
            {
                Momentum = 0.9, LearningRate = 0.2, BatchSize = 1
            }, new QuadraticLossFunction(), TimeSpan.FromMinutes(1));
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_minibatch_GD()
        {
            TestAndGate(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 2
            }, new QuadraticLossFunction(), TimeSpan.FromMinutes(1));
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_batch_GD()
        {
            TestAndGate(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 4
            }, new QuadraticLossFunction(), TimeSpan.FromMinutes(2));
        }
    }
}