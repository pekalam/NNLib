using System;
using System.Threading.Tasks;
using NNLib.ActivationFunction;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class TrainerPerformanceTests : TrainerTestBase
    {
        private readonly ITestOutputHelper _output;

        public TrainerPerformanceTests(ITestOutputHelper output)
        {
            _output = output;
        }

        private void TestAndGate(GradientDescentParams parameters, ILossFunction lossFunction, TimeSpan timeout)
        {
            var net = CreateNetwork(2, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                parameters, lossFunction);

            VerifyTrainingError(0.01, trainer, _output, timeout);
        }

        private async Task TestAndGateAsync(GradientDescentParams parameters, ILossFunction lossFunction, TimeSpan timeout)
        {
            var net = CreateNetwork(2, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                parameters, lossFunction);

            await VerifyTrainingErrorAsync(0.01, trainer, _output, timeout);
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


        [Fact]
        public async Task MLP_approximates_AND_gate_with_online_GD_async()
        {
            await TestAndGateAsync(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 1
            }, new QuadraticLossFunction(), TimeSpan.FromMinutes(1));
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_minibatch_GD_async()
        {
            await TestAndGateAsync(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 2
            }, new QuadraticLossFunction(), TimeSpan.FromMinutes(1));
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_batch_GD_async()
        {
            await TestAndGateAsync(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 4
            }, new QuadraticLossFunction(), TimeSpan.FromMinutes(2));
        }
    }
}