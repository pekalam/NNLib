using System;
using FluentAssertions;
using NNLib.Common;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class MLPTrainerTest : TrainerTestBase
    {
        public MLPTrainerTest(ITestOutputHelper output) : base(output)
        {
        }
        
        private MLPTrainer CreateBasicAndGateTrainer(MLPNetwork net)
        {
            return new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                new GradientDescentAlgorithm(new GradientDescentParams()
                {
                    LearningRate = 0.9,
                    Momentum = 0.1
                }), new QuadraticLossFunction());
        }

        [Fact]
        public void Ctor_throws_when_training_data_sizes_does_not_match_with_network_layers_size()
        {
            var (net1, _) = CreateMockNetwork(1, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));

            Assert.Throws<Exception>(() => CreateBasicAndGateTrainer(net1.Object));

            var (net2, _) = CreateMockNetwork(2, (2, new LinearActivationFunction()), (10, new SigmoidActivationFunction()));

            Assert.Throws<Exception>(() => CreateBasicAndGateTrainer(net2.Object));

            var (net3, _) = CreateMockNetwork(1, (2, new LinearActivationFunction()), (10, new SigmoidActivationFunction()));

            Assert.Throws<Exception>(() => CreateBasicAndGateTrainer(net3.Object));
        }

        [Fact]
        public void Epoch_and_iterations_props_return_valid_results()
        {
            var net = CreateNetwork(2, (1, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
            var trainer = CreateBasicAndGateTrainer(net);

            trainer.DoEpoch();
            trainer.Epochs.Should().Be(1);
            trainer.Iterations.Should().Be(0);
            
            trainer.DoIteration();
            trainer.Iterations.Should().Be(1);
            trainer.Epochs.Should().Be(1);
            
            trainer.ResetEpochs();
            trainer.Iterations.Should().Be(0);
            trainer.Epochs.Should().Be(0);
        }
    }
}