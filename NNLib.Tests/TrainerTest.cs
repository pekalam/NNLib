using NNLib;
using NNLib.ActivationFunction;
using System;
using Xunit;

namespace UnitTests
{
    public class TrainerTest : TrainerTestBase
    {
        private MLPTrainer CreateBasicAndGateTrainer(MLPNetwork net)
        {
            return new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                new GradientDescentParams()
                {
                    LearningRate = 0.9,
                    Momentum = 0.1
                }, new QuadraticLossFunction());
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

    }
}