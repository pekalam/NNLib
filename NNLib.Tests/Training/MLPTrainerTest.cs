using System;
using FluentAssertions;
using NNLib.Common;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class MLPTrainerTest
    {
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
            var net1 = TrainingTestUtils.CreateNetwork(1, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));

            Assert.Throws<Exception>(() => CreateBasicAndGateTrainer(net1));

            var net2 = TrainingTestUtils.CreateNetwork(2, (2, new LinearActivationFunction()), (10, new SigmoidActivationFunction()));

            Assert.Throws<Exception>(() => CreateBasicAndGateTrainer(net2));

            var net3 = TrainingTestUtils.CreateNetwork(1, (2, new LinearActivationFunction()), (10, new SigmoidActivationFunction()));

            Assert.Throws<Exception>(() => CreateBasicAndGateTrainer(net3));
        }

        [Fact]
        public void Epoch_and_iterations_props_return_valid_results()
        {
            var net = TrainingTestUtils.CreateNetwork(2, (1, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
            var trainer = CreateBasicAndGateTrainer(net);

            trainer.DoEpoch();
            trainer.Epochs.Should().Be(1);
            trainer.Iterations.Should().Be(4);
            
            trainer.DoIteration();
            trainer.Iterations.Should().Be(5);
            trainer.Epochs.Should().Be(1);
            
            trainer.Reset();
            trainer.Iterations.Should().Be(0);
            trainer.Epochs.Should().Be(0);
        }

        [Fact]
        public void Reset_resets_current_training_progress()
        {
            var net1 = TrainingTestUtils.CreateNetwork(2, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
            var trainer = CreateBasicAndGateTrainer(net1);
            trainer.DoIteration();

            trainer.Reset();

            trainer.Iterations.Should().Be(0);
            trainer.Epochs.Should().Be(0);
            (trainer.Algorithm as GradientDescentAlgorithm).BatchTrainer.CurrentBatch.Should().Be(0);
        }


        [Fact]
        public void TrainingSets_when_set_sets_new_batch_trainer()
        {
            var net1 = TrainingTestUtils.CreateNetwork(2, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
            var trainer = CreateBasicAndGateTrainer(net1);
            trainer.DoIteration();

            trainer.TrainingSets = new SupervisedTrainingSets(TrainingTestUtils.AndGateSet());

            trainer.Iterations.Should().Be(0);
            trainer.Epochs.Should().Be(0);
            (trainer.Algorithm as GradientDescentAlgorithm).BatchTrainer.CurrentBatch.Should().Be(0);
        }
    }
}