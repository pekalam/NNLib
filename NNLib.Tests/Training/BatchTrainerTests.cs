using System;
using FluentAssertions;
using NNLib.Common;
using Xunit;
using static NNLib.Tests.TrainingTestUtils;

namespace NNLib.Tests
{
    public class BatchTrainerTests
    {
        private readonly MLPNetwork _net;
        private SupervisedSet _set;

        public BatchTrainerTests()
        {
            _net = CreateNetwork(2, (1, new SigmoidActivationFunction()));
        }

        private GradientDescentAlgorithm CreateBatchTrainer(GradientDescentParams parameters)
        {
            var algorithm = new GradientDescentAlgorithm(parameters);
            algorithm.Setup(_set = AndGateSet(), _net, new QuadraticLossFunction());
            return algorithm;
        }

        [Fact]
        public void When_constructed_has_valid_state()
        {
            var learningParams = new GradientDescentParams
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 4,
            };
            
            var trainer = CreateBatchTrainer(learningParams);
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(1);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void When_invalid_parameters_throws()
        {
            var learningParams = new GradientDescentParams
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 10
            };

            Assert.Throws<ArgumentException>(() => CreateBatchTrainer(learningParams));
        }

        [Fact]
        public void Batch_training_iteration_returns_epoch_result()
        {
            var learningParams = new GradientDescentParams
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 4
            };
            var trainer = CreateBatchTrainer(learningParams);

            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(1);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);

            var result = trainer.DoIteration();
            
            result.Should().BeTrue();
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(1);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void Mini_Batch_training_iterations_returns_epoch_result()
        {
            var learningParams = new GradientDescentParams
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 2
            };
            var trainer = CreateBatchTrainer(learningParams);

            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(2);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);

            var result = trainer.DoIteration();
            result.Should().BeFalse();
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(2);
            trainer.BatchTrainer.CurrentBatch.Should().Be(1);

            result = trainer.DoIteration();
            result.Should().BeTrue();
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(2);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);
        }


        [Fact]
        public void Online_training_iterations_returns_epoch_result()
        {
            var learningParams = new GradientDescentParams
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 1
            };
            var trainer = CreateBatchTrainer(learningParams);

            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(4);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);

            for (int i = 0; i < _set.Input.Count; i++)
            {

                trainer.BatchTrainer.IterationsPerEpoch.Should().Be(4);
                trainer.BatchTrainer.CurrentBatch.Should().Be(i % _set.Input.Count);
                var result = trainer.DoIteration();

                if (i == _set.Input.Count - 1)
                {
                    result.Should().BeTrue();
                }
                else
                {
                    result.Should().BeFalse();
                }
            }
        }
    }
}