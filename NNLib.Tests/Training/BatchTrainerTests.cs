using System;
using FluentAssertions;
using NNLib.ActivationFunction;
using Xunit;

namespace NNLib.Tests
{
    public class BatchTrainerTests : TrainerTestBase
    {
        private readonly MLPNetwork _net;

        public BatchTrainerTests()
        {
            _net = CreateNetwork(2, (1, new SigmoidActivationFunction()));
        }

        private BatchTrainer CreateBatchTrainer(GradientDescentParams parameters)
        {
            var method = new BatchTrainer(new GradientDescentAlgorithm(_net, parameters))
            {
                TrainingSet = TrainingTestUtils.AndGateSet()
            };
            return method;
        }

        [Fact]
        public void When_constructed_has_valid_state()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 4
            };

            var method = CreateBatchTrainer(learningParams);
            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(1);
            method.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void When_parameters_change_props_are_correct()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 4
            };

            var method = CreateBatchTrainer(learningParams);

            var learningParams2 = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 1
            };

            method.Parameters = learningParams2;

            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(4);
            method.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void When_invalid_parameters_throws()
        {
            var learningParams = new GradientDescentParams()
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
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 4
            };
            var method = CreateBatchTrainer(learningParams);

            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(1);
            method.CurrentBatch.Should().Be(0);

            var result = method.DoIteration(_net, new QuadraticLossFunction());
            
            result.Should().NotBeNull();
            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(1);
            method.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void Mini_Batch_training_iterations_returns_epoch_result()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 2
            };
            var method = CreateBatchTrainer(learningParams);

            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(2);
            method.CurrentBatch.Should().Be(0);

            var result = method.DoIteration(_net, new QuadraticLossFunction());
            result.Should().BeNull();
            method.Iterations.Should().Be(1);
            method.IterationsPerEpoch.Should().Be(2);
            method.CurrentBatch.Should().Be(1);

            result = method.DoIteration(_net, new QuadraticLossFunction());
            result.Should().NotBeNull();
            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(2);
            method.CurrentBatch.Should().Be(0);
        }


        [Fact]
        public void Online_training_iterations_returns_epoch_result()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 1
            };
            var method = CreateBatchTrainer(learningParams);

            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(4);
            method.CurrentBatch.Should().Be(0);

            for (int i = 0; i < method.TrainingSet.Input.Count; i++)
            {

                method.Iterations.Should().Be(i % method.TrainingSet.Input.Count);
                method.IterationsPerEpoch.Should().Be(4);
                method.CurrentBatch.Should().Be(i % method.TrainingSet.Input.Count);
                var result = method.DoIteration(_net, new QuadraticLossFunction());

                if (i == method.TrainingSet.Input.Count - 1)
                {
                    result.Should().NotBeNull();
                }
                else
                {
                    result.Should().BeNull();
                }
            }
        }
    }
}