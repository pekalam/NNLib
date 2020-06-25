using System;
using FluentAssertions;
using NNLib.ActivationFunction;
using NNLib.Training;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class BatchTrainerTests : TrainerTestBase
    {
        private readonly MLPNetwork _net;
        private AlgorithmBase _algorithm;
        
        public BatchTrainerTests(ITestOutputHelper output) : base(output)
        {
            _net = CreateNetwork(2, (1, new SigmoidActivationFunction()));
        }

        private BatchTrainer CreateBatchTrainer(GradientDescentParams parameters, BatchParams batchParams)
        {
            _algorithm = new GradientDescentAlgorithm(parameters);
            var trainer = new BatchTrainer(batchParams)
            {
                TrainingSet = TrainingTestUtils.AndGateSet()
            };
            return trainer;
        }

        [Fact]
        public void When_constructed_has_valid_state()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
            };
            
            var trainer = CreateBatchTrainer(learningParams, new BatchParams(){BatchSize = 4});
            trainer.Iterations.Should().Be(0);
            trainer.IterationsPerEpoch.Should().Be(1);
            trainer.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void When_parameters_change_props_are_correct()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
            };

            var trainer = CreateBatchTrainer(learningParams, new BatchParams(){BatchSize = 4});

            trainer.Parameters = new BatchParams(){BatchSize = 1};

            trainer.Iterations.Should().Be(0);
            trainer.IterationsPerEpoch.Should().Be(4);
            trainer.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void When_invalid_parameters_throws()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
            };

            Assert.Throws<ArgumentException>(() => CreateBatchTrainer(learningParams, new BatchParams(){BatchSize = 10}));
        }

        [Fact]
        public void Batch_training_iteration_returns_epoch_result()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
            };
            var trainer = CreateBatchTrainer(learningParams, new BatchParams(){BatchSize = 4});

            trainer.Iterations.Should().Be(0);
            trainer.IterationsPerEpoch.Should().Be(1);
            trainer.CurrentBatch.Should().Be(0);

            var result = trainer.DoIteration(_net, new QuadraticLossFunction(), _algorithm);
            
            result.Should().NotBeNull();
            trainer.Iterations.Should().Be(0);
            trainer.IterationsPerEpoch.Should().Be(1);
            trainer.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void Mini_Batch_training_iterations_returns_epoch_result()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
            };
            var trainer = CreateBatchTrainer(learningParams, new BatchParams(){BatchSize = 2});

            trainer.Iterations.Should().Be(0);
            trainer.IterationsPerEpoch.Should().Be(2);
            trainer.CurrentBatch.Should().Be(0);

            var result = trainer.DoIteration(_net, new QuadraticLossFunction(), _algorithm);
            result.Should().BeNull();
            trainer.Iterations.Should().Be(1);
            trainer.IterationsPerEpoch.Should().Be(2);
            trainer.CurrentBatch.Should().Be(1);

            result = trainer.DoIteration(_net, new QuadraticLossFunction(), _algorithm);
            result.Should().NotBeNull();
            trainer.Iterations.Should().Be(0);
            trainer.IterationsPerEpoch.Should().Be(2);
            trainer.CurrentBatch.Should().Be(0);
        }


        [Fact]
        public void Online_training_iterations_returns_epoch_result()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
            };
            var trainer = CreateBatchTrainer(learningParams, new BatchParams(){BatchSize = 1});

            trainer.Iterations.Should().Be(0);
            trainer.IterationsPerEpoch.Should().Be(4);
            trainer.CurrentBatch.Should().Be(0);

            for (int i = 0; i < trainer.TrainingSet.Input.Count; i++)
            {

                trainer.Iterations.Should().Be(i % trainer.TrainingSet.Input.Count);
                trainer.IterationsPerEpoch.Should().Be(4);
                trainer.CurrentBatch.Should().Be(i % trainer.TrainingSet.Input.Count);
                var result = trainer.DoIteration(_net, new QuadraticLossFunction(), _algorithm);

                if (i == trainer.TrainingSet.Input.Count - 1)
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