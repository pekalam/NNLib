using System;
using FluentAssertions;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class BatchTrainerTests : TrainerTestBase
    {
        private readonly MLPNetwork _net;
        
        public BatchTrainerTests(ITestOutputHelper output) : base(output)
        {
            _net = CreateNetwork(2, (1, new SigmoidActivationFunction()));
        }

        private GradientDescentAlgorithm CreateBatchTrainer(GradientDescentParams parameters)
        {
            var algorithm = new GradientDescentAlgorithm(parameters);
            algorithm.Setup(TrainingTestUtils.AndGateSet(), _net, new QuadraticLossFunction());
            return algorithm;
        }

        [Fact]
        public void When_constructed_has_valid_state()
        {
            var learningParams = new GradientDescentParams()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 4,
            };
            
            var trainer = CreateBatchTrainer(learningParams);
            trainer.Iterations.Should().Be(0);
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(1);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);
        }

        // [Fact]
        // public void When_parameters_change_props_are_correct()
        // {
        //     var learningParams = new GradientDescentParams()
        //     {
        //         LearningRate = 0.1,
        //         Momentum = 0.9,
        //     };
        //
        //     var trainer = CreateBatchTrainer(learningParams, new BatchParams(){BatchSize = 4});
        //
        //     trainer.Parameters = new BatchParams(){BatchSize = 1};
        //
        //     trainer.Iterations.Should().Be(0);
        //     trainer.BatchTrainer.IterationsPerEpoch.Should().Be(4);
        //     trainer.BatchTrainer.CurrentBatch.Should().Be(0);
        // }

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
            var trainer = CreateBatchTrainer(learningParams);

            trainer.Iterations.Should().Be(0);
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(1);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);

            var result = trainer.DoIteration();
            
            result.Should().BeTrue();
            trainer.Iterations.Should().Be(1);
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(1);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);
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
            var trainer = CreateBatchTrainer(learningParams);

            trainer.Iterations.Should().Be(0);
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(2);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);

            var result = trainer.DoIteration();
            result.Should().BeFalse();
            trainer.Iterations.Should().Be(1);
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(2);
            trainer.BatchTrainer.CurrentBatch.Should().Be(1);

            result = trainer.DoIteration();
            result.Should().BeTrue();
            trainer.Iterations.Should().Be(2);
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(2);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);
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
            var trainer = CreateBatchTrainer(learningParams);

            trainer.Iterations.Should().Be(0);
            trainer.BatchTrainer.IterationsPerEpoch.Should().Be(4);
            trainer.BatchTrainer.CurrentBatch.Should().Be(0);

            for (int i = 0; i < trainer.BatchTrainer.TrainingSet.Input.Count; i++)
            {

                trainer.Iterations.Should().Be(i % trainer.BatchTrainer.TrainingSet.Input.Count);
                trainer.BatchTrainer.IterationsPerEpoch.Should().Be(4);
                trainer.BatchTrainer.CurrentBatch.Should().Be(i % trainer.BatchTrainer.TrainingSet.Input.Count);
                var result = trainer.DoIteration();

                if (i == trainer.BatchTrainer.TrainingSet.Input.Count - 1)
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