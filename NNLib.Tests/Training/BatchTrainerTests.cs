using System;
using FluentAssertions;
using NNLib.ActivationFunction;
using NNLib.Data;
using NNLib.LossFunction;
using NNLib.MLP;
using NNLib.Training.GradientDescent;
using Xunit;
using static NNLib.Tests.Training.TrainingTestUtils;

namespace NNLib.Tests.Training
{
    public class BatchTrainerTests
    {
        private readonly MLPNetwork _net;
        private NNLib.Data.SupervisedTrainingSamples _trainingSamples;

        public BatchTrainerTests()
        {
            _net = CreateNetwork(2, (1, new SigmoidActivationFunction()));
        }

        private GradientDescentAlgorithm CreateBatchGd(GradientDescentParams parameters)
        {
            _trainingSamples = AndGateSet();
            var algorithm = new GradientDescentAlgorithm(parameters);
            var data = new SupervisedTrainingData(_trainingSamples);
            var lossFunction = new QuadraticLossFunction();
            lossFunction.InitializeMemory(_net.Layers[^1], data.TrainingSet);
            algorithm.Setup(_trainingSamples, new LoadedSupervisedTrainingData(data), _net, lossFunction);
            _net.InitializeMemoryForData(data.TrainingSet);
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
            
            var algorithm = CreateBatchGd(learningParams);
            algorithm.IterationsPerEpoch.Should().Be(1);
            algorithm.BatchIterations.Should().Be(0);
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

            Assert.Throws<ArgumentException>(() => CreateBatchGd(learningParams));
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
            var algorithm = CreateBatchGd(learningParams);

            algorithm.IterationsPerEpoch.Should().Be(1);
            algorithm.BatchIterations.Should().Be(0);

            var result = algorithm.DoIteration();
            
            result.Should().BeTrue();
            algorithm.BatchIterations.Should().Be(1);

            result = algorithm.DoIteration();
            result.Should().BeTrue();
            algorithm.BatchIterations.Should().Be(1);
        }

        [Theory]
        [InlineData(2)]
        [InlineData(3)]
        public void Mini_Batch_training_iterations_returns_epoch_result(int batchSize)
        {
            var learningParams = new GradientDescentParams
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = batchSize
            };
            var algorithm = CreateBatchGd(learningParams);

            algorithm.IterationsPerEpoch.Should().Be(2);
            algorithm.BatchIterations.Should().Be(0);

            var result = algorithm.DoIteration();
            result.Should().BeFalse();
            algorithm.BatchIterations.Should().Be(1);

            result = algorithm.DoIteration();
            result.Should().BeTrue();
            algorithm.BatchIterations.Should().Be(2);

            result = algorithm.DoIteration();
            result.Should().BeFalse();
            algorithm.BatchIterations.Should().Be(1);
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
            var trainer = CreateBatchGd(learningParams);

            trainer.IterationsPerEpoch.Should().Be(4);
            trainer.BatchIterations.Should().Be(0);

            for (int i = 0; i < _trainingSamples.Input.Count; i++)
            {
                trainer.BatchIterations.Should().Be(i % _trainingSamples.Input.Count);
                var result = trainer.DoIteration();
                trainer.BatchIterations.Should().Be(i % _trainingSamples.Input.Count + 1);


                if (i == _trainingSamples.Input.Count - 1)
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