using System;
using FluentAssertions;
using NNLib.ActivationFunction;
using NNLib.Common;
using NNLib.Data;
using NNLib.LossFunction;
using NNLib.MLP;
using NNLib.Training.GradientDescent;
using Xunit;
using static NNLib.Tests.TrainingTestUtils;

namespace NNLib.Tests
{
    public class BatchTrainerTests
    {
        private readonly MLPNetwork _net;
        private Data.SupervisedTrainingSamples _trainingSamples;

        public BatchTrainerTests()
        {
            _net = CreateNetwork(2, (1, new SigmoidActivationFunction()));
        }

        private GradientDescentAlgorithm CreateBatchTrainer(GradientDescentParams parameters)
        {
            _trainingSamples = AndGateSet();
            var algorithm = new GradientDescentAlgorithm(parameters);
            var data = new SupervisedTrainingData(_trainingSamples);
            var lossFunction = new QuadraticLossFunction();
            lossFunction.InitializeMemory(_net.Layers[^1], data.TrainingSet);
            algorithm.Setup(_trainingSamples, new LoadedSupervisedTrainingData(data), _net, lossFunction);
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
            trainer.IterationsPerEpoch.Should().Be(1);
            trainer.BatchIterations.Should().Be(0);
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

            trainer.IterationsPerEpoch.Should().Be(1);
            trainer.BatchIterations.Should().Be(0);

            var result = trainer.DoIteration();
            
            result.Should().BeTrue();
            trainer.IterationsPerEpoch.Should().Be(1);
            trainer.BatchIterations.Should().Be(0);
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

            trainer.IterationsPerEpoch.Should().Be(2);
            trainer.BatchIterations.Should().Be(0);

            var result = trainer.DoIteration();
            result.Should().BeFalse();
            trainer.IterationsPerEpoch.Should().Be(2);
            trainer.BatchIterations.Should().Be(1);

            result = trainer.DoIteration();
            result.Should().BeTrue();
            trainer.IterationsPerEpoch.Should().Be(2);
            trainer.BatchIterations.Should().Be(0);
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

            trainer.IterationsPerEpoch.Should().Be(4);
            trainer.BatchIterations.Should().Be(0);

            for (int i = 0; i < _trainingSamples.Input.Count; i++)
            {

                trainer.IterationsPerEpoch.Should().Be(4);
                trainer.BatchIterations.Should().Be(i % _trainingSamples.Input.Count);
                var result = trainer.DoIteration();

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