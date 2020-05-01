using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using FluentAssertions;
using NNLib;
using NNLib.ActivationFunction;
using Xunit;
using Xunit.Sdk;
using M = MathNet.Numerics.LinearAlgebra.Matrix<double>;

namespace UnitTests
{
    public static class TrainingTestUtils
    {
        public static SupervisedSet AndGateSet()
        {
            var input = new[]
            {
                new []{0d,0d},
                new []{0d,1d},
                new []{1d,0d},
                new []{1d,1d},
            };

            var expected = new[]
            {
                new []{0d},
                new []{0d},
                new []{0d},
                new []{1d},
            };

            return SupervisedSet.FromArrays(input, expected);
        }
    }

    public abstract class VectorSetInvariantsTest<T> where T : IVectorSet
    {
        protected abstract T CreateZeroSizeVectorSet();

        [Fact]
        public void ctor_when_set_count_is_zero_throws()
        {
            Assert.Throws<ArgumentException>(() => CreateZeroSizeVectorSet());
        } 
    }

    public class DefaultVectorSetInvariantsTest : VectorSetInvariantsTest<DefaultVectorSet>
    {
        protected override DefaultVectorSet CreateZeroSizeVectorSet()
        {
            return new DefaultVectorSet(new List<M>());
        }
    }

    public class GradientDescentTests : TrainerTestBase
    {
        MLPNetwork net;

        public GradientDescentTests()
        {
            net = CreateNetwork(2, (1, new SigmoidActivationFunction()));
        }

        private GradientDescent CreateLearningMethod(GradientDescentLearningParameters parameters)
        {
            var method = new GradientDescent(parameters);
            method.TrainingSet = TrainingTestUtils.AndGateSet();
            return method;
        }

        [Fact]
        public void When_constructed_has_valid_state()
        {
            var learningParams = new GradientDescentLearningParameters()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 4
            };

            var method = CreateLearningMethod(learningParams);
            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(1);
            method.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void When_parameters_change_state_changes()
        {
            var learningParams = new GradientDescentLearningParameters()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 4
            };

            var method = CreateLearningMethod(learningParams);

            var learningParams2 = new GradientDescentLearningParameters()
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
            var learningParams = new GradientDescentLearningParameters()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 10
            };

            Assert.Throws<ArgumentException>(() => CreateLearningMethod(learningParams));
        }

        [Fact]
        public void Batch_training_iteration_returns_epoch_result()
        {
            var learningParams = new GradientDescentLearningParameters()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 4
            };
            var method = CreateLearningMethod(learningParams);

            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(1);
            method.CurrentBatch.Should().Be(0);

            var result = method.DoIteration(net, new QuadraticLossFunction());
            
            result.Should().NotBeNull();
            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(1);
            method.CurrentBatch.Should().Be(0);
        }

        [Fact]
        public void Mini_Batch_training_iterations_returns_epoch_result()
        {
            var learningParams = new GradientDescentLearningParameters()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 2
            };
            var method = CreateLearningMethod(learningParams);

            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(2);
            method.CurrentBatch.Should().Be(0);

            var result = method.DoIteration(net, new QuadraticLossFunction());
            result.Should().BeNull();
            method.Iterations.Should().Be(1);
            method.IterationsPerEpoch.Should().Be(2);
            method.CurrentBatch.Should().Be(1);

            result = method.DoIteration(net, new QuadraticLossFunction());
            result.Should().NotBeNull();
            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(2);
            method.CurrentBatch.Should().Be(0);
        }


        [Fact]
        public void Online_training_iterations_returns_epoch_result()
        {
            var learningParams = new GradientDescentLearningParameters()
            {
                LearningRate = 0.1,
                Momentum = 0.9,
                BatchSize = 1
            };
            var method = CreateLearningMethod(learningParams);

            method.Iterations.Should().Be(0);
            method.IterationsPerEpoch.Should().Be(4);
            method.CurrentBatch.Should().Be(0);

            for (int i = 0; i < method.TrainingSet.Input.Count; i++)
            {

                method.Iterations.Should().Be(i % method.TrainingSet.Input.Count);
                method.IterationsPerEpoch.Should().Be(4);
                method.CurrentBatch.Should().Be(i % method.TrainingSet.Input.Count);
                var result = method.DoIteration(net, new QuadraticLossFunction());

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

    public class SupervisedSetTests
    {
        [Fact]
        public void ctor_when_vector_sets_differ_in_length_throws()
        {
            Assert.Throws<ArgumentException>(() => new SupervisedSet(new DefaultVectorSet(new List<M>()
            {
                M.Build.Random(2,1), M.Build.Random(2,1),
            }), new DefaultVectorSet(new List<M>()
            {
                M.Build.Random(2,1),
            })));
        }
    }
}