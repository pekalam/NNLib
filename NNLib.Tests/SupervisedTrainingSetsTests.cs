using FluentAssertions;
using NNLib;
using System;
using Xunit;

namespace UnitTests
{
    public class SupervisedTrainingSetsTests
    {
        [Fact]
        public void FromArrays_builds_set_with_default_vector_set()
        {
            var input = new double[][]
            {
                new[] {0d, 0d},
                new[] {0d, 1d},
                new[] {1d, 0d},
                new[] {1d, 1d},
            };

            var expected = new double[][]
            {
                new[] {2d, 2d},
                new[] {3d, 3d},
                new[] {4d, 5d},
                new[] {7d, 6d},
            };

            var trainingData = SupervisedSet.FromArrays(input, expected);

            trainingData.Input.Count.Should().Be(4);
            for (int i = 0; i < trainingData.Input.Count; i++)
            {
                trainingData.Input[i].ColumnCount.Should().Be(1);
                trainingData.Input[i].RowCount.Should().Be(2);
                for (int j = 0; j < trainingData.Input[i].RowCount; j++)
                {
                    trainingData.Input[i][j, 0].Should().Be(input[i][j]);
                }
            }

            trainingData.Target.Count.Should().Be(4);
            for (int i = 0; i < trainingData.Target.Count; i++)
            {
                trainingData.Target[i].ColumnCount.Should().Be(1);
                trainingData.Target[i].RowCount.Should().Be(2);
                for (int j = 0; j < trainingData.Target[i].RowCount; j++)
                {
                    trainingData.Target[i][j, 0].Should().Be(expected[i][j]);
                }
            }
        }

        [Fact]
        public void ctor_when_training_set_null_throws()
        {
            Assert.Throws<NullReferenceException>(() =>
                new SupervisedTrainingSets(null));
        }
    }
}