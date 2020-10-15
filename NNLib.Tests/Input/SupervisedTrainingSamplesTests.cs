using System;
using System.Collections.Generic;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;
using NNLib.Data;
using Xunit;

namespace NNLib.Tests
{
    public class SupervisedTrainingSamplesTests
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

            var samples = Data.SupervisedTrainingSamples.FromArrays(input, expected);

            samples.Input.Count.Should().Be(4);
            for (int i = 0; i < samples.Input.Count; i++)
            {
                samples.Input[i].ColumnCount.Should().Be(1);
                samples.Input[i].RowCount.Should().Be(2);
                for (int j = 0; j < samples.Input[i].RowCount; j++)
                {
                    samples.Input[i][j, 0].Should().Be(input[i][j]);
                }
            }

            samples.Target.Count.Should().Be(4);
            for (int i = 0; i < samples.Target.Count; i++)
            {
                samples.Target[i].ColumnCount.Should().Be(1);
                samples.Target[i].RowCount.Should().Be(2);
                for (int j = 0; j < samples.Target[i].RowCount; j++)
                {
                    samples.Target[i][j, 0].Should().Be(expected[i][j]);
                }
            }
        }


        [Fact]
        public void ctor_when_vector_sets_differ_in_length_throws()
        {
            Assert.Throws<ArgumentException>(() => new Data.SupervisedTrainingSamples(new DefaultVectorSet(new List<Matrix<double>>()
            {
                Matrix<double>.Build.Random(2,1), Matrix<double>.Build.Random(2,1),
            }), new DefaultVectorSet(new List<Matrix<double>>()
            {
                Matrix<double>.Build.Random(2,1),
            })));
        }
    }
}