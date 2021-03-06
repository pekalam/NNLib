using System;
using System.Collections.Generic;
using System.Linq;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using Xunit;

namespace NNLib.Tests.Data
{
    public class ConcatenatedVectorSetTests
    {
        [Fact]
        public void f()
        {
            var v1 = SupervisedTrainingSamples.FromArrays(Enumerable.Range(0, 5).Select(i => new double[] { i }).ToArray(),
                                                                        Enumerable.Range(0, 5).Select(i => new double[] { i }).ToArray());
            var v2 = SupervisedTrainingSamples.FromArrays(Enumerable.Range(0, 2).Select(i => new double[] { i+5 }).ToArray(),
                                                                        Enumerable.Range(0, 2).Select(i => new double[] { i+5 }).ToArray());

            var concatenated = new ConcatenatedVectorSet();
            concatenated.AddVectorSet(v1.Input);
            concatenated.AddVectorSet(v2.Input);

            for (int i = 0; i < 7; i++)
            {
                concatenated[i][0, 0].Should().Be(i);
            }
        }
    }

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

            var target = new double[][]
            {
                new[] {2d, 2d},
                new[] {3d, 3d},
                new[] {4d, 5d},
                new[] {7d, 6d},
            };

            var samples = NNLib.Data.SupervisedTrainingSamples.FromArrays(input, target);

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
                    samples.Target[i][j, 0].Should().Be(target[i][j]);
                }
            }
        }


        [Fact]
        public void ctor_when_vector_sets_differ_in_length_throws()
        {
            Assert.Throws<ArgumentException>(() => new NNLib.Data.SupervisedTrainingSamples(new DefaultVectorSet(new List<Matrix<double>>()
            {
                Matrix<double>.Build.Random(2,1), Matrix<double>.Build.Random(2,1),
            }), new DefaultVectorSet(new List<Matrix<double>>()
            {
                Matrix<double>.Build.Random(2,1),
            })));
        }


        [Fact]
        public void ReadAllSamples_returns_matrices_with_all_samples()
        {
            var input = new double[][]
            {
                new[] {0d, 0d},
                new[] {0d, 1d},
                new[] {1d, 0d},
                new[] {1d, 1d},
            };

            var target = new double[][]
            {
                new[] {2d},
                new[] {3d},
                new[] {4d},
                new[] {7d},
            };

            var samples = NNLib.Data.SupervisedTrainingSamples.FromArrays(input, target);


            var (I,T) = (samples.ReadInputSamples(), samples.ReadTargetSamples());

            I.RowCount.Should().Be(2);
            I.ColumnCount.Should().Be(4);

            for (int i = 0; i < I.ColumnCount; i++)
            {
                for (int j = 0; j < I.RowCount; j++)
                {
                    I[j, i].Should().Be(input[i][j]);
                }
            }

            T.RowCount.Should().Be(1);
            T.ColumnCount.Should().Be(4);


            for (int i = 0; i < T.ColumnCount; i++)
            {
                for (int j = 0; j < T.RowCount; j++)
                {
                    T[j, i].Should().Be(target[i][j]);
                }
            }
        }
    }
}