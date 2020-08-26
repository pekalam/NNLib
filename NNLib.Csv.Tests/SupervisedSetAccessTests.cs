using System;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using NNLib.Common;
using Xunit;

namespace NNLib.Csv.Tests
{
    public class SupervisedSetAccessTests : IDisposable
    {
        //class under test
        private readonly SupervisedSet _trainingSet;

        public SupervisedSetAccessTests()
        {
            var fileName = @"plik.csv";

            var trainingData = CsvFacade.LoadSets(fileName, new LinearDataSetDivider(), new DataSetDivisionOptions()
            {
                TrainingSetPercent = 100,
            }, new SupervisedSetVariableIndexes(new[] {0}, new[] {1}));

            _trainingSet = trainingData.sets.TrainingSet;
        }

        [Fact]
        public void SupervisedSet_provides_read_access_to_vector_sets()
        {
            for (int i = 0; i < _trainingSet.Input.Count; i++)
            {
                _trainingSet.Input[i][0, 0].Should().Be(i + 1);
                _trainingSet.Target[i][0, 0].Should().Be(i + 1);
            }
        }

        [Fact]
        public void SupervisedSet_provides_write_access_to_vector_sets()
        {
            for (int i = 0; i < _trainingSet.Input.Count; i++)
            {
                _trainingSet.Input[i] = Matrix<double>.Build.Dense(1,1,i*i);
                _trainingSet.Target[i] = Matrix<double>.Build.Dense(1, 1, i * i + 1);
            }

            for (int i = 0; i < _trainingSet.Input.Count; i++)
            {
                _trainingSet.Input[i][0, 0].Should().Be(i * i);
                _trainingSet.Target[i][0, 0].Should().Be(i * i + 1);
            }
        }

        //TODO rm dispose
        public void Dispose()
        {
            _trainingSet.Dispose();
        }
    }
}