using System;
using FluentAssertions;
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
                PageSize = 16,
            }, new SupervisedSetVariableIndexes(new[] {0}, new[] {1}));

            _trainingSet = trainingData.sets.TrainingSet;
        }

        [Fact]
        public void SupervisedSet_provides_random_access_to_vector_sets()
        {
            _trainingSet.Input[0][0, 0].Should().Be(1);
            _trainingSet.Target[98][0, 0].Should().Be(99);
            _trainingSet.Input[13][0, 0].Should().Be(14);
            _trainingSet.Target[56][0, 0].Should().Be(57);
            _trainingSet.Input[0][0, 0].Should().Be(1);
        }


        [Fact]
        public void SupervisedSet_provides_sequential_access_to_vector_sets()
        {
            for (int i = 0; i < _trainingSet.Input.Count; i++)
            {
                _trainingSet.Input[i][0, 0].Should().Be(i + 1);
                _trainingSet.Target[i][0, 0].Should().Be(i + 1);
            }
        }

        [Fact]
        public void SupervisedSet_provides_sequential_access_to_vector_sets_with_return_to_begining()
        {
            for (int i = 0; i < _trainingSet.Input.Count; i++)
            {
                _trainingSet.Input[i][0, 0].Should().Be(i + 1);
                _trainingSet.Target[i][0, 0].Should().Be(i + 1);
            }

            for (int i = 0; i < _trainingSet.Input.Count; i++)
            {
                _trainingSet.Input[i][0, 0].Should().Be(i + 1);
                _trainingSet.Target[i][0, 0].Should().Be(i + 1);
            }
        }

        public void Dispose()
        {
            _trainingSet.Dispose();
        }
    }
}