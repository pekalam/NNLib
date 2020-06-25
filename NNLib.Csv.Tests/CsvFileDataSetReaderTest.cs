using System;
using System.Collections.Generic;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using Moq;
using NNLib.Common;
using Xunit;

namespace NNLib.Csv.Tests
{
    public class CsvFileDataSetReaderTest
    {
        [Fact]
        public void ReadAt_sets_valid_read_ranges()
        {
            var stubInternalReader = new Mock<ICsvInternalReader>(MockBehavior.Strict);

            stubInternalReader.Setup(
                    f => f.ReadVectorSets(It.Ref<FilePart>.IsAny, It.IsAny<SupervisedSetVariableIndexes>()))
                .Returns(() =>
                {
                    return new List<(Matrix<double> input, Matrix<double> target)>()
                    {
                        (Matrix<double>.Build.Random(1, 1), Matrix<double>.Build.Random(1, 1))
                    };
                });


            var testFileParts = new List<FilePart>()
            {
                new FilePart(0, 10, 10),
                new FilePart(11, 20, 10),
            };

            var reader = new CsvFileDataSetReader(stubInternalReader.Object,
                new SupervisedSetVariableIndexes(new[] { 0 }, new[] { 1 }),
                new DataSetInfo(testFileParts, 20, 10, DataSetType.Training, new[] { "x", "y" }));



            //first part
            for (int i = 0; i < 10; i++)
            {
                for (int j = i; j >= 0; j--)
                {
                    _ = reader.ReadAt(false, j);
                    reader.ReadRange.start.Should().Be(0);
                    reader.ReadRange.end.Should().Be(10);
                }

            }

            //second part
            for (int i = 10; i < 20; i++)
            {
                for (int j = i; j >= 10; j--)
                {
                    _ = reader.ReadAt(false, j);
                    reader.ReadRange.start.Should().Be(10);
                    reader.ReadRange.end.Should().Be(20);
                }

            }

            //first part
            for (int i = 0; i < 10; i++)
            {
                for (int j = i; j >= 0; j--)
                {
                    _ = reader.ReadAt(false, j);
                    reader.ReadRange.start.Should().Be(0);
                    reader.ReadRange.end.Should().Be(10);
                }

            }


            Assert.Throws<ArgumentOutOfRangeException>(() => reader.ReadAt(false, 20));
        }
    }
}