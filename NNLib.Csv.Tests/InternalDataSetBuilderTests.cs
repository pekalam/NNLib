using System;
using FluentAssertions;
using NNLib.Common;
using Xunit;

namespace NNLib.Csv.Tests
{


    public class CsvVectorSetInvariantsTestClass
    {
        [Fact]
        public void ctor_when_set_count_is_zero_throws()
        {
            Assert.Throws<ArgumentException>(() => new CsvFileVectorSet(null, 0, false));
        }
    }


    public class InternalDataSetBuilderTests
    {
        [Fact]
        public void CreatePartitionedDataSets_when_pageSize_eq_row_size_returns_valid_nummber_of_file_parts()
        {
            var builder =
                new InternalDataSetBuilder(new LinearDataSetDivider());

            var setInfo = builder.CreatePartitionedDataSets(@"plik.csv", new DataSetDivisionOptions()
            {
                TrainingSetPercent = 100,
                PageSize = 16
            });

            setInfo.Length.Should().Be(1);
            setInfo[0].SetSize.Should().Be(99);
            setInfo[0].FileParts.Count.Should().Be(99);

            long prevOffset = -1;
            long prevEnd = 0;
            foreach (var filePart in setInfo[0].FileParts)
            {
                filePart.Offset.Should().BeGreaterThan(prevOffset);
                filePart.End.Should().BeGreaterThan(prevEnd);

                filePart.DataItems.Should().Be(1);
                prevOffset = filePart.Offset;
                prevEnd = filePart.End;
            }
        }
    }
}