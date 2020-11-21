using NNLib.Data;
using System;
using Xunit;

namespace NNLib.Csv.Tests
{
    public class DataSetDivisionOptionsTest
    {
        [Fact]
        public void Properties_when_percentage_sum_gt_100_throws()
        {
            Assert.Throws<ArgumentException>(() => new DataSetDivisionOptions()
            {
                TrainingSetPercent = 100,
                TestSetPercent = 20,
                ValidationSetPercent = 30,
            });
        }

        [Fact]
        public void Properties_when_percentage_lt_0_throws()
        {
            Assert.Throws<ArgumentException>(() => new DataSetDivisionOptions()
            {
                TrainingSetPercent = -1,
                TestSetPercent = 20,
                ValidationSetPercent = 30,
            });

            Assert.Throws<ArgumentException>(() => new DataSetDivisionOptions()
            {
                TrainingSetPercent = 1,
                TestSetPercent = -20,
                ValidationSetPercent = 30,
            });


            Assert.Throws<ArgumentException>(() => new DataSetDivisionOptions()
            {
                TrainingSetPercent = 1,
                TestSetPercent = 20,
                ValidationSetPercent = -30,
            });
        }
    }
}