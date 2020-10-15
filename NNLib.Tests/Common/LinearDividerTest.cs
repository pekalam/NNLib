using System;
using System.Collections.Generic;
using FluentAssertions;
using NNLib.Common;
using NNLib.Data;
using Xunit;

namespace NNLib.Tests.Common
{
    public class LinearDividerTest
    {
        [Fact]
        public void Divide_when_valid_params_divides_set()
        {
            var linearDivider = new LinearDataSetDivider();

            var positions = new List<long>()
            {
                1, 2, 3, 4
            };

            var divisions = linearDivider.Divide(positions, new DataSetDivisionOptions()
            {
                TrainingSetPercent = 100,
                ValidationSetPercent = 0,
                TestSetPercent = 0,
            });

            divisions.Length.Should().Be(1);

            var div0 = divisions[0];
            div0.setType.Should().Be(DataSetType.Training);
            div0.positions.Count.Should().Be(4);
            div0.positions.Should().BeEquivalentTo(positions);
        }

        [Fact]
        public void Divide_when_valid_params_divides_set_into_3_sets()
        {
            var linearDivider = new LinearDataSetDivider();

            var positions = new List<long>()
            {
                1, 2, 3, 4, 5, 6
            };

            var divisions = linearDivider.Divide(positions, new DataSetDivisionOptions()
            {
                TrainingSetPercent = 33,
                ValidationSetPercent = 33,
                TestSetPercent = 33,
            });

            divisions.Length.Should().Be(3);

            var div0 = divisions[0];
            div0.setType.Should().Be(DataSetType.Training);
            div0.positions.Count.Should().Be(2);
            div0.positions.Should().BeEquivalentTo(1, 2);

            var div1 = divisions[1];
            div1.setType.Should().Be(DataSetType.Validation);
            div1.positions.Count.Should().Be(2);
            div1.positions.Should().BeEquivalentTo(3, 4);

            var div2 = divisions[2];
            div2.setType.Should().Be(DataSetType.Test);
            div2.positions.Count.Should().Be(2);
            div2.positions.Should().BeEquivalentTo(5, 6);
        }


        [Fact]
        public void Divide_when_valid_params_divides_set_into_2_sets()
        {
            var linearDivider = new LinearDataSetDivider();

            var positions = new List<long>()
            {
                1, 2, 3, 4, 5, 6
            };

            var divisions = linearDivider.Divide(positions, new DataSetDivisionOptions()
            {
                TrainingSetPercent = 50,
                ValidationSetPercent = 50,
                TestSetPercent = 0,
            });

            divisions.Length.Should().Be(2);

            var div0 = divisions[0];
            div0.setType.Should().Be(DataSetType.Training);
            div0.positions.Count.Should().Be(3);
            div0.positions.Should().BeEquivalentTo(1, 2, 3);

            var div1 = divisions[1];
            div1.setType.Should().Be(DataSetType.Validation);
            div1.positions.Count.Should().Be(3);
            div1.positions.Should().BeEquivalentTo(4, 5, 6);
        }


        [Fact]
        public void Divide_when_empty_positions_throws()
        {
            var linearDivider = new LinearDataSetDivider();


            Assert.Throws<ArgumentException>(() => linearDivider.Divide(new List<long>(), new DataSetDivisionOptions()
            {
                TrainingSetPercent = 100
            }));
        }
    }
}