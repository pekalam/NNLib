using FluentAssertions;
using NNLib.Data;
using System;
using System.Collections.Generic;
using Xunit;

namespace NNLib.Csv.Tests
{
    public class RandomDataSetDividerTest
    {
        [Fact]
        public void When_training_set_percent_is_100_set_should_not_be_in_random_order()
        {
            var divider = new RandomDataSetDivider();
            var positions = new List<long>()
            {
                1, 2, 3, 4,5,6,7,8,9,10
            };

            var result = divider.Divide(positions, new DataSetDivisionOptions()
            {
                TrainingSetPercent = 100,
            });

            result.Length.Should().Be(1);
            result[0].setType.Should().Be(DataSetType.Training);

            //should not be random
            for (int i = 0; i < result[0].positions.Count; i++)
            {
                result[0].positions[i].Should().Be(i + 1);
            }
        }

        [Fact]
        public void When_valid_division_params_returns_randomized_sets()
        {
            var divider = new RandomDataSetDivider();
            var positions = new List<long>()
            {
                1, 2, 3, 4,5,6,7,8,9,10
            };

            var result = divider.Divide(positions, new DataSetDivisionOptions()
            {
                TrainingSetPercent = 50, TestSetPercent = 25,ValidationSetPercent = 25
            });

            result.Length.Should().Be(3);
        }
    }

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