using FluentAssertions;
using NNLib.Common;
using Xunit;

namespace NNLib.Csv.Tests
{
    public class CsvFacadeTests
    {
        string fileName = @"testxyz.csv";


        [Fact]
        public void LoadSets_with_default_options_loads_training_set()
        {
            var (sets, variableNames, ind) = CsvFacade.LoadSets(fileName);

            sets.TestSet.Should().BeNull();
            sets.ValidationSet.Should().BeNull();
            sets.TrainingSet.Should().NotBeNull();
            variableNames.Should().BeEquivalentTo("x", "y", "z");
            ind.Ignored.Should().BeEmpty();
            ind.InputVarIndexes.Should().BeEquivalentTo(0,1);
            ind.TargetVarIndexes.Should().BeEquivalentTo(2);

            sets.TrainingSet.Input.Count.Should().Be(12);
        }

        [Fact]
        public void LoadSets_with_division_options_divides_loaded_set()
        {
            var (sets, variableNames, ind) = CsvFacade.LoadSets(fileName, divisionOptions: new DataSetDivisionOptions()
            {
                TrainingSetPercent = 33, TestSetPercent = 33, ValidationSetPercent = 33,
            });

            ind.Ignored.Should().BeEmpty();
            ind.InputVarIndexes.Should().BeEquivalentTo(0, 1);
            ind.TargetVarIndexes.Should().BeEquivalentTo(2);

            sets.TestSet.Should().NotBeNull();
            sets.ValidationSet.Should().NotBeNull();
            sets.TrainingSet.Should().NotBeNull();
            variableNames.Should().BeEquivalentTo("x", "y", "z");

            sets.TrainingSet.Input.Count.Should().Be(4);
            sets.TrainingSet.Input.Count.Should().Be(4);
            sets.TrainingSet.Input.Count.Should().Be(4);
        }
    }
}