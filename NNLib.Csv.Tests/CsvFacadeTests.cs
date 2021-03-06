using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
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

        [Fact]
        public void ChangeVarIndexes_test()
        {
            var (originalSets, _, orgInd) = CsvFacade.LoadSets(fileName, new LinearDataSetDivider(),
                new DataSetDivisionOptions()
                {
                    TrainingSetPercent = 33,
                    TestSetPercent = 33,
                    ValidationSetPercent = 33,
                });
            var (sets, variableNames, ind) = CsvFacade.LoadSets(fileName, new LinearDataSetDivider(), new DataSetDivisionOptions()
            {
                TrainingSetPercent = 33,
                TestSetPercent = 33,
                ValidationSetPercent = 33,
            });

            sets.TrainingSet.Input[0].RowCount.Should().Be(2);

            var newIndexes = ind.ChangeVariableUse(0, VariableUses.Target);
            CsvFacade.ChangeVariableIndexes(newIndexes, sets);

            sets.TrainingSet.Input[0].RowCount.Should().Be(1);
            sets.TrainingSet.Target[0].RowCount.Should().Be(2);
            sets.TrainingSet.Input.Count.Should().Be(originalSets.TrainingSet.Input.Count);
            sets.TrainingSet.Target.Count.Should().Be(originalSets.TrainingSet.Target.Count);

            for (int i = 0; i < sets.TrainingSet.Input.Count; i++)
            {
                sets.TrainingSet.Input[i][0, 0].Should().Be(originalSets.TrainingSet.Input[i][1, 0]);
                sets.TrainingSet.Target[i][0, 0].Should().Be(originalSets.TrainingSet.Input[i][0, 0]);
                sets.TrainingSet.Target[i][1, 0].Should().Be(originalSets.TrainingSet.Target[i][0, 0]);
            }

            newIndexes = newIndexes.ChangeVariableUse(2, VariableUses.Ignore);
            CsvFacade.ChangeVariableIndexes(newIndexes, sets);


            sets.TrainingSet.Target[0].RowCount.Should().Be(1);
            sets.TrainingSet.Input[0].RowCount.Should().Be(1);
            sets.TrainingSet.Input.Count.Should().Be(originalSets.TrainingSet.Input.Count);
            sets.TrainingSet.Target.Count.Should().Be(originalSets.TrainingSet.Target.Count);

            for (int i = 0; i < sets.TrainingSet.Input.Count; i++)
            {
                sets.TrainingSet.Input[i][0, 0].Should().Be(originalSets.TrainingSet.Input[i][1, 0]);
                sets.TrainingSet.Target[i][0, 0].Should().Be(originalSets.TrainingSet.Input[i][0, 0]);
            }



            newIndexes = newIndexes.ChangeVariableUse(2, VariableUses.Input);
            CsvFacade.ChangeVariableIndexes(newIndexes, sets);

            sets.TrainingSet.Target[0].RowCount.Should().Be(1);
            sets.TrainingSet.Input[0].RowCount.Should().Be(2);
            sets.TrainingSet.Input.Count.Should().Be(originalSets.TrainingSet.Input.Count);
            sets.TrainingSet.Target.Count.Should().Be(originalSets.TrainingSet.Target.Count);

            for (int i = 0; i < sets.TrainingSet.Input.Count; i++)
            {
                sets.TrainingSet.Input[i][0, 0].Should().Be(originalSets.TrainingSet.Input[i][1, 0]);
                sets.TrainingSet.Target[i][0, 0].Should().Be(originalSets.TrainingSet.Input[i][0, 0]);
                sets.TrainingSet.Input[i][1, 0].Should().Be(originalSets.TrainingSet.Target[i][0, 0]);
            }

        }

        [Fact]
        public void Copy_creates_deep_copy_of_all_sets()
        {
            var (sets, variableNames, ind) = CsvFacade.LoadSets(fileName, divisionOptions: new DataSetDivisionOptions()
            {
                TrainingSetPercent = 33,
                TestSetPercent = 33,
                ValidationSetPercent = 33,
            });

            var newSets = CsvFacade.Copy(sets);

            newSets.TrainingSet.Input[0] = Matrix<double>.Build.Dense(1, 1, 99);
            newSets.TestSet.Input[0] = Matrix<double>.Build.Dense(1, 1, 100);
            newSets.ValidationSet.Input[0] = Matrix<double>.Build.Dense(1, 1, 101);

            newSets.TrainingSet.Input[0].Should().NotBeEquivalentTo(sets.TrainingSet.Input[0]);
            newSets.ValidationSet.Input[0].Should().NotBeEquivalentTo(sets.ValidationSet.Input[0]);
            newSets.TestSet.Input[0].Should().NotBeEquivalentTo(sets.TestSet.Input[0]);
        }
    }
}