using System;
using System.Linq;
using NNLib.Common;

namespace NNLib.Csv
{
    public static class CsvFacade
    {
        public static (SupervisedTrainingSets sets, string[] variableNames, SupervisedSetVariableIndexes indexes)
            LoadSets(string fileName,
                IDataSetDivider? divider = null, DataSetDivisionOptions? divisionOptions = null,
                SupervisedSetVariableIndexes? variableIndexes = null)
        {
            divider ??= new LinearDataSetDivider();
            divisionOptions ??= new DataSetDivisionOptions()
            {
                TrainingSetPercent = 100, TestSetPercent = 0, ValidationSetPercent = 0
            };

            var builder = new DataSetInfoBuilder(divider);

            var setInfos = builder.CreatePartitionedDataSets(fileName, divisionOptions);

            if (setInfos.All(info => info.DataSetType != DataSetType.Training))
                throw new Exception("setInfos does not contain training set");

            variableIndexes ??= CreateDefaultIndexes(setInfos[0]);

            var trainingVecSet =
                CsvVectorSetFactory.CreateCsvSupervisedVectorSet(fileName, variableIndexes,
                    setInfos.First(info => info.DataSetType == DataSetType.Training));
            var sets = new SupervisedTrainingSets(new SupervisedSet(trainingVecSet.input, trainingVecSet.target));

            foreach (var setInfo in setInfos.Where(i => i.DataSetType != DataSetType.Training))
            {
                var vectorSets = CsvVectorSetFactory.CreateCsvSupervisedVectorSet(fileName, variableIndexes, setInfo);
                if (setInfo.DataSetType == DataSetType.Test)
                {
                    sets.TestSet = new SupervisedSet(vectorSets.input, vectorSets.target);
                }
                else if (setInfo.DataSetType == DataSetType.Validation)
                {
                    sets.ValidationSet = new SupervisedSet(vectorSets.input, vectorSets.target);
                }
            }


            return (sets, setInfos[0].VariableNames, variableIndexes);
        }

        private static SupervisedSetVariableIndexes CreateDefaultIndexes(DataSetInfo info)
        {
            var inputInd = info.VariableNames.Select((_, ind) => ind).Take(info.VariableNames.Length - 1).ToArray();
            var targetInd = new int[] {info.VariableNames.Length - 1};
            return new SupervisedSetVariableIndexes(inputInd, targetInd);
        }

        public static void ChangeVariableIndexes(SupervisedSetVariableIndexes variableIndexes,
            SupervisedTrainingSets sets)
        {
            //TODO double dispatch
            (sets.TrainingSet.Input as CsvFileVectorSet)?.FileReader.ChangeVariables(variableIndexes);
            (sets.TestSet?.Input as CsvFileVectorSet)?.FileReader.ChangeVariables(variableIndexes);
            (sets.ValidationSet?.Input as CsvFileVectorSet)?.FileReader.ChangeVariables(variableIndexes);
        }

        public static SupervisedTrainingSets Copy(SupervisedTrainingSets sets)
        {
            var readerCpy = (sets.TrainingSet.Input as CsvFileVectorSet)?.FileReader.Copy()!;
            var trainingSet = new SupervisedSet((sets.TrainingSet.Input as CsvFileVectorSet)!.Copy(readerCpy),
                (sets.TrainingSet.Target as CsvFileVectorSet)!.Copy(readerCpy));

            var newSets = new SupervisedTrainingSets(trainingSet);


            if (sets.ValidationSet != null)
            {
                readerCpy = (sets.ValidationSet.Input as CsvFileVectorSet)?.FileReader.Copy()!;
                var validationSet = new SupervisedSet((sets.ValidationSet.Input as CsvFileVectorSet)!.Copy(readerCpy),
                    (sets.ValidationSet.Target as CsvFileVectorSet)!.Copy(readerCpy));
                newSets.ValidationSet = validationSet;
            }

            if (sets.TestSet != null)
            {
                readerCpy = (sets.TestSet.Input as CsvFileVectorSet)?.FileReader.Copy()!;
                var testSet = new SupervisedSet((sets.TestSet.Input as CsvFileVectorSet)!.Copy(readerCpy),
                    (sets.TestSet.Target as CsvFileVectorSet)!.Copy(readerCpy));
                newSets.TestSet = testSet;

            }

            return newSets;
        }
    }
}