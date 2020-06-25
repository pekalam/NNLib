using System;
using System.Linq;
using NNLib.Common;

namespace NNLib.Csv
{
    public static class CsvFacade
    {
        public static (SupervisedTrainingSets sets, string[] variableNames, SupervisedSetVariableIndexes indexes) LoadSets(string fileName,
            IDataSetDivider? divider = null, DataSetDivisionOptions? divisionOptions = null,
            SupervisedSetVariableIndexes? variableIndexes = null)
        {
            divider ??= new LinearDataSetDivider();
            divisionOptions ??= new DataSetDivisionOptions()
            {
                TrainingSetPercent = 100,TestSetPercent = 0,ValidationSetPercent = 0
            };

            var builder = new InternalDataSetBuilder(divider);

            var setInfos = builder.CreatePartitionedDataSets(fileName, divisionOptions);
             
            if (setInfos.All(info => info.DataSetType != DataSetType.Training)) throw new Exception("setInfos does not contain training set");

            variableIndexes ??= CreateDefaultIndexes(setInfos[0]);

            var trainingVecSet =
                CsvVectorSetFactory.CreateCsvSupervisedVectorSet(fileName, variableIndexes,
                    setInfos.First(info => info.DataSetType == DataSetType.Training));
            var sets = new SupervisedTrainingSets(FromVectorSetsPair(trainingVecSet));

            foreach (var setInfo in setInfos)
            {
                var vectorSets = CsvVectorSetFactory.CreateCsvSupervisedVectorSet(fileName, variableIndexes, setInfo);
                if (setInfo.DataSetType == DataSetType.Test)
                {
                    sets.TestSet = FromVectorSetsPair(vectorSets);
                }
                else if (setInfo.DataSetType == DataSetType.Validation)
                {
                    sets.ValidationSet = FromVectorSetsPair(vectorSets);
                }
            }


            return (sets, setInfos[0].VariableNames, variableIndexes);
        }

        private static SupervisedSet FromVectorSetsPair(in (IVectorSet input, IVectorSet target) vectorSets) =>
            new SupervisedSet(vectorSets.input, vectorSets.target);

        private static SupervisedSetVariableIndexes CreateDefaultIndexes(DataSetInfo info)
        {
            var inputInd = info.VariableNames.Select((_, ind) => ind).Take(info.VariableNames.Length - 1).ToArray();
            var targetInd = new int[] { info.VariableNames.Length - 1 };
            return new SupervisedSetVariableIndexes(inputInd, targetInd);
        }

        public static void ChangeVariableIndexes(SupervisedSetVariableIndexes variableIndexes, SupervisedTrainingSets sets)
        {
            //TODO double dispatch
            (sets.TrainingSet.Input as CsvFileVectorSet)?.FileReader.ChangeVariables(variableIndexes);
            (sets.TestSet?.Input as CsvFileVectorSet)?.FileReader.ChangeVariables(variableIndexes);
            (sets.ValidationSet?.Input as CsvFileVectorSet)?.FileReader.ChangeVariables(variableIndexes);
        }
    }
}