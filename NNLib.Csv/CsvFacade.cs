﻿using NNLib.Data;
using System;
using System.Linq;

namespace NNLib.Csv
{
    public static class CsvFacade
    {
        public static (SupervisedTrainingData sets, string[] variableNames, SupervisedSetVariableIndexes indexes)
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
            var sets = new SupervisedTrainingData(new SupervisedTrainingSamples(trainingVecSet.input, trainingVecSet.target));

            foreach (var setInfo in setInfos.Where(i => i.DataSetType != DataSetType.Training))
            {
                var vectorSets = CsvVectorSetFactory.CreateCsvSupervisedVectorSet(fileName, variableIndexes, setInfo);
                if (setInfo.DataSetType == DataSetType.Test)
                {
                    sets.TestSet = new SupervisedTrainingSamples(vectorSets.input, vectorSets.target);
                }
                else if (setInfo.DataSetType == DataSetType.Validation)
                {
                    sets.ValidationSet = new SupervisedTrainingSamples(vectorSets.input, vectorSets.target);
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
            SupervisedTrainingData data)
        {
            var tReader = (data.TrainingSet.Input as CsvFileVectorSet)?.FileReader!;
            var prevIndexes = tReader.CurrentIndexes.Clone();

            tReader.ChangeVariables(variableIndexes);
            if (!prevIndexes.InputVarIndexes.SequenceEqual(variableIndexes.InputVarIndexes))
            {
                (data.TrainingSet.Input as CsvFileVectorSet)!.RaiseModified();
            }
            if (!prevIndexes.TargetVarIndexes.SequenceEqual(variableIndexes.TargetVarIndexes))
            {
                (data.TrainingSet.Target as CsvFileVectorSet)!.RaiseModified();
            }

            var tsReader = (data.TestSet?.Input as CsvFileVectorSet)?.FileReader;
            if (tsReader != null)
            {
                tsReader.ChangeVariables(variableIndexes);
                if (!prevIndexes.InputVarIndexes.SequenceEqual(variableIndexes.InputVarIndexes))
                {
                    (data.TestSet!.Input as CsvFileVectorSet)!.RaiseModified();
                }
                if (!prevIndexes.TargetVarIndexes.SequenceEqual(variableIndexes.TargetVarIndexes))
                {
                    (data.TestSet!.Target as CsvFileVectorSet)!.RaiseModified();
                }
            }
            
            var vReader = (data.ValidationSet?.Input as CsvFileVectorSet)?.FileReader;
            if (vReader != null)
            {
                vReader.ChangeVariables(variableIndexes);
                if (!prevIndexes.InputVarIndexes.SequenceEqual(variableIndexes.InputVarIndexes))
                {
                    (data.ValidationSet!.Input as CsvFileVectorSet)!.RaiseModified();
                }
                if (!prevIndexes.TargetVarIndexes.SequenceEqual(variableIndexes.TargetVarIndexes))
                {
                    (data.ValidationSet!.Target as CsvFileVectorSet)!.RaiseModified();
                }
            }
        }

        public static SupervisedTrainingData Copy(SupervisedTrainingData data)
        {
            var readerCpy = (data.TrainingSet.Input as CsvFileVectorSet)?.FileReader.Copy()!;
            var trainingSet = new SupervisedTrainingSamples((data.TrainingSet.Input as CsvFileVectorSet)!.Copy(readerCpy),
                (data.TrainingSet.Target as CsvFileVectorSet)!.Copy(readerCpy));

            var newSets = new SupervisedTrainingData(trainingSet);


            if (data.ValidationSet != null)
            {
                readerCpy = (data.ValidationSet.Input as CsvFileVectorSet)?.FileReader.Copy()!;
                var validationSet = new SupervisedTrainingSamples((data.ValidationSet.Input as CsvFileVectorSet)!.Copy(readerCpy),
                    (data.ValidationSet.Target as CsvFileVectorSet)!.Copy(readerCpy));
                newSets.ValidationSet = validationSet;
            }

            if (data.TestSet != null)
            {
                readerCpy = (data.TestSet.Input as CsvFileVectorSet)?.FileReader.Copy()!;
                var testSet = new SupervisedTrainingSamples((data.TestSet.Input as CsvFileVectorSet)!.Copy(readerCpy),
                    (data.TestSet.Target as CsvFileVectorSet)!.Copy(readerCpy));
                newSets.TestSet = testSet;

            }

            return newSets;
        }
    }
}