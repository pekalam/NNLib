using NNLib.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NNLib.Csv
{
    internal class DataSetDividerHelpers
    {
        public static (int trainingSetCount, int validationSetCount, int testSetCount) GetSetDivision(
            DataSetDivisionOptions divOptions, int totalLines)
        {
            long totalLinesLong = (long)totalLines;
            var trainingSetCount = (int)Math.Ceiling(divOptions.TrainingSetPercent * totalLinesLong / 100m);
            trainingSetCount = trainingSetCount > totalLines ? totalLines : trainingSetCount;
            totalLines -= trainingSetCount;

            var validationSetCount = (int)Math.Ceiling(divOptions.ValidationSetPercent * totalLinesLong / 100m);
            validationSetCount = validationSetCount > totalLines ? totalLines : validationSetCount;
            totalLines -= validationSetCount;

            var testSetCount = (int)Math.Ceiling(divOptions.TestSetPercent * totalLinesLong / 100m);
            testSetCount = testSetCount > totalLines ? totalLines : testSetCount;

            if (trainingSetCount == 0)
            {
                throw new ArgumentException("Training set count cannot be 0");
            }

            return (trainingSetCount, validationSetCount, testSetCount);
        }
    }

    public class RandomDataSetDivider : IDataSetDivider
    {
        public (DataSetType setType, List<long> positions)[] Divide(List<long> fileNewLinePositions, DataSetDivisionOptions divOptions)
        {
            var rnd = new Random();
            var divided = new List<(DataSetType setType, List<long> positions)>();
            int totalLines = fileNewLinePositions.Count;
            var randomIndexes = Enumerable.Range(0, totalLines).OrderBy(_ => rnd.Next(0, totalLines)).ToArray();

            if (divOptions.TrainingSetPercent == 100)
            {
                return new[] {(DataSetType.Training, fileNewLinePositions)};
            }

            var (trainingSetCount, validationSetCount, testSetCount) =
                DataSetDividerHelpers.GetSetDivision(divOptions, fileNewLinePositions.Count);

            var ind = 0;
            var setPositions = new List<long>();
            for (int i = 0; i < trainingSetCount; i++)
            {
                setPositions.Add(fileNewLinePositions[randomIndexes[ind++]]);
            }
            divided.Add((DataSetType.Training, setPositions.Select(_ => _).ToList()));
            setPositions.Clear();

            if (validationSetCount > 0)
            {
                for (int i = 0; i < validationSetCount; i++)
                {
                    setPositions.Add(fileNewLinePositions[randomIndexes[ind++]]);
                }
                divided.Add((DataSetType.Validation, setPositions.Select(_ => _).ToList()));
                setPositions.Clear();
            }

            if (testSetCount > 0)
            {
                for (int i = 0; i < testSetCount; i++)
                {
                    setPositions.Add(fileNewLinePositions[randomIndexes[ind++]]);
                }
                divided.Add((DataSetType.Test, setPositions.Select(_ => _).ToList()));
                setPositions.Clear();
            }

            return divided.ToArray();
        }
    }

    public class LinearDataSetDivider : IDataSetDivider
    {
        public (DataSetType setType, List<long> positions)[] Divide(List<long> fileNewLinePositions,
            DataSetDivisionOptions divOptions)
        {
            if (fileNewLinePositions.Count <= 0)
            {
                throw new ArgumentException("fileNewLinePositions is empty");
            }
            if (divOptions.TrainingSetPercent <= 0)
            {
                throw new ArgumentException("Invalid trainingSetPercent parameter");
            }

            var divided = new List<(DataSetType setType, List<long> positions)>();

            var (trainingSetCount, validationSetCount, testSetCount) =
                DataSetDividerHelpers.GetSetDivision(divOptions, fileNewLinePositions.Count);

            var trainingSetPos = new List<long>();
            var trainingRange = fileNewLinePositions.GetRange(0, trainingSetCount);
            trainingSetPos.AddRange(trainingRange);

            divided.Add((DataSetType.Training, trainingSetPos));

            var nextDivOffset = trainingSetCount;

            if (validationSetCount > 0)
            {

                var validationSetPos = new List<long>();
                var validationRange = fileNewLinePositions.GetRange(nextDivOffset, validationSetCount);
                validationSetPos.AddRange(validationRange);

                divided.Add((DataSetType.Validation, validationSetPos));
                nextDivOffset += validationSetCount;
            }

            if (testSetCount > 0)
            {
                var testSetPos = new List<long>();
                var testRange = fileNewLinePositions.GetRange(nextDivOffset, testSetCount);
                testSetPos.AddRange(testRange);

                divided.Add((DataSetType.Test, testSetPos));
            }



            return divided.ToArray();
        }
    }
}