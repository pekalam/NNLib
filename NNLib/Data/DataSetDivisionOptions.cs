using System;

namespace NNLib.Data
{
    public class DataSetDivisionOptions
    {
        private int _trainingSetPercent;
        private int _validationSetPercent;
        private int _testSetPercent;

        private void ValidatePercentage()
        {
            if (TrainingSetPercent + ValidationSetPercent + TestSetPercent > 100)
            {
                throw new ArgumentException(
                    $"Sum of percentages is gt than 100 training: {TrainingSetPercent} validation: {ValidationSetPercent} test:{TestSetPercent}");
            }
        }

        private void CheckGtOrEqZero(int percents)
        {
            if (percents < 0)
            {
                throw new ArgumentException("Percents cannot be lower than 0");
            }
        }

        public int TrainingSetPercent
        {
            get => _trainingSetPercent;
            set
            {
                CheckGtOrEqZero(value);
                _trainingSetPercent = value;
                ValidatePercentage();
            }
        }

        public int ValidationSetPercent
        {
            get => _validationSetPercent;
            set
            {
                CheckGtOrEqZero(value);
                _validationSetPercent = value;
                ValidatePercentage();
            }
        }

        public int TestSetPercent
        {
            get => _testSetPercent;
            set
            {
                CheckGtOrEqZero(value);
                _testSetPercent = value;
                ValidatePercentage();
            }
        }
    }
}