using System;

namespace NNLib.Data
{
    public class DataSetDivisionOptions
    {
        private decimal _trainingSetPercent;
        private decimal _validationSetPercent;
        private decimal _testSetPercent;

        private void ValidatePercentage()
        {
            if (TrainingSetPercent + ValidationSetPercent + TestSetPercent > 100m)
            {
                throw new ArgumentException(
                    $"Sum of percentages is gt than 100 training: {TrainingSetPercent} validation: {ValidationSetPercent} test:{TestSetPercent}");
            }
        }

        private void CheckGtOrEqZero(decimal percents)
        {
            if (percents < 0m)
            {
                throw new ArgumentException("Percents cannot be lower than 0");
            }
        }

        public decimal TrainingSetPercent
        {
            get => _trainingSetPercent;
            set
            {
                CheckGtOrEqZero(value);
                _trainingSetPercent = value;
                ValidatePercentage();
            }
        }

        public decimal ValidationSetPercent
        {
            get => _validationSetPercent;
            set
            {
                CheckGtOrEqZero(value);
                _validationSetPercent = value;
                ValidatePercentage();
            }
        }

        public decimal TestSetPercent
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