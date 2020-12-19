using System;

namespace NNLib.Data
{
    public class SupervisedTrainingData : IDisposable
    {
        private ConcatenatedVectorSet _concatenatedInput;
        private ConcatenatedVectorSet _concatenatedTarget;
        private SupervisedTrainingSamples? _validationSet;
        private SupervisedTrainingSamples? _testSet;

        public SupervisedTrainingSamples TrainingSet { get; }

        public SupervisedTrainingSamples? ValidationSet
        {
            get => _validationSet;
            set
            {
                if (_validationSet != null)
                {
                    _concatenatedInput.Remove(_validationSet.Input);
                    _concatenatedTarget.Remove(_validationSet.Target);
                }
                _validationSet = value;
                if (value != null)
                {
                    _concatenatedInput.AddVectorSet(value.Input);
                    _concatenatedTarget.AddVectorSet(value.Target);
                }
            }
        }

        public SupervisedTrainingSamples? TestSet
        {
            get => _testSet;
            set
            {
                if (_testSet != null)
                {
                    _concatenatedInput.Remove(_testSet.Input);
                    _concatenatedTarget.Remove(_testSet.Target);
                }
                _testSet = value;
                if (value != null)
                {
                    _concatenatedInput.AddVectorSet(value.Input);
                    _concatenatedTarget.AddVectorSet(value.Target);
                }
            }
        }

        public IVectorSet ConcatenatedInput => _concatenatedInput;
        public IVectorSet ConcatenatedTarget => _concatenatedTarget;

        public SupervisedTrainingData(SupervisedTrainingSamples trainingTrainingSamples)
        {
            _concatenatedInput = new ConcatenatedVectorSet();
            _concatenatedTarget = new ConcatenatedVectorSet();
            Guards._NotNull(trainingTrainingSamples);
            TrainingSet = trainingTrainingSamples;
            _concatenatedInput.AddVectorSet(TrainingSet.Input);
            _concatenatedTarget.AddVectorSet(TrainingSet.Target);
        }

        public void Dispose()
        {
            TrainingSet?.Dispose();
            ValidationSet?.Dispose();
            TestSet?.Dispose();
        }
    }
}