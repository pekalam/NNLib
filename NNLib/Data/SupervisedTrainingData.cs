using System;

namespace NNLib.Data
{
    public class SupervisedTrainingData : IDisposable
    {
        private readonly ConcatenatedVectorSet _concatenatedInput;
        private readonly ConcatenatedVectorSet _concatenatedTarget;
        private SupervisedTrainingSamples? _validationSet;

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

        public SupervisedTrainingSamples? TestSet { get; set; }

        /// <summary>
        /// Concatenated TrainingSet.Input + ValidationSet.Input
        /// </summary>
        public IVectorSet ConcatenatedInput => _concatenatedInput;
        /// <summary>
        /// Concatenated TrainingSet.Target + ValidationSet.Target
        /// </summary>
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