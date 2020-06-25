using System;

namespace NNLib
{
    public enum SupervisedSetType
    {
        Training, Validation,Test
    }

    public class SupervisedTrainingSets : IDisposable
    {
        public SupervisedSet TrainingSet { get; }
        public SupervisedSet? ValidationSet { get; set; }
        public SupervisedSet? TestSet { get; set; }

        public SupervisedTrainingSets(SupervisedSet trainingSet)
        {
            Guards._NotNull(trainingSet);
            TrainingSet = trainingSet;
        }

        public void Dispose()
        {
            TrainingSet?.Dispose();
            ValidationSet?.Dispose();
            TestSet?.Dispose();
        }
    }
}