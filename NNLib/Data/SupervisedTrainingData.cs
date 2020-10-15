using System;

namespace NNLib.Data
{
    public class SupervisedTrainingData : IDisposable
    {
        public SupervisedTrainingSamples TrainingSet { get; }
        public SupervisedTrainingSamples? ValidationSet { get; set; }
        public SupervisedTrainingSamples? TestSet { get; set; }

        public SupervisedTrainingData(SupervisedTrainingSamples trainingTrainingSamples)
        {
            Guards._NotNull(trainingTrainingSamples);
            TrainingSet = trainingTrainingSamples;
        }

        public void Dispose()
        {
            TrainingSet?.Dispose();
            ValidationSet?.Dispose();
            TestSet?.Dispose();
        }
    }
}