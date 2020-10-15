using System;

namespace NNLib.Data
{
    /// <summary>
    /// Contains input and target vector sets. Used by supervised training algorithms.
    /// </summary>
    public partial class SupervisedTrainingSamples : IDisposable
    {
        public IVectorSet Input { get; }
        public IVectorSet Target { get; }

        public SupervisedTrainingSamples(IVectorSet input, IVectorSet target)
        {
            if (input.Count != target.Count)
            {
                throw new ArgumentException($"Invalid count of input and target sets: {input.Count} {target.Count}");
            }

            Input = input;
            Target = target;
        }

        public void Dispose()
        {
            Input?.Dispose();
            Target?.Dispose();
        }
    }
}