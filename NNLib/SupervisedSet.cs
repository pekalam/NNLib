using System;

namespace NNLib
{
    public partial class SupervisedSet : IDisposable
    {
        public IVectorSet Input { get; }
        public IVectorSet Target { get; }

        public SupervisedSet(IVectorSet input, IVectorSet target)
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