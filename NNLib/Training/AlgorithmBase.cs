using System.Threading;
using NNLib.Data;
using NNLib.LossFunction;
using NNLib.MLP;

namespace NNLib.Training
{
    public abstract class AlgorithmBase
    {
        internal abstract void Setup(SupervisedSet trainingData, MLPNetwork network, ILossFunction lossFunction);
        internal abstract void Reset();

        internal abstract int Iterations { get; }
        internal abstract bool DoIteration(in CancellationToken ct = default);
    }
}