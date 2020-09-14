using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib
{
    public abstract class AlgorithmBase
    {
        internal abstract void Setup(SupervisedSet trainingData, MLPNetwork network, ILossFunction lossFunction);
        internal abstract void Reset();

        internal abstract int Iterations { get; }
        internal abstract bool DoIteration(in CancellationToken ct = default);
    }
}