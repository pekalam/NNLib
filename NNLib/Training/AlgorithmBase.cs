using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib
{
    public abstract class AlgorithmBase
    {
        public abstract void Setup(SupervisedSet trainingData, MLPNetwork network, ILossFunction lossFunction);
        public abstract void ResetIterations();
        public BatchTrainer? BatchTrainer { get; protected set; }

        public abstract int Iterations { get; }
        public abstract bool DoIteration(in CancellationToken ct = default);
    }
}