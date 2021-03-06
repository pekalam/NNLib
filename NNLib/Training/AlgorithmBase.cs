﻿using NNLib.Data;
using NNLib.LossFunction;
using NNLib.MLP;
using System.Threading;

namespace NNLib.Training
{
    public abstract class AlgorithmBase
    {
        internal abstract double? GetError();
        internal abstract void Setup(SupervisedTrainingSamples set, LoadedSupervisedTrainingData loadedSets, MLPNetwork network, ILossFunction lossFunction);
        internal abstract void Reset();
        internal abstract int Iterations { get; }
        internal abstract bool DoIteration(in CancellationToken ct = default);
    }
}