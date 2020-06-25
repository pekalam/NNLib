using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public abstract class AlgorithmBase
    {
        public abstract void Setup(Common.SupervisedSet trainingData, MLPNetwork network, ILossFunction lossFunction);

        public abstract LearningMethodResult CalculateDelta(MLPNetwork network, Matrix<double> input, Matrix<double> expected, ILossFunction lossFunction);
    }
}