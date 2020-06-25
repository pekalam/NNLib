using MathNet.Numerics.LinearAlgebra;

namespace NNLib.Training
{
    public abstract class AlgorithmBase
    {
        public abstract void Setup(SupervisedSet trainingData, MLPNetwork network);

        public abstract LearningMethodResult CalculateDelta(MLPNetwork network, Matrix<double> input, Matrix<double> expected, ILossFunction lossFunction);
    }
}