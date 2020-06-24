using MathNet.Numerics.LinearAlgebra;

namespace NNLib.Training
{
    public abstract class AlgorithmBase
    {
        public abstract LearningMethodResult CalculateDelta(Matrix<double> input, Matrix<double> expected, ILossFunction lossFunction);
        
        public abstract BatchParams BatchParams { get; }
    }
}