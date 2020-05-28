using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public interface ILossFunction
    {
        Matrix<double> Function(Matrix<double> input, Matrix<double> expected);
        Matrix<double> Derivative(Matrix<double> input, Matrix<double> expected);
    }
}