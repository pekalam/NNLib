using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public interface IActivationFunction
    {
        Matrix<double> Function(Matrix<double> x);
        Matrix<double> DerivativeY(Matrix<double> y);
    }
}