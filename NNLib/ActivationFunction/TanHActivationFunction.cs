using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NNLib
{
    public class TanHActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x)
        {
            return Matrix.Tanh(x);
        }

        public Matrix<double> Derivative(Matrix<double> y)
        {
            return 1 - Matrix.Tanh(y).PointwisePower(2);
        }
    }
}