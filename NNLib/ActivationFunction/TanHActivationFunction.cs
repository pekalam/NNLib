using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NNLib.ActivationFunction
{
    public class TanHActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x)
        {
            return Matrix.Tanh(x);
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            return 1 - Matrix.Tanh(x).PointwisePower(2);
        }
    }
}