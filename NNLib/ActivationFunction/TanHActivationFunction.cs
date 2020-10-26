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

        public void Function(Matrix<double> x, Matrix<double> result)
        {
            x.PointwiseTanh(result);
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            return 1 - Matrix.Tanh(x).PointwisePower(2);
        }

        public void Derivative(Matrix<double> x, Matrix<double> result)
        {
            x.PointwiseTanh(result);
            result.PointwisePower(2, result);
            result.Negate(result);
            result.Add(1, result);
        }
    }
}