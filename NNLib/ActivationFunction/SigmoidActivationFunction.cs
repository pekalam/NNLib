using MathNet.Numerics.Differentiation;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NNLib.ActivationFunction
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x)
        {
            var exp = x.Negate().PointwiseExp();
            var result = 1 / (1 + exp);
            return result;
        }

        public Matrix<double> DerivativeY(Matrix<double> y)
        {
            var exp = y.Negate().PointwiseExp();
            var s = 1 / (1 + exp);
            var result = s % (1 - s);
            return result;
        }
    }
}