using MathNet.Numerics.LinearAlgebra;

namespace NNLib.LossFunction
{
    public class QuadraticLossFunction : ILossFunction
    {
        public Matrix<double> Function(Matrix<double> input, Matrix<double> target)
        {
            var err = target.Subtract(input);
            err.PointwiseMultiply(err, err);
            var result = err.Multiply(0.5d);
            return result;
        }

        public Matrix<double> Derivative(Matrix<double> input, Matrix<double> target)
        {
            var result = input.Subtract(target);
            return result;
        }
    }
}