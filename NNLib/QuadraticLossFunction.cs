using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class QuadraticLossFunction : ILossFunction
    {
        public Matrix<double> Function(Matrix<double> input, Matrix<double> expected)
        {
            var err = input.Subtract(expected);
            err.Power(2, err);
            var result = err.Multiply(0.5d);
            return result;
        }

        public Matrix<double> Derivative(Matrix<double> input, Matrix<double> expected)
        {
            var result = input.Subtract(expected);
            return result;
        }
    }
}