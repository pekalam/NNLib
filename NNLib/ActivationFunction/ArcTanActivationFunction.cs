using MathNet.Numerics.LinearAlgebra;

namespace NNLib.ActivationFunction
{
    public class ArcTanActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x)
        {
            return x.PointwiseAtan();
        }

        public void Function(Matrix<double> x, Matrix<double> result)
        {
            x.PointwiseAtan(result);
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            return 1 / (x.PointwisePower(2) + 1);
        }

        public void Derivative(Matrix<double> x, Matrix<double> result)
        {
            x.PointwisePower(2, result);
            result.Add(1, result);
            result.PointwisePower(-1, result);
        }
    }
}