using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class ArcTanActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x)
        {
            return x.PointwiseAtan();
        }

        public Matrix<double> DerivativeY(Matrix<double> y)
        {
            return 1 / (y.PointwisePower(2) + 1);
        }
    }
}