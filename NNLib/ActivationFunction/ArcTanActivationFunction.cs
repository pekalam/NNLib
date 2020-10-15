using MathNet.Numerics.LinearAlgebra;

namespace NNLib.ActivationFunction
{
    public class ArcTanActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x)
        {
            return x.PointwiseAtan();
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            return 1 / (x.PointwisePower(2) + 1);
        }
    }
}