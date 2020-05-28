using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NNLib.ActivationFunction
{
    public class LinearActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x) => x.Clone();
        public Matrix<double> DerivativeY(Matrix<double> y) => Matrix<double>.Build.Dense(y.RowCount, y.ColumnCount, Matrix<double>.One);
    }

    public class TanHActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x)
        {
            return Matrix.Tanh(x);
        }

        public Matrix<double> DerivativeY(Matrix<double> y)
        {
            return 1 - Matrix.Tanh(y).PointwisePower(2);
        }
    }

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