using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class LinearActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x) => x.Clone();
        public Matrix<double> Derivative(Matrix<double> y) => Matrix<double>.Build.Dense(y.RowCount, y.ColumnCount, Matrix<double>.One);
    }
}