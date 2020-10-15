using MathNet.Numerics.LinearAlgebra;

namespace NNLib.ActivationFunction
{
    public class LinearActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x) => x.Clone();
        public Matrix<double> Derivative(Matrix<double> x) => Matrix<double>.Build.Dense(x.RowCount, x.ColumnCount, Matrix<double>.One);
    }
}