using System;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib.ActivationFunction
{
    public class LinearActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x) => x.Clone();
        public void Function(Matrix<double> x, Matrix<double> result)
        {
            x.CopyTo(result);
        }

        public Matrix<double> Derivative(Matrix<double> x) => Matrix<double>.Build.Dense(x.RowCount, x.ColumnCount, Matrix<double>.One);
        public void Derivative(Matrix<double> x, Matrix<double> result)
        {
            Array.Fill(result.AsColumnMajorArray(), 1);
        }
    }
}