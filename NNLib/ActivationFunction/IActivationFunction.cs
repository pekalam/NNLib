using MathNet.Numerics.LinearAlgebra;

namespace NNLib.ActivationFunction
{
    /// <summary>
    /// Interface implemented by activation functions
    /// </summary>
    public interface IActivationFunction
    {
        /// <summary>
        /// Calculates value of function for parameter <see cref="x"/>
        /// </summary>
        Matrix<double> Function(Matrix<double> x);
        void Function(Matrix<double> x, Matrix<double> result);

        /// <summary>
        /// Calculates derivative with respect to parameter <see cref="x"/>
        /// </summary>
        Matrix<double> Derivative(Matrix<double> x);
        void Derivative(Matrix<double> x, Matrix<double> result);
    }
}