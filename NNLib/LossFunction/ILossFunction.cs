using MathNet.Numerics.LinearAlgebra;

namespace NNLib.LossFunction
{
    /// <summary>
    /// Interface implemented by loss functions
    /// </summary>
    public interface ILossFunction
    {
        /// <summary>
        /// Calculates value of loss function
        /// </summary>
        Matrix<double> Function(Matrix<double> input, Matrix<double> target);
        
        /// <summary>
        /// Calculates derivative with respect to <see cref="input"/> parameter
        /// </summary>
        Matrix<double> Derivative(Matrix<double> input, Matrix<double> target);
    }
}