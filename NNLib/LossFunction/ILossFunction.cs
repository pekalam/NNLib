using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using NNLib.Exceptions;
using NNLib.MLP;

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

        void InitializeMemory(Layer layer, SupervisedTrainingSamples data);

        /// <summary>
        /// Calculates value of loss function using network output for given input
        /// </summary>
        Matrix<double> CalculateError(MLPNetwork network, Matrix<double> input, Matrix<double> target, in CancellationToken ct = default)
        {
            network.CalculateOutput(input);
            TrainingCanceledException.ThrowIfCancellationRequested(ct);
            return Function(network.Output!, target);
        }
    }
}