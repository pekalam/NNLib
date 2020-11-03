using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

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

        public void InitMemory(Layer layer)
        {
            
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            
        }
    }
}