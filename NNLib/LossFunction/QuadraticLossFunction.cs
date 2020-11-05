using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.LossFunction
{
    public class QuadraticLossFunction : ILossFunction
    {
        private Matrix<double> _f;
        private Matrix<double> _df;

        private Matrix<double> _fData;
        private Matrix<double> _dfData;

        public Matrix<double> Function(Matrix<double> input, Matrix<double> target)
        {
            var storage = target.ColumnCount == 1 ? _f : _fData;

            target.Subtract(input, storage);
            storage.PointwiseMultiply(storage, storage);
            storage.Multiply(0.5d, storage);

            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> input, Matrix<double> target)
        {
            var storage = target.ColumnCount == 1 ? _df : _dfData;

            input.Subtract(target, storage);
            return storage;
        }

        public void InitializeMemory(Layer layer, SupervisedTrainingSamples data)
        {
            _f = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _df = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _fData = Matrix<double>.Build.Dense(layer.NeuronsCount, data.Input.Count);
            _dfData = Matrix<double>.Build.Dense(layer.NeuronsCount, data.Input.Count);
        }
    }
}