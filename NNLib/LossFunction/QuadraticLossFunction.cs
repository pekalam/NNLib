using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.LossFunction
{
    public class QuadraticLossFunction : ILossFunction
    {
        private MatrixColPool _fData = null!;
        private MatrixColPool _dfData = null!;

        public Matrix<double> Function(Matrix<double> input, Matrix<double> target)
        {
            var storage = _fData.Get(target.ColumnCount);

            target.Subtract(input, storage);
            storage.PointwiseMultiply(storage, storage);
            storage.Multiply(0.5d, storage);

            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> input, Matrix<double> target)
        {
            var storage = _dfData.Get(target.ColumnCount);

            input.Subtract(target, storage);
            return storage;
        }

        public void InitializeMemory(Layer layer, SupervisedTrainingSamples data)
        {
            _fData = new MatrixColPool(layer.NeuronsCount, 1);
            _dfData = new MatrixColPool(layer.NeuronsCount, 1);
            _fData.AddToPool(data.Input.Count);
            _dfData.AddToPool(data.Input.Count);
        }
    }
}