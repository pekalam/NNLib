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
            var cols = target.ColumnCount;
            var storage = _fData.Get(cols);

            target.Subtract(input, storage);
            storage.PointwiseMultiply(storage, storage);
            storage.Multiply(1/(2d*cols), storage);


            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> input, Matrix<double> target)
        {
            var cols = target.ColumnCount;
            var storage = _dfData.Get(cols);

            input.Subtract(target, storage);
            if (cols > 1)
            {
                storage.Multiply(1d / cols, storage);
            }

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