using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.LossFunction
{
    public class MaeLossFunction : ILossFunction
    {
        private MatrixColPool _fData = null!;
        private MatrixColPool _dfData = null!;
        private MatrixColPool _dfData2 = null!;

        public Matrix<double> Function(Matrix<double> input, Matrix<double> target)
        {
            var cols = target.ColumnCount;
            var storage = _fData.Get(cols);

            target.Subtract(input, storage);
            storage.PointwiseAbs(storage);
            storage = storage.RowSums().ToColumnMatrix();
            storage.Multiply(1d / cols, storage);

            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> input, Matrix<double> target)
        {
            var cols = target.ColumnCount;
            var rows = target.RowCount;
            var storage = _dfData.Get(cols);
            var storage2 = _dfData2.Get(cols);
            
            input.Subtract(target, storage);
            storage.CopyTo(storage2);
            storage.PointwiseSign(storage);

            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    if (storage2.At(j, i) == 0d)
                    {
                        storage.At(j, i, 0d);
                    }
                }
            }

            
            return storage;
        }

        public void InitializeMemory(Layer layer, SupervisedTrainingSamples data)
        {
            _fData = new MatrixColPool(layer.NeuronsCount, 1);
            _dfData = new MatrixColPool(layer.NeuronsCount, 1);
            _dfData2 = new MatrixColPool(layer.NeuronsCount, 1);
        }
    }
}
