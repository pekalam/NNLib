using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        private MatrixColPool _fData = null!;
        private MatrixColPool _dfData = null!;
        private MatrixColPool _dfDataClone = null!;

        private SigmoidActivationFunction(MatrixColPool fData, MatrixColPool dfData, MatrixColPool dfDataClone)
        {
            _fData = fData.Clone();
            _dfData = dfData.Clone();
            _dfDataClone = dfDataClone.Clone();
        }

        public SigmoidActivationFunction()
        {
            
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            var storage = _fData!.Get(x.ColumnCount);

            x.Negate(storage);
            storage.PointwiseExp(storage);
            storage.Add(1, storage);
            storage.PointwisePower(-1, storage);

            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            var cols = x.ColumnCount;
            var storage = _dfData!.Get(cols);
            var storage2 = _dfDataClone!.Get(cols);

            x.Negate(storage);
            storage.PointwiseExp(storage);
            storage.Add(1, storage);
            storage.PointwisePower(-1, storage);

            storage.Negate(storage2);
            storage2.Add(1, storage2);

            storage.PointwiseMultiply(storage2, storage);

            return storage;
        }

        public void InitMemory(Layer layer)
        {
            _fData = new MatrixColPool(layer.NeuronsCount, 1);
            _dfData = new MatrixColPool(layer.NeuronsCount, 1);
            _dfDataClone = new MatrixColPool(layer.NeuronsCount, 1);
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _fData.ClearOtherThanColumnVec();
            _dfData.ClearOtherThanColumnVec();
            _dfDataClone.ClearOtherThanColumnVec();

            _fData.AddToPool(data.Input.Count);
            _dfData.AddToPool(data.Input.Count);
            _dfDataClone.AddToPool(data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new SigmoidActivationFunction(_fData,_dfData,_dfDataClone);
        }
    }
}