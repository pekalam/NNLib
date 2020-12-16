using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class ArcTanActivationFunction : IActivationFunction
    {
        private MatrixColPool _fData = null!;
        private MatrixColPool _dfData = null!;

        private ArcTanActivationFunction(MatrixColPool fData, MatrixColPool dfData)
        {
            _fData = fData.Clone();
            _dfData = dfData.Clone();
        }

        public ArcTanActivationFunction()
        {
            
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            var storage = _fData!.Get(x.ColumnCount);

            x.PointwiseAtan(storage);
            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            var storage = _dfData!.Get(x.ColumnCount);

            x.PointwisePower(2, storage);
            storage.Add(1, storage);
            storage.PointwisePower(-1, storage);
            return storage;
        }

        public void InitMemory(Layer layer)
        {
            _fData = new MatrixColPool(layer.NeuronsCount, 1);
            _dfData = new MatrixColPool(layer.NeuronsCount, 1);
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _dfData.ClearOtherThanColumnVec();
            _fData.ClearOtherThanColumnVec();
            _dfData.AddToPool(data.Input.Count);
            _fData.AddToPool(data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new ArcTanActivationFunction(_fData,_dfData);
        }
    }
}