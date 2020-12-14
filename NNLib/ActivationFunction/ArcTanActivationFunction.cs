using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class ArcTanActivationFunction : IActivationFunction
    {
        private Matrix<double> _f = null!;
        private Matrix<double> _df = null!;

        private MatrixColPool? _fData;
        private MatrixColPool? _dfData;

        private ArcTanActivationFunction(Matrix<double> f, Matrix<double> df, MatrixColPool? fData, MatrixColPool? dfData)
        {
            _f = f.Clone();
            _df = df.Clone();
            _fData = fData?.Clone();
            _dfData = dfData?.Clone();
        }

        public ArcTanActivationFunction()
        {
            
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            var cols = x.ColumnCount;

            Matrix<double> storage = cols == _f.ColumnCount ? _f : _fData!.Get(cols);

            x.PointwiseAtan(storage);
            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            var cols = x.ColumnCount;

            Matrix<double> storage = cols == _df.ColumnCount ? _df : _dfData!.Get(cols);

            x.PointwisePower(2, storage);
            storage.Add(1, storage);
            storage.PointwisePower(-1, storage);
            return storage;
        }

        public void InitMemory(Layer layer)
        {
            _f = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _df = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _dfData = new MatrixColPool(layer.NeuronsCount, data.Input.Count);
            _fData = new MatrixColPool(layer.NeuronsCount, data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new ArcTanActivationFunction(_f,_df,_fData,_dfData);
        }
    }
}