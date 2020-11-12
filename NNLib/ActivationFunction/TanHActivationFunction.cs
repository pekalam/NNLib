using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class TanHActivationFunction : IActivationFunction
    {
        private Matrix<double> _f;
        private Matrix<double> _df;

        private NetDataMatrixPool? _fData;
        private NetDataMatrixPool? _dfData;

        public TanHActivationFunction()
        {
            
        }

        private TanHActivationFunction(Matrix<double> f, Matrix<double> df, NetDataMatrixPool? fData, NetDataMatrixPool? dfData)
        {
            _f = f.Clone();
            _df = df.Clone();
            _fData = fData?.Clone();
            _dfData = dfData?.Clone();
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            var cols = x.ColumnCount;

            Matrix<double> storage = cols == _f.ColumnCount ? _f : _fData!.Get(cols);

            x.PointwiseTanh(storage);
            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            var cols = x.ColumnCount;

            Matrix<double> storage = cols == _df.ColumnCount ? _df : _dfData!.Get(cols);

            x.PointwiseTanh(storage);
            storage.PointwisePower(2, storage);
            storage.Negate(storage);
            storage.Add(1, storage);

            return storage;
        }

        public void InitMemory(Layer layer)
        {
            _f = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _df = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _fData = new NetDataMatrixPool(layer.NeuronsCount, data.Input.Count);
            _dfData = new NetDataMatrixPool(layer.NeuronsCount, data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new TanHActivationFunction(_f,_df,_fData,_dfData);
        }
    }
}