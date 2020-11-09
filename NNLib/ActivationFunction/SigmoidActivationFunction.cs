using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        private Matrix<double> _f;
        private Matrix<double> _df;
        private Matrix<double> _dfClone;

        private Matrix<double> _fData;
        private Matrix<double> _dfData;
        private Matrix<double> _dfDataClone;

        private SigmoidActivationFunction(Matrix<double> f, Matrix<double> df, Matrix<double> dfClone, Matrix<double> fData, Matrix<double> dfData, Matrix<double> dfDataClone)
        {
            _f = f;
            _df = df;
            _dfClone = dfClone;
            _fData = fData;
            _dfData = dfData;
            _dfDataClone = dfDataClone;
        }

        public SigmoidActivationFunction()
        {
            
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            Matrix<double> storage = x.ColumnCount == _f.ColumnCount ? _f : _fData;

            x.Negate(storage);
            storage.PointwiseExp(storage);
            storage.Add(1, storage);
            storage.PointwisePower(-1, storage);

            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            Matrix<double> storage = x.ColumnCount == _df.ColumnCount ? _df : _dfData;
            Matrix<double> storage2 = x.ColumnCount == _df.ColumnCount ? _dfClone : _dfDataClone;

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
            _f = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _df = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _dfClone = _df.Clone();
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _fData = Matrix<double>.Build.Dense(layer.NeuronsCount, data.Input.Count);
            _dfData = Matrix<double>.Build.Dense(layer.NeuronsCount, data.Input.Count);
            _dfDataClone = _dfData.Clone();
        }

        public IActivationFunction Clone()
        {
            return new SigmoidActivationFunction(_f, _df,_dfClone,_fData,_dfData,_dfDataClone);
        }
    }
}