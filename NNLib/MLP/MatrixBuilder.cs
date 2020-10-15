using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib.MLP
{
    /// <summary>
    /// Base class for matrix builders which contain strategy for layer parameters initializaion.
    /// </summary>
    public abstract class MatrixBuilder
    {
        protected Layer Layer = null!;
        internal void SetLayer(Layer layer)
        {
            Layer = layer;
        }

        public abstract void BuildAllMatrices(int neuronsCount, int inputsCount);
        /// <summary>
        /// Adjust layer parameters if <see cref="neuronsCount"/> or <see cref="inputsCount"/> of network are changed.
        /// </summary>
        public abstract void AdjustMatrices(int neuronsCount, int inputsCount);
    }


    public class NormDistMatrixBuilder : MatrixBuilder
    {
        public override void BuildAllMatrices(int neuronsCount, int inputsCount)
        {
            Layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new Normal());
            Layer.Biases = Matrix<double>.Build.Dense(neuronsCount, 1, 0);
        }

        public override void AdjustMatrices(int neuronsCount, int inputsCount)
        {
            if (inputsCount > Layer.InputsCount)
            {
                while (inputsCount != Layer.InputsCount)
                {
                    var col = Matrix<double>.Build.Random(Layer.NeuronsCount, 1, new Normal()).Column(0);
                    Layer.Weights = Layer.Weights.InsertColumn(Layer.Weights.ColumnCount, col);
                }
            }
            else if (inputsCount < Layer.InputsCount)
            {
                while (inputsCount != Layer.InputsCount)
                {
                    Layer.Weights = Layer.Weights.RemoveColumn(Layer.Weights.ColumnCount - 1);
                }
            }

            if (neuronsCount > Layer.NeuronsCount)
            {
                while (neuronsCount != Layer.NeuronsCount)
                {
                    var Wrow = Matrix<double>.Build.Random(1, Layer.InputsCount, new Normal()).Row(0);
                    var Brow = Matrix<double>.Build.Dense(1, 1, 0).Column(0);
                    Layer.Weights = Layer.Weights.InsertRow(Layer.Weights.RowCount, Wrow);
                    Layer.Biases = Layer.Biases.InsertRow(Layer.Biases.RowCount, Brow);
                }
            }
            else if (neuronsCount < Layer.NeuronsCount)
            {
                while (neuronsCount != Layer.NeuronsCount)
                {
                    Layer.Weights = Layer.Weights.RemoveRow(Layer.Weights.RowCount - 1);
                    Layer.Biases = Layer.Biases.RemoveRow(Layer.Biases.RowCount - 1);
                }
            }
        }
    }


    public class XavierMatrixBuilder : MatrixBuilder
    {
        public override void BuildAllMatrices(int neuronsCount, int inputsCount)
        {
            var stddev = 1d / inputsCount;
            Layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new Normal(0, stddev));
            Layer.Biases = Matrix<double>.Build.Dense(neuronsCount, 1, 0);
        }

        public override void AdjustMatrices(int neuronsCount, int inputsCount)
        {
            var stddev = 1d / inputsCount;

            if (inputsCount != Layer.InputsCount)
            {
                Layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new Normal(0, stddev));
            }

            if (neuronsCount != Layer.NeuronsCount)
            {
                Layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new Normal(0, stddev));
                Layer.Biases = Matrix<double>.Build.Dense(neuronsCount, 1, 0);
            }
        }
    }
}