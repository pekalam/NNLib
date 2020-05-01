using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class ReadMatrixWrapper
    {
        private readonly Matrix<double> _matrix;

        public ReadMatrixWrapper(Matrix<double> matrix)
        {
            _matrix = matrix;
        }

        public static implicit operator ReadMatrixWrapper(Matrix<double> matrix) => new ReadMatrixWrapper(matrix);

        public double this[int r, int c] => _matrix[r,c];

        public int RowCount => _matrix.RowCount;
        public int ColumnCount => _matrix.ColumnCount;
    }
}