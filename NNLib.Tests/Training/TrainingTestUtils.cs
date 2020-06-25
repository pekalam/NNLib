using NNLib.Common;
using M = MathNet.Numerics.LinearAlgebra.Matrix<double>;

namespace NNLib.Tests
{
    public static class TrainingTestUtils
    {
        public static SupervisedSet AndGateSet()
        {
            var input = new[]
            {
                new []{0d,0d},
                new []{0d,1d},
                new []{1d,0d},
                new []{1d,1d},
            };

            var expected = new[]
            {
                new []{0d},
                new []{0d},
                new []{0d},
                new []{1d},
            };

            return SupervisedSet.FromArrays(input, expected);
        }

        public static bool CompareTo(this M m1, M m2)
        {
            if (m1.RowCount != m2.RowCount || m1.ColumnCount != m2.ColumnCount)
            {
                return false;
            }

            for (int i = 0; i < m1.RowCount; i++)
            {
                for (int j = 0; j < m1.ColumnCount; j++)
                {
                    if (m1[i, j] != m2[i, j])
                    {
                        return false;
                    }
                }
            }

            return true;
        }
    }
}