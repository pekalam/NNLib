using System;
using System.Collections.Generic;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    internal class NetDataMatrixPool
    {
        private readonly Dictionary<int,Matrix<double>> _pool = new Dictionary<int, Matrix<double>>();

        //cache of size 4 - for training, validation, test and new data. In case of overflow last item is substituted
        private Matrix<double>[] _previous = new Matrix<double>[4];
        private int[] _previousCols = new int[4];
        private int _cacheLastInd = -1;

        private readonly double _defaultValue;
        private readonly int _defaultRows;

        private NetDataMatrixPool(Dictionary<int, Matrix<double>> pool, int[] previousCols, double defaultValue, int defaultRows, int cacheLastInd)
        {
            int i = 0;
            foreach (var (k, v) in pool)
            {
                var matCpy = v.Clone();
                _pool.Add(k, matCpy);
                if (previousCols[i] == k)
                {
                    _previous[i] = matCpy;
                    i++;
                }
            }

            Debug.Assert(i - 1 == cacheLastInd);

            _cacheLastInd = cacheLastInd;
            _defaultRows = defaultRows;
            _defaultValue = defaultValue;
            Array.Copy(previousCols, _previousCols, previousCols.Length);
        }

        public NetDataMatrixPool(int r, int c, double value = 0)
        {
            _defaultRows = r;
            _defaultValue = value;
            AddToPool(c);
        }

        private void AddToPool(int c)
        {
            if (_cacheLastInd != _previousCols.Length - 1)
            {
                _cacheLastInd++;
            }

            _previous[_cacheLastInd] = Matrix<double>.Build.Dense(_defaultRows,c, _defaultValue);
            _previousCols[_cacheLastInd] = c;
            _pool.Add(c, _previous[_cacheLastInd]);
        }

        public Matrix<double> Get(int requestedColumns)
        {
            if (requestedColumns == _previousCols[0]) return _previous[0];
            if (requestedColumns == _previousCols[1]) return _previous[1];
            if (requestedColumns == _previousCols[2]) return _previous[2];
            if (requestedColumns == _previousCols[3]) return _previous[3];

            if (_pool.TryGetValue(requestedColumns, out var mat))
            {
                _previous[_cacheLastInd] = mat;
                _previousCols[_cacheLastInd] = requestedColumns;
                return mat;
            }

            AddToPool(requestedColumns);
            return _previous[_cacheLastInd];
        }

        public NetDataMatrixPool Clone()
        {
            return new NetDataMatrixPool(_pool,_previousCols,_defaultValue, _defaultRows, _cacheLastInd);
        }
    }
}