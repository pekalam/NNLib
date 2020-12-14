using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace NNLib
{
    internal class MatrixColPool
    {
        private readonly Dictionary<int,Matrix<double>> _pool = new Dictionary<int, Matrix<double>>();

        //cache of size 4 - for training, validation, test and user data. In case of overflow last item is substituted.
        private Matrix<double>[] _prevMat = new Matrix<double>[4];
        private int[] _prevCols = new int[4];
        private int _cacheLastInd = -1;

        private readonly double _defaultValue;
        private readonly int _defaultRows;

        private MatrixColPool(Dictionary<int, Matrix<double>> pool, int[] previousCols, double defaultValue, int defaultRows, int cacheLastInd)
        {
            int i = 0;
            foreach (var (k, v) in pool)
            {
                var matCpy = v.Clone();
                _pool.Add(k, matCpy);
                if (previousCols[i] == k)
                {
                    _prevMat[i] = matCpy;
                    i++;
                }
            }

            Debug.Assert(i - 1 == cacheLastInd);

            _cacheLastInd = cacheLastInd;
            _defaultRows = defaultRows;
            _defaultValue = defaultValue;
            Array.Copy(previousCols, _prevCols, previousCols.Length);
        }

        public MatrixColPool(int r, int c, double value = 0)
        {
            _defaultRows = r;
            _defaultValue = value;
            AddToPool(c);
        }

        private void AddToPool(int c)
        {
            if (_cacheLastInd != _prevCols.Length - 1)
            {
                _cacheLastInd++;
            }
            //fill cache
            _prevMat[_cacheLastInd] = Matrix<double>.Build.Dense(_defaultRows,c, _defaultValue);
            _prevCols[_cacheLastInd] = c;
            //add to pool
            _pool.Add(c, _prevMat[_cacheLastInd]);
        }

        public Matrix<double> Get(int requestedColumns)
        {
            if (requestedColumns == _prevCols[0]) return _prevMat[0];
            if (requestedColumns == _prevCols[1]) return _prevMat[1];
            if (requestedColumns == _prevCols[2]) return _prevMat[2];
            if (requestedColumns == _prevCols[3]) return _prevMat[3];

            if (_pool.TryGetValue(requestedColumns, out var mat))
            {
                //substitute last item of cache
                _prevMat[_cacheLastInd] = mat;
                _prevCols[_cacheLastInd] = requestedColumns;
                return mat;
            }

            AddToPool(requestedColumns);
            return _prevMat[_cacheLastInd];
        }

        public MatrixColPool Clone()
        {
            return new MatrixColPool(_pool,_prevCols,_defaultValue, _defaultRows, _cacheLastInd);
        }
    }
}