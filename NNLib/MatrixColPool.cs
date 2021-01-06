using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace NNLib
{
    internal class MatrixColPool
    {
        private readonly Dictionary<int,Matrix<double>> _pool = new Dictionary<int, Matrix<double>>();

        //cache of size 5 - for column vector, training, validation, test and user data. In case of overflow last item is substituted.
        private Matrix<double>[] _prevMat = new Matrix<double>[5];
        private int[] _prevCols = new int[5];
        //pointing to last item in cache
        private int _cacheLastInd = -1;

        private readonly double _defaultValue;
        private readonly int _defaultRows;

        private MatrixColPool(Dictionary<int, Matrix<double>> pool, int[] previousCols, double defaultValue, int defaultRows, int cacheLastInd)
        {
            int cacheCopiedInd = 0;
            foreach (var (k, v) in pool)
            {
                var matCpy = v.Clone();
                _pool.Add(k, matCpy);
                int ind;
                if ((ind = Array.IndexOf(previousCols,k)) != -1)
                {
                    _prevMat[ind] = matCpy;
                    cacheCopiedInd++;
                }
            }

            Debug.Assert(cacheCopiedInd - 1 == cacheLastInd);

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

        public void ClearOtherThanColumnVec()
        {
            for (int i = 1; i < _prevCols.Length; i++)
            {
                _pool.Remove(_prevCols[i]);
                _prevCols[i] = 0;
                _prevMat[i] = null!;
            }

            _cacheLastInd = 0;
        }

        public void AddToPool(int c)
        {
            if (_pool.ContainsKey(c))
            {
                return;
            }

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
            if (requestedColumns == _prevCols[4]) return _prevMat[4];

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