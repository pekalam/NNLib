using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib.Data
{
    internal class RandomVectorSetEnumerator : IEnumerator<Matrix<double>>
    {
        private readonly IVectorSet _vectorSet;
        private int _index = -1;
        private int[] _indexTable;

        public RandomVectorSetEnumerator(IVectorSet vectorSet)
        {
            _vectorSet = vectorSet;
            _indexTable = GenerateIndexTable();
        }

        private int[] GenerateIndexTable()
        {
            var rnd = new Random();
            return Enumerable.Range(0, _vectorSet.Count).OrderBy(_ => rnd.Next(0, _vectorSet.Count)).ToArray();
        }

        public bool MoveNext()
        {
            _index++;
            if (_index < _vectorSet.Count)
            {
                Current = _vectorSet[_indexTable[_index]];
                return true;
            }

            return false;
        }

        public void Reset()
        {
            _index = -1;
            _indexTable = GenerateIndexTable();
        }

        object? IEnumerator.Current => Current;

        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public Matrix<double> Current { get; private set; } = null!;
    }


    internal class VectorSetEnumerator : IEnumerator<Matrix<double>>
    {
        private readonly IVectorSet _vectorSet;
        private int _index = -1;

        public VectorSetEnumerator(IVectorSet vectorSet)
        {
            _vectorSet = vectorSet;
        }

        public bool MoveNext()
        {
            _index++;
            if (_index < _vectorSet.Count)
            {
                Current = _vectorSet[_index];
                return true;
            }

            return false;
        }

        public void Reset()
        {
            _index = -1;
        }

        public Matrix<double> Current { get; private set; } = null!;

        object? IEnumerator.Current => Current;

        public void Dispose()
        {
        }
    }

    /// <summary>
    /// Contains set of vectors (input or target) from training data.
    /// </summary>
    public interface IVectorSet : IDisposable
    {
        /// <summary>
        /// Returns vector from set with given index
        /// </summary>
        Matrix<double> this[int index] { get; set; }
        int Count { get; }

        bool Modified { get; }

        public IEnumerator<Matrix<double>> GetEnumerator()
        {
            return new VectorSetEnumerator(this);
        }
    }
}