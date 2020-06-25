using System;
using System.Collections;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib.Common
{
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

        public Matrix<double> Current { get; private set; }

        object? IEnumerator.Current => Current;

        public void Dispose()
        {
        }
    }

    public interface IVectorSet : IDisposable
    {
        Matrix<double> this[int index] { get; set; }
        int Count { get; }

        public IEnumerator<Matrix<double>> GetEnumerator()
        {
            return new VectorSetEnumerator(this);
        }
    }
}