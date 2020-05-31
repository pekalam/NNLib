using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("NNLib.Tests")]
namespace NNLib
{
    public class DefaultVectorSet : IVectorSet
    {
        private readonly List<Matrix<double>> _setOfVectors;

        public DefaultVectorSet(List<Matrix<double>> setOfVectors)
        {
            _setOfVectors = setOfVectors;
        }

        public Matrix<double> this[int index]
        {
            get => _setOfVectors[index];
            set => _setOfVectors[index] = value;
        }

        public int Count => _setOfVectors.Count;
        
        public void Dispose()
        {
        }
    }
}