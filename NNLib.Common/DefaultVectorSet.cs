﻿using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;

[assembly: InternalsVisibleTo("NNLib.Tests")]
namespace NNLib.Common
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

        public DefaultVectorSet Clone()
        {
            return new DefaultVectorSet(_setOfVectors.Select(m => m.Clone()).ToList());
        }
    }
}