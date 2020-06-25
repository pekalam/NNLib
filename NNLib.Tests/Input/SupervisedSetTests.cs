using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;
using Xunit;

namespace NNLib.Tests
{
    public class SupervisedSetTests
    {
        [Fact]
        public void ctor_when_vector_sets_differ_in_length_throws()
        {
            Assert.Throws<ArgumentException>(() => new SupervisedSet(new DefaultVectorSet(new List<Matrix<double>>()
            {
                Matrix<double>.Build.Random(2,1), Matrix<double>.Build.Random(2,1),
            }), new DefaultVectorSet(new List<Matrix<double>>()
            {
                Matrix<double>.Build.Random(2,1),
            })));
        }
    }
}