﻿using MathNet.Numerics.LinearAlgebra;

namespace NNLib.ActivationFunction
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        public Matrix<double> Function(Matrix<double> x)
        {
            var exp = x.Negate().PointwiseExp();
            var result = 1 / (1 + exp);
            return result;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            var exp = x.Negate().PointwiseExp();
            var s = 1 / (1 + exp);
            var result = s.PointwiseMultiply(1 - s);
            return result;
        }
    }
}