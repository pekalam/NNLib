﻿using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib.Common
{
    public partial class SupervisedSet
    {
        private static void ValidateInputAndTargetArrays(double[][] input, double[][] target)
        {
            Guards._GtZero(input.Length).GtZero(target.Length);
            var li = input[0].Length;
            foreach (var i in input)
            {
                if (i.Length != li)
                {
                    throw new Exception();
                }
            }
            var lt = target[0].Length;
            foreach (var t in target)
            {
                if (t.Length != lt)
                {
                    throw new Exception();
                }
            }
        }

        public static Common.SupervisedSet FromArrays(double[][] input, double[][] target)
        {
            ValidateInputAndTargetArrays(input, target);
            var inputVectors = new List<Matrix<double>>();
            foreach (double[] inputVec in input)
            {
                var vector = Vector<double>.Build.DenseOfArray(inputVec).ToColumnMatrix();
                inputVectors.Add(vector);
            }
            var defaultInputVectorSet = new DefaultVectorSet(inputVectors);

            var targetVectors = new List<Matrix<double>>();
            foreach (double[] targetVec in target)
            {
                var vector = Vector<double>.Build.DenseOfArray(targetVec).ToColumnMatrix();
                targetVectors.Add(vector);
            }
            var defaultTargetVectorSet = new DefaultVectorSet(targetVectors);

            return new Common.SupervisedSet(defaultInputVectorSet, defaultTargetVectorSet);
        }

        public static Common.SupervisedSet Empty()
        {
            return new Common.SupervisedSet(new DefaultVectorSet(new List<Matrix<double>>()), new DefaultVectorSet(new List<Matrix<double>>()));
        }
    }
}