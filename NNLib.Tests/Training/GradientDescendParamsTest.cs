using System;
using NNLib.Training.GradientDescent;
using Xunit;

namespace NNLib.Tests
{
    public class GradientDescendParamsTest
    {
        [Fact]
        public void GradientDescendLearningParameters_throws_when_invalid_params()
        {
            var par = new GradientDescentParams();
            Assert.Throws<ArgumentException>(() => par.LearningRate = 0);
            Assert.Throws<ArgumentException>(() => par.LearningRate = -1);
            Assert.Throws<ArgumentException>(() => par.LearningRate = double.PositiveInfinity);
            Assert.Throws<ArgumentException>(() => par.LearningRate = double.NegativeInfinity);
            Assert.Throws<ArgumentException>(() => par.LearningRate = double.NaN);

            par.Momentum = 0;
            Assert.Throws<ArgumentException>(() => par.Momentum = -1);
            Assert.Throws<ArgumentException>(() => par.Momentum = double.PositiveInfinity);
            Assert.Throws<ArgumentException>(() => par.Momentum = double.NegativeInfinity);
            Assert.Throws<ArgumentException>(() => par.Momentum = double.NaN);

        }
    }
}