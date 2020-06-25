using System;
using Xunit;

namespace NNLib.Tests
{
    public class GradientDescendParamsTest
    {
        [Fact]
        public void GradientDescendLearningParameters_throws_when_invalid_params()
        {
            var par = new GradientDescentParams();
            Assert.Throws<InvalidOperationException>(() => par.LearningRate = 0);
            Assert.Throws<InvalidOperationException>(() => par.LearningRate = -1);
            Assert.Throws<InvalidOperationException>(() => par.LearningRate = double.PositiveInfinity);
            Assert.Throws<InvalidOperationException>(() => par.LearningRate = double.NegativeInfinity);
            Assert.Throws<InvalidOperationException>(() => par.LearningRate = double.NaN);

            par.Momentum = 0;
            Assert.Throws<InvalidOperationException>(() => par.Momentum = -1);
            Assert.Throws<InvalidOperationException>(() => par.Momentum = double.PositiveInfinity);
            Assert.Throws<InvalidOperationException>(() => par.Momentum = double.NegativeInfinity);
            Assert.Throws<InvalidOperationException>(() => par.Momentum = double.NaN);

        }
    }
}