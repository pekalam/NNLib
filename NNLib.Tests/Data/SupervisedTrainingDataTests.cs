using System;
using NNLib.Data;
using Xunit;

namespace NNLib.Tests.Data
{
    public class SupervisedTrainingDataTests
    {
        [Fact]
        public void ctor_when_training_set_null_throws()
        {
            Assert.Throws<NullReferenceException>(() =>
                new SupervisedTrainingData(null));
        }
    }
}