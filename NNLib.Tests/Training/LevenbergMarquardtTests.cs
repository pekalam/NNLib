using System;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class LevenbergMarquardtTests : TrainerTestBase
    {
        MLPNetwork net;

        public LevenbergMarquardtTests(ITestOutputHelper output) : base(output)
        {
            net = CreateNetwork(2, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
        }
        
        [Fact]
        public void MLP_approximates_AND_gate_with_online_GD()
        {
            TestAndGate(net, new LevenbergMarquardtAlgorithm(new LevenbergMarquardtParams()), new QuadraticLossFunction(), new BatchParams(),  TimeSpan.FromMinutes(1), 20_000);
        }
    }
}