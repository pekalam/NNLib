using System;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class GradientDescentTests : TrainerTestBase
    {
        MLPNetwork net;

        public GradientDescentTests(ITestOutputHelper output) : base(output)
        {
            net = CreateNetwork(2, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
        }
        
        [Fact]
        public void MLP_approximates_AND_gate_with_online_GD()
        {
            TestAndGate(net, new GradientDescentAlgorithm( new GradientDescentParams()
            {
                Momentum = 0.9, LearningRate = 0.2,
            }), new QuadraticLossFunction(), new BatchParams(){BatchSize = 1}, TimeSpan.FromMinutes(1), varianceCheck: false);
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_minibatch_GD()
        {
            TestAndGate(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
            }), new QuadraticLossFunction(),new BatchParams(){BatchSize = 2}, TimeSpan.FromMinutes(1), varianceCheck: false);
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_batch_GD()
        {
            TestAndGate(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
            }), new QuadraticLossFunction(),new BatchParams(){BatchSize = 4}, TimeSpan.FromMinutes(2), varianceCheck: false);
        }


        [Fact]
        public async Task MLP_approximates_AND_gate_with_online_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
            }), new QuadraticLossFunction(),new BatchParams(){BatchSize = 1}, TimeSpan.FromMinutes(1), varianceCheck:false);
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_minibatch_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
            }), new QuadraticLossFunction(),new BatchParams(){BatchSize = 2}, TimeSpan.FromMinutes(1), varianceCheck: false);
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_batch_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
            }), new QuadraticLossFunction(),new BatchParams(){BatchSize = 4}, TimeSpan.FromMinutes(2), varianceCheck: false);
        }
    }
}