using System;
using System.Threading.Tasks;
using NNLib.ActivationFunction;
using NNLib.Training.LevenbergMarquardt;
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
            TestAndGate(net, new LevenbergMarquardtAlgorithm(new LevenbergMarquardtParams(), net), new QuadraticLossFunction(), TimeSpan.FromMinutes(1));
        }
    }
    
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
            TestAndGate(net, new GradientDescentAlgorithm(net, new GradientDescentParams()
            {
                Momentum = 0.9, LearningRate = 0.2, BatchSize = 1
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(1));
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_minibatch_GD()
        {
            TestAndGate(net,new GradientDescentAlgorithm(net, new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 2
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(1));
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_batch_GD()
        {
            TestAndGate(net,new GradientDescentAlgorithm(net,new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 4
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(2));
        }


        [Fact]
        public async Task MLP_approximates_AND_gate_with_online_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(net,new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 1
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(1));
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_minibatch_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(net,new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 2
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(1));
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_batch_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(net,new GradientDescentParams()
            {
                Momentum = 0.9,
                LearningRate = 0.2,
                BatchSize = 4
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(2));
        }
    }
}