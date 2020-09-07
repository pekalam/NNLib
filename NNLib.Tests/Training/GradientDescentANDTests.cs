using System;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class TrainingAlgorithmsTests : TrainerTestBase
    {
        private static MLPNetwork[] _networks = new MLPNetwork[]
        {
            CreateNetwork(1, (2, new TanHActivationFunction()), (90, new TanHActivationFunction()), (1, new SigmoidActivationFunction()))
        };

        public TrainingAlgorithmsTests(ITestOutputHelper output) : base(output)
        {
        }

        [Theory]
        [InlineData(0, "sin.csv")]
        public void GradientDescent_tests(int netNum, string fileName)
        {
            var net = _networks[netNum];
            TestFromCsv(fileName, net, new GradientDescentAlgorithm(
                new GradientDescentParams()
                {
                    LearningRate = 0.0001, Momentum = 0.1, BatchSize = 1
                }), new QuadraticLossFunction(), TimeSpan.FromSeconds(5), epochs: 1_000);
        }

        [Theory]
        [InlineData(0, "sin.csv")]
        public void f(int netNum, string fileName)
        {
            var net = _networks[netNum];
            TestFromCsv(fileName, net, new LevenbergMarquardtAlgorithm(
                new LevenbergMarquardtParams()
                {
                    DampingParamIncFactor = 11, DampingParamDecFactor = 0.1
                }), new QuadraticLossFunction(), TimeSpan.FromSeconds(60), epochs: 1_000);
        }
    }

    public class GradientDescentANDTests : TrainerTestBase
    {
        MLPNetwork net;

        public GradientDescentANDTests(ITestOutputHelper output) : base(output)
        {
            net = CreateNetwork(2, (2, new LinearActivationFunction()), (1, new SigmoidActivationFunction()));
        }
        
        [Fact]
        public void MLP_approximates_AND_gate_with_online_GD()
        {
            TestAndGate(net, new GradientDescentAlgorithm( new GradientDescentParams()
            {
                Momentum = 0.3, LearningRate = 0.002,BatchSize = 1
            }), new QuadraticLossFunction(), TimeSpan.FromSeconds(5), epochs: 1_000);
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_minibatch_GD()
        {
            TestAndGate(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchSize = 2 
            }), new QuadraticLossFunction(), TimeSpan.FromSeconds(5), epochs: 1_000);
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_batch_GD()
        {
            TestAndGate(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchSize = 4
            }), new QuadraticLossFunction(), TimeSpan.FromSeconds(5), epochs: 1_000);
        }


        [Fact]
        public async Task MLP_approximates_AND_gate_with_online_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchSize = 1,
            }), new QuadraticLossFunction(), TimeSpan.FromSeconds(5), epochs: 1_000);
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_minibatch_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchSize = 2,
            }), new QuadraticLossFunction(), TimeSpan.FromSeconds(5), epochs: 1_000);
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_batch_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchSize = 2,
            }), new QuadraticLossFunction(), TimeSpan.FromSeconds(5), epochs: 1_000);
        }
    }
}