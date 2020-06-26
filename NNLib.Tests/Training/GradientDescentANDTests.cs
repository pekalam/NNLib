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
                    LearningRate = 0.0001, Momentum = 0.1, BatchParams = { BatchSize = 1 }
                }), new QuadraticLossFunction(), TimeSpan.FromMinutes(2), samples: 20_000,varianceCheck:false);
        }

        [Theory]
        [InlineData(0, "sin.csv")]
        public void f(int netNum, string fileName)
        {
            var net = _networks[netNum];
            TestFromCsv(fileName, net, new LevenbergMarquardtAlgorithm(
                new LevenbergMarquardtParams()
                {
                    Eps = 0.01, DampingParamFactor = 1.1
                }), new QuadraticLossFunction(), TimeSpan.FromMinutes(2), samples: 1_000, varianceCheck: false);
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
                Momentum = 0.3, LearningRate = 0.002,BatchParams = { BatchSize = 1}
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(1), varianceCheck: false);
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_minibatch_GD()
        {
            TestAndGate(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchParams = { BatchSize = 2 },
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(1), varianceCheck: false);
        }

        [Fact]
        public void MLP_approximates_AND_gate_with_batch_GD()
        {
            TestAndGate(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchParams = { BatchSize = 4 },
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(2), varianceCheck: false);
        }


        [Fact]
        public async Task MLP_approximates_AND_gate_with_online_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchParams = { BatchSize = 1 },
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(1), varianceCheck:false);
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_minibatch_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchParams = { BatchSize = 2 },
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(1), varianceCheck: false);
        }

        [Fact]
        public async Task MLP_approximates_AND_gate_with_batch_GD_async()
        {
            await TestAndGateAsync(net,new GradientDescentAlgorithm(new GradientDescentParams()
            {
                Momentum = 0.3,
                LearningRate = 0.002,
                BatchParams = { BatchSize = 2 },
            }), new QuadraticLossFunction(), TimeSpan.FromMinutes(2), varianceCheck: false);
        }
    }
}