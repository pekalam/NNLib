using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Moq;
using NNLib.ActivationFunction;
using NNLib.Training;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class TrainerTestBase
    {
        private readonly ITestOutputHelper _output;

        public TrainerTestBase(ITestOutputHelper output)
        {
            _output = output;
        }

        protected MLPNetwork CreateNetwork(int inputs,
            params (int neuronsCount, IActivationFunction activationFunction)[] layers)
        {
            var netLayers = new PerceptronLayer[layers.Length];
            var inputLayer = new PerceptronLayer(inputs, layers[0].neuronsCount, layers[0].activationFunction);

            netLayers[0] = inputLayer;

            for (int i = 1; i < layers.Length; i++)
            {
                var layer = new PerceptronLayer(netLayers[i - 1].NeuronsCount, layers[i].neuronsCount,
                    layers[i].activationFunction);
                netLayers[i] = layer;
                i++;
            }

            var net = new MLPNetwork(netLayers);
            return net;
        }

        protected (Mock<MLPNetwork> net, List<Mock<PerceptronLayer>> layerMocks) CreateMockNetwork(int inputs, params (int neuronsCount, IActivationFunction activationFunction)[] layers)
        {
            var netLayers = new PerceptronLayer[layers.Length];
            var inputLayer = new Mock<PerceptronLayer>(inputs, layers[0].neuronsCount, layers[0].activationFunction);

            var layerMocks = new List<Mock<PerceptronLayer>>();

            netLayers[0] = inputLayer.Object;

            
            for(int i = 1; i < layers.Length; i++)
            {
                var mockLayer = new Mock<PerceptronLayer>(netLayers[i - 1].NeuronsCount, layers[i].neuronsCount,
                    layers[i].activationFunction);
                mockLayer.CallBase = true;
                layerMocks.Add(mockLayer);
                netLayers[i] = mockLayer.Object;
                i++;
            }

            var net = new Mock<MLPNetwork>(netLayers);
            net.CallBase = true;
            return (net, layerMocks);
        }


        private void VerifyTrainingError(double target, MLPTrainer trainer, ITestOutputHelper output, TimeSpan timeout, int samples = 7_000)
        {
            var src = new CancellationTokenSource(timeout);

            var sampleList = new List<double>();

            while(sampleList.Count != samples)
            {
                trainer.DoEpoch();
                sampleList.Add(trainer.Error);
                output.WriteLine("Error: " + trainer.Error);
                if (src.Token.IsCancellationRequested)
                {
                    Assert.False(true, "Training timeout");
                }
            }

            Assert.True(sampleList.First() - sampleList.Last() > 0.0d);
        }

        private async Task<Task> VerifyTrainingErrorAsync(double target, MLPTrainer trainer, ITestOutputHelper output, TimeSpan timeout,
            int samples = 7_000)
        {
            var src = new CancellationTokenSource(timeout);

            var sampleList = new List<double>();

            while (sampleList.Count != samples)
            {
                await trainer.DoEpochAsync(src.Token);
                sampleList.Add(trainer.Error);
                output.WriteLine("Error: " + trainer.Error);
                if (src.Token.IsCancellationRequested)
                {
                    Assert.False(true, "Training timeout");
                }
            }

            Assert.True(sampleList.First() - sampleList.Last() > 0.0d);

            return Task.CompletedTask;
        }

        
        protected void TestAndGate(MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, TimeSpan timeout)
        {
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                algorithm, lossFunction);

            VerifyTrainingError(0.01, trainer, _output, timeout);
        }

        protected async Task TestAndGateAsync(MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, TimeSpan timeout)
        {
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                algorithm, lossFunction);

            await VerifyTrainingErrorAsync(0.01, trainer, _output, timeout);
        }
        
    }
}