using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
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

        private void CheckVariance(List<double> samples)
        {
            var m = Vector<double>.Build.Dense(samples.ToArray());
            var variance = Math.Sqrt(m.Variance());
            var mean = m.Mean();
            _output.WriteLine("Variance: " + variance);
            _output.WriteLine("Mean: " + mean);


            int eq = 0;
            for (int i = 0; i < m.Count; i++)
            {
                if (m[i] - variance*2 <= mean && m[i] + variance*2 >= mean) eq++;
            }

            if (eq >= m.Count / 3)
            {
                Assert.False(true, $"({eq}/{m.Count}) error values are equal or same as {mean}");
            }
        }


        private void VerifyTrainingError(double target, MLPTrainer trainer, TimeSpan timeout, int samples = 7_000)
        {
            var src = new CancellationTokenSource(timeout);

            var sampleList = new List<double>();

            while(sampleList.Count != samples)
            {
                trainer.DoEpoch();
                sampleList.Add(trainer.Error);
                _output.WriteLine("Error: " + trainer.Error);
                if (double.IsNaN(trainer.Error)) Assert.False(true, "NaN error value");
                if (src.Token.IsCancellationRequested)
                {
                    Assert.False(true, "Training timeout");
                }

                if (trainer.Error <= target)
                {
                    Assert.True(true, $"Target {target} reached");
                    return;
                }
            }

            CheckVariance(sampleList);
            Assert.True(sampleList.First() - sampleList.Last() > 0.0d, "Error is not decreasing");
        }

        private async Task<Task> VerifyTrainingErrorAsync(double target, MLPTrainer trainer, TimeSpan timeout,
            int samples = 7_000)
        {
            var src = new CancellationTokenSource(timeout);

            var sampleList = new List<double>();

            while (sampleList.Count != samples)
            {
                await trainer.DoEpochAsync(src.Token);
                sampleList.Add(trainer.Error);
                _output.WriteLine("Error: " + trainer.Error);
                if (double.IsNaN(trainer.Error)) Assert.False(true, "NaN error value");
                if (src.Token.IsCancellationRequested)
                {
                    Assert.False(true, "Training timeout");
                }

                if (trainer.Error <= target)
                {
                    Assert.True(true, $"Target {target} reached");
                    return Task.CompletedTask;
                }
            }

            CheckVariance(sampleList);
            Assert.True(sampleList.First() - sampleList.Last() > 0.0d, "Error is not decreasing");

            return Task.CompletedTask;
        }

        
        protected void TestAndGate(MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, BatchParams batchParams, TimeSpan timeout, int samples = 7000)
        {
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                algorithm, lossFunction, batchParams);

            VerifyTrainingError(0.01, trainer, timeout, samples);
        }

        protected async Task TestAndGateAsync(MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, BatchParams batchParams, TimeSpan timeout, int samples = 7000)
        {
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                algorithm, lossFunction, batchParams);

            await VerifyTrainingErrorAsync(0.01, trainer, timeout, samples);
        }
        
    }
}