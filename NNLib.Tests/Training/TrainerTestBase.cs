using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using Moq;
using NNLib.Common;
using NNLib.Csv;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class TrainerTestBase
    {
        protected readonly ITestOutputHelper _output;

        public TrainerTestBase(ITestOutputHelper output)
        {
            _output = output;
        }

        protected static MLPNetwork CreateNetwork(int inputs,
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
            }

            var net = new MLPNetwork(netLayers);
            return net;
        }

        protected static (Mock<MLPNetwork> net, List<Mock<PerceptronLayer>> layerMocks) CreateMockNetwork(int inputs, params (int neuronsCount, IActivationFunction activationFunction)[] layers)
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

            if (variance < 0.001)
            {
                Assert.False(true, $"Too low variance");
            }
        }


        private void VerifyTrainingError(double target, MLPTrainer trainer, TimeSpan timeout, int samples, bool varianceCheck)
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

            if(varianceCheck) CheckVariance(sampleList);
            Assert.True(sampleList.First() - sampleList.Last() > 0.0d, "Error is not decreasing");
        }

        private async Task<Task> VerifyTrainingErrorAsync(double target, MLPTrainer trainer, TimeSpan timeout,
            int samples, bool varianceCheck)
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

            if (varianceCheck) CheckVariance(sampleList);
            Assert.True(sampleList.First() - sampleList.Last() > 0.0d, "Error is not decreasing");

            return Task.CompletedTask;
        }

        protected void TestFromCsv(string fileName, MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, TimeSpan timeout, int samples = 7000, bool varianceCheck = true)
        {
            var trainer = new MLPTrainer(net, CsvFacade.LoadSets(fileName).sets,
                algorithm, lossFunction);

            VerifyTrainingError(0.01, trainer, timeout, samples, varianceCheck);
        }
        
        protected void TestAndGate(MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, TimeSpan timeout, int samples = 7000, bool varianceCheck = true)
        {
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                algorithm, lossFunction);

            VerifyTrainingError(0.01, trainer, timeout, samples, varianceCheck);
        }

        protected async Task TestAndGateAsync(MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, TimeSpan timeout, int samples = 7000, bool varianceCheck = true)
        {
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                algorithm, lossFunction);

            await VerifyTrainingErrorAsync(0.01, trainer, timeout, samples, varianceCheck);
        }
        
    }
}