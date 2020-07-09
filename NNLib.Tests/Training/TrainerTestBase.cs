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


        private void VerifyTrainingError(double target, MLPTrainer trainer, TimeSpan timeout, int epochs)
        {
            var src = new CancellationTokenSource(timeout);


            while (trainer.Epochs != epochs)
            {
                trainer.DoEpoch();
                _output.WriteLine("Error: " + trainer.Error);
                if (double.IsNaN(trainer.Error))
                {
                    _output.WriteLine("NaN error");
                    break;
                }
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
        }

        private async Task<Task> VerifyTrainingErrorAsync(double target, MLPTrainer trainer, TimeSpan timeout,
            int epochs)
        {
            var src = new CancellationTokenSource(timeout);

            
            while (trainer.Epochs != epochs)
            {
                await trainer.DoEpochAsync(src.Token);
                _output.WriteLine("Error: " + trainer.Error);
                if (double.IsNaN(trainer.Error))
                {
                    _output.WriteLine("NaN error");
                    break;
                }
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

            return Task.CompletedTask;
        }

        protected void TestFromCsv(string fileName, MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, TimeSpan timeout, int epochs = 7000)
        {
            var trainer = new MLPTrainer(net, CsvFacade.LoadSets(fileName).sets,
                algorithm, lossFunction);

            VerifyTrainingError(0.01, trainer, timeout, epochs);
        }
        
        protected void TestAndGate(MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, TimeSpan timeout, int epochs = 7000)
        {
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                algorithm, lossFunction);

            VerifyTrainingError(0.01, trainer, timeout, epochs);
        }

        protected async Task TestAndGateAsync(MLPNetwork net, AlgorithmBase algorithm, ILossFunction lossFunction, TimeSpan timeout, int epochs = 7000)
        {
            var trainer = new MLPTrainer(net, new SupervisedTrainingSets(TrainingTestUtils.AndGateSet()),
                algorithm, lossFunction);

            await VerifyTrainingErrorAsync(0.01, trainer, timeout, epochs);
        }
        
    }
}