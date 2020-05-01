using System;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class GradientDescent
    {
        private GradientDescentLearningParameters _parameters;
        private SupervisedSet _trainingSet;
        private int _setIndex;
        private readonly GradientDescentAlgorithm _gradientDescent;
        private LearningMethodResult[] _methodResults;

        public GradientDescent(GradientDescentLearningParameters parameters)
        {
            Guards._NotNull(parameters);
            _parameters = parameters;
            _gradientDescent = new GradientDescentAlgorithm(parameters);
        }

        public SupervisedSet TrainingSet
        {
            get => _trainingSet;
            set
            {
                Guards._NotNull(_parameters);
                ValidateParamsForSet(value);
                ResetTrainingVars();
                IterationsPerEpoch = value.Input.Count / _parameters.BatchSize;
                _methodResults = new LearningMethodResult[IterationsPerEpoch];
                _trainingSet = value;
            }
        }

        public GradientDescentLearningParameters Parameters
        {
            get => _parameters;
            set
            {
                Guards._NotNull(_trainingSet);
                ValidateParamsForSet(_trainingSet);
                ResetTrainingVars();
                IterationsPerEpoch = _trainingSet.Input.Count / value.BatchSize;
                _methodResults = new LearningMethodResult[IterationsPerEpoch];
                _parameters = value;
            }
        }

        private void ValidateParamsForSet(SupervisedSet set)
        {
            if (_parameters.BatchSize > set.Input.Count)
            {
                throw new ArgumentException($"Invalid batch size {_parameters.BatchSize} for training set with count {set.Input.Count}");
            }

            if (set.Input.Count % _parameters.BatchSize != 0)
            {
                //TODO
                throw new ArgumentException($"Cannot divide training set");
            }
        }


        private void CheckTrainingCancelationIsRequested(in CancellationToken ct)
        {
            if (ct.IsCancellationRequested)
            {
                throw new TrainingCanceledException();
            }
        }

        private void ResetTrainingVars()
        {
            Iterations = IterationsPerEpoch = CurrentBatch = _setIndex = 0;
        }

        public int Iterations { get; private set; }
        public int IterationsPerEpoch { get; private set; }
        public int CurrentBatch { get; private set; }

        private LearningMethodResult EndEpoch()
        {
            var result = _methodResults[0];

            for (int i = 1; i < _methodResults.Length; i++)
            {
                for (int j = 0; j < result.Weigths.Count; j++)
                {
                    result.Weigths[j] = result.Weigths[j] + _methodResults[i].Weigths[j];
                }

                for (int j = 0; j < result.Biases.Count; j++)
                {
                    result.Biases[j] = result.Biases[j] + _methodResults[i].Biases[j];
                }
            }

            return result;
        }

        public LearningMethodResult DoIteration(MLPNetwork network, ILossFunction lossFunction, in CancellationToken ct = default)
        {
            var input = _trainingSet.Input[_setIndex];
            var expected = _trainingSet.Target[_setIndex];

            network.CalculateOutput(input);

            CheckTrainingCancelationIsRequested(ct);

            var result = _gradientDescent.CalculateDelta(network, input, expected, lossFunction);
            _methodResults[Iterations] = result;

            _setIndex++;
            _setIndex %= _trainingSet.Input.Count;

            CurrentBatch = ++CurrentBatch % (_trainingSet.Input.Count / _parameters.BatchSize);

            Iterations++;
            if (Iterations == IterationsPerEpoch)
            {
                Iterations = 0;
                return EndEpoch();
            }

            return null;
        }

        public LearningMethodResult DoEpoch(MLPNetwork network, ILossFunction lossFunction, in CancellationToken ct = default)
        {
            for (int i = 0; i < IterationsPerEpoch - 1; i++)
            {
                CheckTrainingCancelationIsRequested(ct);
                DoIteration(network, lossFunction, ct);
            }
            CheckTrainingCancelationIsRequested(ct);
            var result = DoIteration(network, lossFunction, ct);
            return result;
        }

        public Task<LearningMethodResult> DoEpochAsync(MLPNetwork network, ILossFunction lossFunction, CancellationToken ct = default)
        {
            return Task.Run(() =>
            {
                return DoEpoch(network, lossFunction, ct);
            }, ct);
        }
    }


    internal class GradientDescentAlgorithm
    {
        private LearningMethodResult _previousLearningMethodResult;

        public GradientDescentAlgorithm(GradientDescentLearningParameters learningParameters)
        {
            LearningParameters = learningParameters;
        }

        public GradientDescentLearningParameters LearningParameters { get; set; }

        private void CalculateDeltaForOutputLayer(MLPNetwork network, int layerInd, ILossFunction lossFunction, LearningMethodResult result,
            Matrix<double> input, Matrix<double> expected, out Matrix<double> previousDelta)
        {
            var layer = network.Layers[layerInd];
            var layerInput = layer.IsInputLayer ? input : network.Layers[layerInd - 1].Output;

            var lossFuncDerivative = lossFunction.Derivative(layer.Output, expected);
            var outputDerivative = layer.ActivationFunction.DerivativeY(layer.Output);

            var delta = lossFuncDerivative.PointwiseMultiply(outputDerivative);
            var biasesDelta = delta.Multiply(LearningParameters.LearningRate);
            previousDelta = delta;

            
            var weightsDelta = delta.TransposeAndMultiply(layerInput).Multiply(LearningParameters.LearningRate);

            if (_previousLearningMethodResult != null)
            {
                _previousLearningMethodResult.Weigths[layerInd] = _previousLearningMethodResult.Weigths[layerInd]
                    .Multiply(LearningParameters.Momentum);
                weightsDelta.Add(_previousLearningMethodResult.Weigths[layerInd], weightsDelta);
            }

            result.Weigths[layerInd] = weightsDelta;
            result.Biases[layerInd] = biasesDelta;
        }

        private void CalculateDeltaForInnerLayer(MLPNetwork network, int layerInd, LearningMethodResult result,
            Matrix<double> input, ref Matrix<double> previousDelta)
        {
            var layer = network.Layers[layerInd];
            var nextLayer = network.Layers[layerInd + 1];
            var layerInput = layer.IsInputLayer ? input : network.Layers[layerInd - 1].Output;

            var prevTransp = nextLayer.Weights.TransposeThisAndMultiply(previousDelta);
            var outputDerivative = layer.ActivationFunction.DerivativeY(layer.Output);

            var delta = prevTransp.PointwiseMultiply(outputDerivative);
            var biasesDelta = delta.Multiply(LearningParameters.LearningRate);
            previousDelta = delta;

            var weightsDelta = delta.TransposeAndMultiply(layerInput).Multiply(LearningParameters.LearningRate);

            if (_previousLearningMethodResult != null)
            {
                _previousLearningMethodResult.Weigths[layerInd] = _previousLearningMethodResult.Weigths[layerInd]
                    .Multiply(LearningParameters.Momentum);
                weightsDelta.Add(_previousLearningMethodResult.Weigths[layerInd], weightsDelta);
            }

            result.Weigths[layerInd] = weightsDelta;
            result.Biases[layerInd] = biasesDelta;
        }

        public LearningMethodResult CalculateDelta(MLPNetwork network, Matrix<double> input, Matrix<double> expected, ILossFunction lossFunction)
        {
            var learningResult = LearningMethodResult.FromNetwork(network);
            Matrix<double> previousDelta = null;

            for (int i = network.Layers.Count - 1; i >= 0; --i)
            {
                var layer = network.Layers[i];

                if (layer.IsOutputLayer)
                {
                    CalculateDeltaForOutputLayer(network, i, lossFunction, learningResult, input, expected, out previousDelta);
                }
                else
                {
                    CalculateDeltaForInnerLayer(network, i, learningResult, input, ref previousDelta);
                }
            }

            if (LearningParameters.Momentum != 0d)
            {
                _previousLearningMethodResult = learningResult;
            }

            return learningResult;
        }
    }
}