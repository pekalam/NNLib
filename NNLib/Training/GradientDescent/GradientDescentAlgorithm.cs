using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class GradientDescentAlgorithm : AlgorithmBase
    {
        private LearningMethodResult? _previousLearningMethodResult;
        private MLPNetwork? _network;
        private ILossFunction? _lossFunction;
        private int _iterations;

        public GradientDescentAlgorithm(GradientDescentParams parameters)
        {
            Params = parameters;
        }

        public GradientDescentParams Params { get; set; }
        public BatchTrainer? BatchTrainer { get; set; }
        internal override int Iterations => _iterations;

        internal override void Setup(Common.SupervisedSet trainingData, MLPNetwork network, ILossFunction lossFunction)
        {
            Guards._NotNull(trainingData).NotNull(network).NotNull(lossFunction);
            _previousLearningMethodResult = null;
            _lossFunction = lossFunction;
            _network = network;
            _iterations = 0;
            BatchTrainer = new BatchTrainer(Params.BatchSize, trainingData, Params.Randomize);
        }

        internal override void Reset()
        {
            _iterations = 0;
            BatchTrainer?.Reset();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Matrix<double> CalcUpdate(int layerInd, PerceptronLayer layer, LearningMethodResult result,
            Matrix<double> input, Matrix<double> next)
        {
            var outputDerivative = layer.ActivationFunction.DerivativeY(layer.Output!);
            var delta = next.PointwiseMultiply(outputDerivative);
            var deltaLr = delta.Multiply(Params.LearningRate);
            var biasesDelta = deltaLr;
            var weightsDelta = deltaLr.TransposeAndMultiply(input);

            result.Weights[layerInd] = weightsDelta;
            result.Biases[layerInd] = biasesDelta;

            return delta;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsAndBiasesWithDeltaRule(LearningMethodResult result)
        {
            Debug.Assert(_network != null, nameof(_network) + " != null");

            if (result.Weights.Length != result.Biases.Length)
            {
                throw new Exception();
            }
        
            for (int i = 0; i < result.Weights.Length; i++)
            {
                _network.Layers[i].Weights.Subtract(result.Weights[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Subtract(result.Biases[i], _network.Layers[i].Biases);
            }
        }

        private LearningMethodResult CalculateDelta(Matrix<double> input, Matrix<double> expected)
        {
            Debug.Assert(_network != null && _lossFunction != null, "Setup was not called");

            var learningResult = LearningMethodResult.FromNetwork(_network);

            _network.CalculateOutput(input);

            Matrix<double> next = _lossFunction.Derivative(_network.Layers[^1].Output!, expected);
            for (var i = _network.Layers.Count - 1; i >= 0; --i)
            {
                var layer = _network.Layers[i];

                var previousDelta = CalcUpdate(i, layer, learningResult, i > 0 ? _network.Layers[i - 1].Output! : input, next);

                next = layer.Weights.TransposeThisAndMultiply(previousDelta);


                if (_previousLearningMethodResult != null)
                {
                    _previousLearningMethodResult.Weights[i] = _previousLearningMethodResult.Weights[i]
                        .Multiply(Params.Momentum);
                    learningResult.Weights[i].Add(_previousLearningMethodResult.Weights[i], learningResult.Weights[i]);
                }
            }

            if (Params.Momentum > 0d)
            {
                _previousLearningMethodResult = learningResult;
            }

            return learningResult;
        }

        internal override bool DoIteration(in CancellationToken ct = default)
        { 
            var result = BatchTrainer!.DoIteration(CalculateDelta);
            _iterations++;

            if (result != null)
            {
                UpdateWeightsAndBiasesWithDeltaRule(result);
                return true;
            }

            return false;
        }
    }
}