using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using NNLib.LossFunction;
using NNLib.MLP;

namespace NNLib.Training.GradientDescent
{
    public class GradientDescentAlgorithm : AlgorithmBase
    {
        private ParametersUpdate? _previousLearningMethodResult;
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

        internal override void Setup(SupervisedTrainingSamples set, MLPNetwork network, ILossFunction lossFunction)
        {
            Guards._NotNull(set).NotNull(network).NotNull(lossFunction);
            _previousLearningMethodResult = null;
            _lossFunction = lossFunction;
            _network = network;
            _iterations = 0;
            BatchTrainer = new BatchTrainer(Params.BatchSize, set, Params.Randomize);
        }

        internal override void Reset()
        {
            _iterations = 0;
            BatchTrainer?.Reset();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsAndBiasesWithDeltaRule(ParametersUpdate result)
        {
            Debug.Assert(_network != null, nameof(_network) + " != null");
            Debug.Assert(result.Weights.Length == result.Biases.Length);

            for (int i = 0; i < result.Weights.Length; i++)
            {
                _network.Layers[i].Weights.Subtract(result.Weights[i] / Params.BatchSize, _network.Layers[i].Weights);
                _network.Layers[i].Biases.Subtract(result.Biases[i] / Params.BatchSize, _network.Layers[i].Biases);
            }
        }

        private ParametersUpdate CalculateDelta(Matrix<double> input, Matrix<double> expected)
        {
            Debug.Assert(_network != null && _lossFunction != null, "Setup was not called");

            var update = ParametersUpdate.FromNetwork(_network);
            _network.CalculateOutput(input);

            Matrix<double> delta1W1 = _lossFunction.Derivative(_network.Layers[^1].Output!, expected);
            for (var i = _network.Layers.Count - 1; i >= 0; --i)
            {
                var dA = _network.Layers[i].ActivationFunction.Derivative(_network.Layers[i].Net!);
                var delta = delta1W1.PointwiseMultiply(dA);
                var deltaLr = delta.Multiply(Params.LearningRate);

                delta1W1 = _network.Layers[i].Weights.TransposeThisAndMultiply(delta);

                update.Biases[i] = deltaLr;
                update.Weights[i] = deltaLr.TransposeAndMultiply(i > 0 ? _network.Layers[i-1].Output : input);

                if (_previousLearningMethodResult != null)
                {
                     _previousLearningMethodResult.Weights[i].Multiply(Params.Momentum, _previousLearningMethodResult.Weights[i]);
                    update.Weights[i].Add(_previousLearningMethodResult.Weights[i], update.Weights[i]);
                }
            }

            if (Params.Momentum > 0d)
            {
                _previousLearningMethodResult = update;
            }

            return update;
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