using System;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class GradientDescentAlgorithm : AlgorithmBase
    {
        private LearningMethodResult? _previousLearningMethodResult;
        private MLPNetwork _network;
        private ILossFunction _lossFunction;
        private int _iterations;

        public GradientDescentAlgorithm(GradientDescentParams parameters)
        {
            Params = parameters;
        }

        public GradientDescentParams Params { get; set; }
        public override int Iterations => _iterations;

        public override void Setup(Common.SupervisedSet trainingData, MLPNetwork network,ILossFunction lossFunction)
        {
            _previousLearningMethodResult = null;
            _lossFunction = lossFunction;
            _network = network;
            BatchTrainer = new BatchTrainer(Params.BatchParams)
            {
                TrainingSet = trainingData,Parameters = Params.BatchParams,
            };
        }

        public override void ResetIterations()
        {
            _iterations = 0;
            BatchTrainer?.Reset();
        }


        private Matrix<double> CalcUpdate(int layerInd, PerceptronLayer layer, LearningMethodResult result,
            Matrix<double> input, Matrix<double> next)
        {
            var outputDerivative = layer.ActivationFunction.DerivativeY(layer.Output);
            var delta = next.PointwiseMultiply(outputDerivative);
            var biasesDelta = delta.Multiply(Params.LearningRate);
            var weightsDelta = delta.TransposeAndMultiply(input).Multiply(Params.LearningRate);

            result.Weigths[layerInd] = weightsDelta;
            result.Biases[layerInd] = biasesDelta;

            return delta;
        }


        private void UpdateWeightsAndBiasesWithDeltaRule(LearningMethodResult result)
        {
            if (result.Weigths.Length != result.Biases.Length)
            {
                throw new Exception();
            }
        
            for (int i = 0; i < result.Weigths.Length; i++)
            {
                _network.Layers[i].Weights.Subtract(result.Weigths[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Subtract(result.Biases[i], _network.Layers[i].Biases);
            }
        }

        private LearningMethodResult CalculateDelta(Matrix<double> input, Matrix<double> expected)
        {
            var learningResult = LearningMethodResult.FromNetwork(_network);

            Matrix<double> next = _lossFunction.Derivative(_network.Layers[^1].Output, expected);
            for (int i = _network.Layers.Count - 1; i >= 0; --i)
            {
                var layer = _network.Layers[i];

                var previousDelta = CalcUpdate(i, layer, learningResult, i > 0 ? _network.Layers[i - 1].Output : input, next);

                next = layer.Weights.TransposeThisAndMultiply(previousDelta);


                if (_previousLearningMethodResult != null)
                {
                    _previousLearningMethodResult.Weigths[i] = _previousLearningMethodResult.Weigths[i]
                        .Multiply(Params.Momentum);
                    learningResult.Weigths[i].Add(_previousLearningMethodResult.Weigths[i], learningResult.Weigths[i]);
                }
            }

            if (Params.Momentum > 0d)
            {
                _previousLearningMethodResult = learningResult;
            }

            return learningResult;
        }


        public override bool DoIteration(in CancellationToken ct = default)
        { 
            var result = BatchTrainer.DoIteration(CalculateDelta);
            _iterations = BatchTrainer.Iterations;
            if (result != null)
            {
                UpdateWeightsAndBiasesWithDeltaRule(result);
                return true;
            }

            return false;
        }
    }
}