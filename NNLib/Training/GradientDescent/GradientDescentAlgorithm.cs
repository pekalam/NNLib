using MathNet.Numerics.LinearAlgebra;
using NNLib.Training;

namespace NNLib
{
    public class GradientDescentAlgorithm : AlgorithmBase
    {
        private LearningMethodResult? _previousLearningMethodResult;

        public GradientDescentAlgorithm(GradientDescentParams parameters)
        {
            Params = parameters;
        }

        public GradientDescentParams Params { get; set; }

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

        public override void Setup(SupervisedSet trainingData, MLPNetwork network)
        {
            _previousLearningMethodResult = null;
        }

        public override LearningMethodResult CalculateDelta(MLPNetwork network, Matrix<double> input, Matrix<double> expected, ILossFunction lossFunction)
        {
            var learningResult = LearningMethodResult.FromNetwork(network);

            Matrix<double> next = lossFunction.Derivative(network.Layers[^1].Output, expected);
            for (int i = network.Layers.Count - 1; i >= 0; --i)
            {
                var layer = network.Layers[i];

                var previousDelta = CalcUpdate(i, layer, learningResult, i > 0 ? network.Layers[i - 1].Output : input, next);

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

    }
}