using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
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