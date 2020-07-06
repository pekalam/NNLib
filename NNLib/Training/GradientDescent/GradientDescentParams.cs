using System;

namespace NNLib
{
    public class GradientDescentParams : ICloneable
    {
        private double _learningRate = 0.001;
        private double _momentum = 0;

        public double LearningRate
        {
            get => _learningRate;
            set
            {
                if (!double.IsFinite(value))
                {
                    throw new InvalidOperationException();
                }
                if (value <= 0)
                {
                    throw new InvalidOperationException();
                }
                _learningRate = value;
            }
        }

        public double Momentum
        {
            get => _momentum;
            set
            {
                if (!double.IsFinite(value))
                {
                    throw new InvalidOperationException();
                }
                if (value < 0)
                {
                    throw new InvalidOperationException();
                }
                _momentum = value;
            }
        }

        public BatchParams BatchParams { get; set; } = new BatchParams();
        public object Clone()
        {
            return new GradientDescentParams()
            {
                Momentum = _momentum,BatchParams = new BatchParams() { BatchSize = BatchParams.BatchSize},LearningRate = _learningRate,
            };
        }

        public override bool Equals(object? obj)
        {
            if (obj == null) return false;

            if (obj is GradientDescentParams o)
            {
                return Momentum == o.Momentum && LearningRate == o.LearningRate &&
                       o.BatchParams.BatchSize == o.BatchParams.BatchSize;
            }

            return false;
        }
    }
}