using System;

namespace NNLib
{
    public class GradientDescentParams
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
    }
}