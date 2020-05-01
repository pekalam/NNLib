﻿using System;

namespace NNLib
{
    public class GradientDescentLearningParameters
    {
        private double _learningRate = 0.1;
        private double _momentum = 0;
        private int _batchSize = 1;

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

        public int BatchSize
        {
            get => _batchSize;
            set
            {
                if (value <= 0)
                {
                    throw new InvalidOperationException();
                }
                _batchSize = value;
            }
        }
    }
}