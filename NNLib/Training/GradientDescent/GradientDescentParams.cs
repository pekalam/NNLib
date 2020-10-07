﻿using System;

namespace NNLib
{
    public class GradientDescentParams : ICloneable
    {
        private double _learningRate = 0.001;
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

        public object Clone()
        {
            return new GradientDescentParams()
            {
                Momentum = _momentum, BatchSize = _batchSize,
                LearningRate = _learningRate,
            };
        }

        public override bool Equals(object? obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((GradientDescentParams) obj);
        }

        protected bool Equals(GradientDescentParams other)
        {
            return _learningRate.Equals(other._learningRate) && _momentum.Equals(other._momentum) && _batchSize == other._batchSize;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = _learningRate.GetHashCode();
                hashCode = (hashCode * 397) ^ _momentum.GetHashCode();
                hashCode = (hashCode * 397) ^ _batchSize;
                return hashCode;
            }
        }
    }
}