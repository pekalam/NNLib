using System;

namespace NNLib.Training.LevenbergMarquardt
{
    public class LevenbergMarquardtParams : ICloneable
    {
        private double _dampingParamIncFactor = 10;
        private double _dampingParamDecFactor = 0.10;

        public double DampingParamIncFactor
        {
            get => _dampingParamIncFactor;
            set
            {
                if (value <= 0)
                {
                    throw new ArgumentException("Factor must be greater than zero");
                }
                _dampingParamIncFactor = value;
            }
        }

        public double DampingParamDecFactor
        {
            get => _dampingParamDecFactor;
            set
            {
                if (value <= 0)
                {
                    throw new ArgumentException("Factor must be greater than zero");
                }
                _dampingParamDecFactor = value;
            }
        }

        public object Clone()
        {
            return MemberwiseClone();
        }

        protected bool Equals(LevenbergMarquardtParams other)
        {
            return DampingParamIncFactor.Equals(other.DampingParamIncFactor) && DampingParamDecFactor.Equals(other.DampingParamDecFactor);
        }

        public override bool Equals(object? obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((LevenbergMarquardtParams) obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (DampingParamIncFactor.GetHashCode() * 397) ^ DampingParamDecFactor.GetHashCode();
            }
        }
    }
}