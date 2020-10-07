using System;

namespace NNLib
{
    public class LevenbergMarquardtParams : ICloneable
    { 

        public double DampingParamIncFactor { get; set; } = 11;
        public double DampingParamDecFactor { get; set; } = 0.15;
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