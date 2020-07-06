using System;

namespace NNLib
{
    public class LevenbergMarquardtParams : ICloneable
    { 

        public double DampingParamIncFactor { get; set; } = 12;
        public double DampingParamDecFactor { get; set; } = 0.15;
        public object Clone()
        {
            return MemberwiseClone();
        }

        public override bool Equals(object? obj)
        {
            if (obj == null) return false;

            if (obj is LevenbergMarquardtParams o)
            {
                return DampingParamIncFactor == o.DampingParamIncFactor &&
                       DampingParamDecFactor == o.DampingParamDecFactor;
            }

            return false;
        }
    }
}