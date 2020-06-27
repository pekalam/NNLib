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
    }
}