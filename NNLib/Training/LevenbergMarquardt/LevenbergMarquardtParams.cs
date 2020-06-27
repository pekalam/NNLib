namespace NNLib
{
    public class LevenbergMarquardtParams
    { 
        public double Eps { get; set; }

        public double DampingParamIncFactor { get; set; } = 12;
        public double DampingParamDecFactor { get; set; } = 0.15;
    }
}