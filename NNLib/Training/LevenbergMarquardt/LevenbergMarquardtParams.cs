namespace NNLib.Training.LevenbergMarquardt
{
    public class LevenbergMarquardtParams
    { 
        public double Eps { get; set; }

        public double DampingParameterFactor { get; set; } = 1.01;

    }
}