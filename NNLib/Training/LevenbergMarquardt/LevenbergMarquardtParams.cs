namespace NNLib.Training.LevenbergMarquardt
{
    public class LevenbergMarquardtParams
    {
        /// <summary>
        /// Lambda parameter
        /// </summary>
        public double lambda { get; set; } = 10000;

        public double Eps { get; set; }

        public double LambdaStep { get; set; } = 0.1;

    }
}