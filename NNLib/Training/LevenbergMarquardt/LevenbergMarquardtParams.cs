namespace NNLib.Training.LevenbergMarquardt
{
    public class LevenbergMarquardtParams : BatchParams
    {
        /// <summary>
        /// Lambda parameter
        /// </summary>
        public double lambda { get; set; } = 100000;

        public double Eps { get; set; }
    }
}