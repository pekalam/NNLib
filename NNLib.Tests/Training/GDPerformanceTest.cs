using System;
using System.Diagnostics;
using System.Threading.Tasks;
using NNLib.Csv;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class GDPerformanceTest : TrainerTestBase
    {
        public GDPerformanceTest(ITestOutputHelper output) : base(output)
        {
        }

        [Theory]
        [InlineData("sin.csv")]
        public void GradientDescent_tests(string fileName)
        {
            int totalTests = 10; 
            long totalTime = 0;

            var net = CreateNetwork(1, (2, new TanHActivationFunction()), (100, new TanHActivationFunction()),
                (100, new TanHActivationFunction()), (1, new SigmoidActivationFunction()));

            var s = new Stopwatch();


            for (int j = 0; j < totalTests; j++)
            {

                var trainer = new MLPTrainer(net.Clone(), CsvFacade.LoadSets(fileName).sets,
                    new GradientDescentAlgorithm(new GradientDescentParams()), new QuadraticLossFunction());

                s.Restart();
                int i = 0;
                while (i++ < 100)
                    trainer.DoEpoch();
                s.Stop();
                totalTime += s.ElapsedMilliseconds;
                _output.WriteLine(s.ElapsedMilliseconds.ToString());
            }



            _output.WriteLine("Mean ms: " + totalTime / totalTests);
        }
    }

}