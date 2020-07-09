using System;
using NNLib;
using NNLib.Csv;

namespace LMTest
{
    class Program
    {
        static void Main(string[] args)
        {
            var net = new MLPNetwork(new []
            {
                new PerceptronLayer(1,10, new TanHActivationFunction()),
                new PerceptronLayer(10,10, new TanHActivationFunction()),
                new PerceptronLayer(10,1,new LinearActivationFunction()), 
            });

            var trainer = new MLPTrainer(net, CsvFacade.LoadSets("sin.csv").sets, new LevenbergMarquardtAlgorithm(new LevenbergMarquardtParams()), new QuadraticLossFunction());

            int i = 0;
            double err;

            do
            {
                err = trainer.DoEpoch();
                Console.WriteLine(err);
            } while (i < 10_000 && err > 0.01);

            Console.WriteLine(i + " " + err);
        }
    }
}
