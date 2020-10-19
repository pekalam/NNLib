using System;
using MathNet.Numerics;
using NNLib;
using NNLib.ActivationFunction;
using NNLib.Csv;
using NNLib.Data;
using NNLib.Exceptions;
using NNLib.LossFunction;
using NNLib.MLP;
using NNLib.Training.LevenbergMarquardt;

namespace LMTest
{


    class Program
    {
        static void Main(string[] args)
        {
            Control.UseNativeMKL();

            var net = new MLPNetwork(new []
            {
                new PerceptronLayer(1, 10, new TanHActivationFunction(), new XavierMatrixBuilder()),
                new PerceptronLayer(10, 1, new LinearActivationFunction(), new XavierMatrixBuilder())
            });

            var sets = CsvFacade.LoadSets("C:\\Users\\Marek\\Desktop\\f-zloz-sin.csv").sets;
            var trainer = new MLPTrainer(net,sets, new LevenbergMarquardtAlgorithm(new LevenbergMarquardtParams()
            {
            }), new QuadraticLossFunction());

            int i = 0;
            double err;


            try
            {
                do
                {
                    err = trainer.DoEpoch();
                    Console.WriteLine(err);
                } while (i < 10_000 && err > 0.01);

                Console.WriteLine(i + " " + err);
            }
            catch (TrainingCanceledException e)
            {
                Console.WriteLine(e);
            }

        }
    }
}
