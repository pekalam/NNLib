using NNLib;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using CsvHelper;
using MathNet.Numerics;
using NNLib.Common;
using NNLib.Csv;
using NNLib.Tests;

namespace TrainingTest
{
    public class Pt
    {
        public double X { get; set; }
        public double Y { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Control.UseNativeMKL();
            //Control.UseManaged();

            var set = CsvFacade.LoadSets("C:\\Users\\Marek\\Desktop\\sinus2.csv").sets.TrainingSet;
            var lossFunction = new QuadraticLossFunction();
            var net = new MLPNetwork(new PerceptronLayer(1, 1, new SigmoidActivationFunction(), new XavierMatrixBuilder()),
                new PerceptronLayer(1, 2, new SigmoidActivationFunction(), new XavierMatrixBuilder()),
                new PerceptronLayer(2, 1, new LinearActivationFunction(), new XavierMatrixBuilder()));
            var trainer = new MLPTrainer(net,
                new SupervisedTrainingSets(set),
                new GradientDescentAlgorithm(new GradientDescentParams()
                {
                    LearningRate = 0.01, Momentum = 0.09, BatchSize = 1,Randomize = false,
                }),
                lossFunction
            );


            int i = 0;
            var s = Stopwatch.StartNew();
            while (trainer.Error < 0 || trainer.Error > 0.01)
            {
                trainer.DoEpoch();
                i++;
                Console.WriteLine($"{trainer.Epochs} {trainer.Error}");
                if (i == 1000)
                {
                    i = 0;
                }

                if (Console.KeyAvailable)
                {
                    var a = Console.ReadKey();

                    if(a.Key == ConsoleKey.A)
                        break;
                }
            }

            s.Stop();

            Console.WriteLine(trainer.Error + " " + s.Elapsed);

            Console.WriteLine(net.CalculateOutput(-3.14));


            double start = -3.14;
            double end = 3.14;
            double inc = 0.1;



            List<Pt> records = new List<Pt>(100_000);

            while (start <= end)
            {
                net.CalculateOutput(start);

                records.Add(new Pt() { X = start, Y = net.Output![0,0] });

                start += inc;
            }

            using (var writer = new StreamWriter("C:\\Users\\Marek\\Desktop\\Moje.csv"))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(records);
            }




            Console.ReadKey();
        }
    }
}