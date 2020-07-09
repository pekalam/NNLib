using System;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public abstract class RandomGenerator
    {
        public abstract Matrix<double> GenerateMat(int r, int c);
        public abstract Vector<double> GenerateColVec(int n);
        public abstract Vector<double> GenerateRowVec(int n);
    }

    public class NormalRandomGenerator : RandomGenerator
    {
        public override Matrix<double> GenerateMat(int r, int c)
        {
            return Matrix<double>.Build.Random(r, c, new Normal());
        }

        public override Vector<double> GenerateColVec(int n)
        {
            return Matrix<double>.Build.Random(n, 1, new Normal()).Column(0);
        }

        public override Vector<double> GenerateRowVec(int n)
        {
            return Matrix<double>.Build.Random(1, n, new Normal()).Row(0);
        }
    }

    public class PerceptronLayer : Layer
    {
        private IActivationFunction _activationFunction;


        public PerceptronLayer(int inputsCount, int neuronsCount, IActivationFunction activationFunction, RandomGenerator randomGenerator = null)
            : base(BuildWeightsMatrix(inputsCount, neuronsCount, randomGenerator ?? new NormalRandomGenerator()),
                BuildBiasesMatrix(inputsCount, neuronsCount, randomGenerator ?? new NormalRandomGenerator()), BuildOutputMatrix(inputsCount, neuronsCount,
                    randomGenerator ?? new NormalRandomGenerator()))
        {
            Guards._GtZero(inputsCount).GtZero(neuronsCount).NotNull(activationFunction);

            RandomGenerator = randomGenerator ?? new NormalRandomGenerator();
            ActivationFunction = activationFunction;
        }

        private PerceptronLayer(Matrix<double> weights, Matrix<double> biases, Matrix<double> output,
            IActivationFunction activationFunction) : base(weights, biases, output)
        {
            ActivationFunction = activationFunction;
        }

        public IActivationFunction ActivationFunction
        {
            get => _activationFunction;
            set
            {
                Guards._NotNull(value);
                _activationFunction = value;
            }
        }

        private static Matrix<double> BuildWeightsMatrix(int inputsCount, int neuronsCount, RandomGenerator randomGenerator) =>
            randomGenerator.GenerateMat(neuronsCount, inputsCount);

        private static Matrix<double> BuildBiasesMatrix(int inputsCount, int neuronsCount, RandomGenerator randomGenerator) =>
            randomGenerator.GenerateMat(neuronsCount, 1);

        private static Matrix<double> BuildOutputMatrix(int inputsCount, int neuronsCount, RandomGenerator randomGenerator) =>
            randomGenerator.GenerateMat(neuronsCount, 1);


        internal PerceptronLayer Clone() =>
            new PerceptronLayer(Weights.Clone(), Biases.Clone(), Output.Clone(), ActivationFunction);

        protected override void BuildMatrices(int inputsCount, int neuronsCount)
        {
            if (Weights == null && Biases == null)
            {
                Weights = BuildWeightsMatrix(inputsCount, neuronsCount, RandomGenerator);
                Biases = BuildBiasesMatrix(inputsCount, neuronsCount, RandomGenerator);
                //Output = BuildOutputMatrix(inputsCount, neuronsCount, RandomGenerator);
            }
            else
            {
                if (inputsCount > InputsCount)
                {
                    while (inputsCount != InputsCount)
                    {
                        Weights = Weights.InsertColumn(Weights.ColumnCount, RandomGenerator.GenerateColVec(NeuronsCount));
                    }
                }
                else if (inputsCount < InputsCount)
                {
                    while (inputsCount != InputsCount)
                    {
                        Weights = Weights.RemoveColumn(Weights.ColumnCount - 1);
                    }
                }

                if (neuronsCount > NeuronsCount)
                {
                    while (neuronsCount != NeuronsCount)
                    {
                        Weights = Weights.InsertRow(Weights.RowCount, RandomGenerator.GenerateRowVec(InputsCount));
                        Biases = Biases.InsertRow(Biases.RowCount, RandomGenerator.GenerateColVec(1));
                        //Output = Output.InsertRow(Output.RowCount, RandomGenerator.GenerateColVec(1));
                    }
                }
                else if (neuronsCount < NeuronsCount)
                {
                    while (neuronsCount != NeuronsCount)
                    {
                        Weights = Weights.RemoveRow(Weights.RowCount - 1);
                        Biases = Biases.RemoveRow(Biases.RowCount - 1);
                        //Output = Output.RemoveRow(Output.RowCount - 1);
                    }
                }
            }
        }

        public override void CalculateOutput(Matrix<double> input)
        {
            if (input.RowCount != InputsCount)
            {
                throw new ArgumentException();
            }

            Output = Weights.Multiply(input);

            if (input.ColumnCount == 1)
            {
                Output.Add(Biases, Output);
            }
            else
            {
                Output.Add(Biases * Matrix<double>.Build.Dense(1, Output.ColumnCount, 1), Output);
            }

            Output = ActivationFunction.Function(Output);
        }
    }
}