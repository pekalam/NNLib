using System;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.MLP
{
    public class LoadedSupervisedTrainingData
    {
        private SupervisedTrainingData _data;

        public LoadedSupervisedTrainingData(SupervisedTrainingData data)
        {
            _data = data;
            (I_Train, T_Train) = (data.TrainingSet.ReadInputSamples(), data.TrainingSet.ReadTargetSamples());
            data.TrainingSet.Input.Modified += TrainingInputOnModified;
            data.TrainingSet.Target.Modified += TrainingTargetOnModified;
            if (data.ValidationSet != null)
            {
                (I_Val, T_Val) = (data.ValidationSet.ReadInputSamples(), data.ValidationSet.ReadTargetSamples());
                data.ValidationSet.Input.Modified += ValidationInputOnModified;
                data.ValidationSet.Target.Modified += ValidationTargetOnModified;
            }

            if (data.TestSet != null)
            {
                (I_Test, T_Test) = (data.TestSet.ReadInputSamples(), data.TestSet.ReadTargetSamples());
                data.TestSet.Input.Modified += TestInputOnModified;
                data.TestSet.Target.Modified += TestTargetOnModified;
            }

        }

        private void TrainingInputOnModified()
        {
            I_Train = _data.TrainingSet.ReadInputSamples();
        }

        private void TrainingTargetOnModified()
        {
            T_Train = _data.TrainingSet.ReadTargetSamples();
        }

        private void ValidationInputOnModified()
        {
            I_Val = _data.ValidationSet!.ReadInputSamples();
        }

        private void ValidationTargetOnModified()
        {
            T_Val = _data.ValidationSet!.ReadTargetSamples();
        }

        private void TestInputOnModified()
        {
            I_Test = _data.TestSet!.ReadInputSamples();
        }

        private void TestTargetOnModified()
        {
            T_Test = _data.TestSet!.ReadTargetSamples();
        }

        public Matrix<double> I_Train;
        public Matrix<double> T_Train;

        public Matrix<double>? I_Val;
        public Matrix<double>? T_Val;
        
        public Matrix<double>? I_Test;
        public Matrix<double>? T_Test;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public (Matrix<double> I, Matrix<double> T) GetSamples(DataSetType setType)
        {
            if (setType == DataSetType.Training)
            {
                return (I_Train, T_Train);
            }

            if(setType == DataSetType.Validation)

            {
                return (I_Val!, T_Val!);
            }
            if(setType == DataSetType.Test)

            {
                return (I_Test!, T_Test!);
            }

            throw new NotImplementedException();
        }
    }
}