using System.Linq;
using System.Threading;
using NNLib.Data;
using NNLib.Exceptions;
using NNLib.LossFunction;
using NNLib.MLP;

namespace NNLib
{
    public interface IErrorCalculation
    {
        double CalculateError(ILossFunction lossFunction, MLPNetwork network, SupervisedTrainingSamples data, in CancellationToken ct);

        double CalculateError(ILossFunction lossFunction, MLPNetwork network, LoadedSupervisedTrainingData data,
            DataSetType setType, in CancellationToken c);
    }

    public class MSEError : IErrorCalculation
    {
        public double CalculateError(ILossFunction lossFunction, MLPNetwork network, SupervisedTrainingSamples data, in CancellationToken ct)
        {
            network.CalculateOutput(data.Input[0]);
            var m = lossFunction.Function(network.Output!, data.Target[0]);

            for (int i = 1; i < data.Input.Count; i++)
            {
                network.CalculateOutput(data.Input[i]);
             
                TrainingCanceledException.ThrowIfCancellationRequested(ct);

                m += lossFunction.Function(network.Output!, data.Target[i]);
            }

            return m.Enumerate().Sum() / data.Input.Count;
        }
        
        public double CalculateError(ILossFunction lossFunction, MLPNetwork network, LoadedSupervisedTrainingData data, DataSetType setType, in CancellationToken ct)
        {
            var (I, T) = data.GetSamples(setType);
            network.CalculateOutput(I);

            TrainingCanceledException.ThrowIfCancellationRequested(ct);

            return lossFunction.Function(network.Output!, T).Enumerate().Sum() / I.ColumnCount;
        }


    }
}
