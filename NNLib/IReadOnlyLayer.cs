namespace NNLib
{
    public interface IReadOnlyLayer
    {
        ReadMatrixWrapper ReadOutput { get; }
        ReadMatrixWrapper ReadWeights { get; }
        int NeuronsCount { get; }
        int InputsCount { get; }
        bool IsOutputLayer { get; }
        bool IsInputLayer { get; }
        bool HasBiases { get; }
    }
}