using System.Collections.Generic;
using System.IO;

namespace NNLib.Csv
{
    internal static class FileHelpers
    {
        private const char CR = '\r';
        private const char LF = '\n';
        private const char NULL = (char)0;

        //TODO async enumerable
        public static List<long> CountLinesAndGetPositions(string fileName)
        {
            using var fs = File.OpenRead(fileName);

            var linePositions = new List<long>();

            var byteBuffer = new byte[1024 * 1024];
            var detectedEOL = NULL;
            var currentChar = NULL;

            int bytesRead;
            long pos = 0L;
            while ((bytesRead = fs.Read(byteBuffer, 0, byteBuffer.Length)) > 0)
            {
                for (var i = 0; i < bytesRead; i++)
                {
                    currentChar = (char)byteBuffer[i];

                    if (detectedEOL != NULL)
                    {
                        if (currentChar == detectedEOL)
                        {
                            linePositions.Add(pos + i);
                        }
                    }
                    else if (currentChar == LF || currentChar == CR)
                    {
                        detectedEOL = currentChar;
                        linePositions.Add(pos + i);
                    }
                }

                pos += bytesRead;
            }

            if (currentChar != LF && currentChar != CR && currentChar != NULL)
            {
                linePositions.Add(pos - 1);
            }
            return linePositions;
        }
    }
}