using System.Collections.Generic;
using System.IO;

namespace NNLib.Csv
{
    internal static class FileHelpers
    {
        private const char CR = '\r';
        private const char LF = '\n';
        private const char NULL = '\0';

        public static List<long> CountLinesAndGetPositions(string fileName)
        {
            using var fs = File.OpenRead(fileName);

            var linePositions = new List<long>(3000);
            var buffer = new byte[1024 * 1024];

            int read;
            long pos = 0L;
            char previousChar = NULL, currentChar = NULL;
            while ((read = fs.Read(buffer, 0, buffer.Length)) > 0)
            {
                for (var i = 0; i < read; i++)
                {
                    currentChar = (char)buffer[i];

                    if (currentChar == LF || currentChar == CR)
                    {
                        if (currentChar == LF && previousChar == CR) continue;

                        previousChar = currentChar;
                        linePositions.Add(pos + i);
                    }
                }

                pos += read;
            }
            if (currentChar != LF && currentChar != CR && currentChar != NULL)
            {
                linePositions.Add(pos);
            }

            return linePositions;
        }
    }
}