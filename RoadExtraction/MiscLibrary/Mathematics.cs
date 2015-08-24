using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MiscLibrary
{
  public static class Mathematics
  {
    /// <summary>
    /// Return the mean value of a 3d tensor
    /// </summary>
    /// <param name="inputPatch"></param>
    /// <returns></returns>
    public static double CalculateMean(double[][][] tensor)
    {
      double meanValue = 0;
      int dim1 = tensor.Length;
      int dim2 = tensor[0].Length;
      int dim3 = tensor[0][0].Length;

      int totalNrOfDataPoints = dim1 * dim2 * dim3;

      for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
          for (int k = 0; k < dim3; ++k) {
            meanValue += tensor[i][j][k];
          }
        }
      }
      meanValue /= totalNrOfDataPoints;

      return meanValue;
    }


    /// <summary>
    /// Subtracts every value of the 3d tensor with the value 
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="value"></param>
    public static void SubtractByValue(double[][][] tensor, double value)
    {
      int dim1 = tensor.Length;
      int dim2 = tensor[0].Length;
      int dim3 = tensor[0][0].Length;

      for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
          for (int k = 0; k < dim3; ++k) {
            tensor[i][j][k] -= value;
          }
        }
      }
    }


    /// <summary>
    /// Divides every value of the 3d tensor with the value 
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="standardDeviation"></param>
    public static void DivideByValue(double[][][] tensor, double value)
    {
      int dim1 = tensor.Length;
      int dim2 = tensor[0].Length;
      int dim3 = tensor[0][0].Length;

      for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
          for (int k = 0; k < dim3; ++k) {
            tensor[i][j][k] /= value;
          }
        }
      }
    }


    /// <summary>
    /// Multiply every value of the 3d tensor with the value 
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="value"></param>
    public static void MultiplyByValue(byte[][][] tensor, byte value)
    {
      int dim1 = tensor.Length;
      int dim2 = tensor[0].Length;
      int dim3 = tensor[0][0].Length;

      for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
          for (int k = 0; k < dim3; ++k) {
            tensor[i][j][k] *= value;
          }
        }
      }

    }


    /// <summary>
    /// Rotates a 3d tensor determined by rotation type
    /// 1 -> 90 degrees clockwise
    /// 2 -> 90 degrees counterclockwise
    /// 3 -> 180 degrees
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="rotationType"></param>
    public static void Rotate3dTensor(double[][][] tensor, byte rotationType)
    {
      int sideSize = tensor.Length;
      
      switch(rotationType) {
        case 1: // rotate 90 degrees clockwise
          #region
          for (int layer = 0; layer < sideSize/2; ++layer) {
            int first = layer; 
            int last = sideSize - 1 - layer; 
            for (int i = first; i < last; ++i) {
              int offset = i - first; 
              //save top 
              double top0 = tensor[first][i][0];
              double top1 = tensor[first][i][1];
              double top2 = tensor[first][i][2];

              // left -> top
              tensor[first][i][0] = tensor[last-offset][first][0];
              tensor[first][i][1] = tensor[last-offset][first][1];
              tensor[first][i][2] = tensor[last-offset][first][2];
            
              // bottom -> left
              tensor[last-offset][first][0] = tensor[last][last-offset][0];
              tensor[last-offset][first][1] = tensor[last][last-offset][1];
              tensor[last-offset][first][2] = tensor[last][last-offset][2];
            
              // right -> bottom
              tensor[last][last-offset][0] = tensor[i][last][0];
              tensor[last][last-offset][1] = tensor[i][last][1];
              tensor[last][last-offset][2] = tensor[i][last][2];

              // top -> right
              tensor[i][last][0] = top0;
              tensor[i][last][1] = top1;
              tensor[i][last][2] = top2;
            }
          }
          break;
          #endregion
        case 2: // rotate 90 degrees counter clockwise
          #region
          for (int layer = 0; layer < sideSize/2; ++layer) {
            int first = layer; 
            int last = sideSize - 1 - layer; 
            for (int i = first; i < last; ++i) {
              int offset = i - first; 
              //save top 
              double top0 = tensor[first][i][0];
              double top1 = tensor[first][i][1];
              double top2 = tensor[first][i][2];

              // right -> top
              tensor[first][i][0] = tensor[i][last][0];
              tensor[first][i][1] = tensor[i][last][1];
              tensor[first][i][2] = tensor[i][last][2];
            
              // bottom -> right
              tensor[i][last][0] = tensor[last][last-offset][0];
              tensor[i][last][1] = tensor[last][last-offset][1];
              tensor[i][last][2] = tensor[last][last-offset][2];
            
              // left -> bottom
              tensor[last][last - offset][0] = tensor[last - offset][first][0];
              tensor[last][last - offset][1] = tensor[last - offset][first][1];
              tensor[last][last - offset][2] = tensor[last - offset][first][2];

              // top -> left
              tensor[last-offset][first][0] = top0;
              tensor[last-offset][first][1] = top1;
              tensor[last-offset][first][2] = top2;
            }
          }
          break;
          #endregion
        case 3: // Rotate 180 degrees
          #region
          for (int layer = 0; layer < sideSize / 2; ++layer) {
            int first = layer;
            int last = sideSize - 1 - layer;
            for (int i = first; i < last; ++i) {
              int offset = i - first;
              //save top 
              double top0 = tensor[first][i][0];
              double top1 = tensor[first][i][1];
              double top2 = tensor[first][i][2];

              //save right
              double right0 = tensor[i][last][0];
              double right1 = tensor[i][last][1];
              double right2 = tensor[i][last][2];

              // bottom -> top
              tensor[first][i][0] = tensor[last][last - offset][0];
              tensor[first][i][1] = tensor[last][last - offset][1];
              tensor[first][i][2] = tensor[last][last - offset][2];

              // top -> bottom
              tensor[last][last - offset][0] = top0;
              tensor[last][last - offset][1] = top1;
              tensor[last][last - offset][2] = top2;

              // left -> right
              tensor[i][last][0] = tensor[last - offset][first][0];
              tensor[i][last][1] = tensor[last - offset][first][1];
              tensor[i][last][2] = tensor[last - offset][first][2];

              // right -> left
              tensor[last - offset][first][0] = right0;
              tensor[last - offset][first][1] = right1;
              tensor[last - offset][first][2] = right2;
            }
          }
          break;
          #endregion
      }
    }


    /// <summary>
    /// Rotates a 3d tensor determined by rotation type
    /// 1 -> 90 degrees clockwise
    /// 2 -> 90 degrees counterclockwise
    /// 3 -> 180 degrees
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="rotationType"></param>
    public static void Rotate3dTensor(byte[][][] tensor, byte rotationType)
    {
      int sideSize = tensor.Length;

      switch (rotationType) {
        case 1: // rotate 90 degrees clockwise
          #region
          for (int layer = 0; layer < sideSize / 2; ++layer) {
            int first = layer;
            int last = sideSize - 1 - layer;
            for (int i = first; i < last; ++i) {
              int offset = i - first;
              //save top 
              byte top0 = tensor[first][i][0];
              byte top1 = tensor[first][i][1];
              byte top2 = tensor[first][i][2];

              // left -> top
              tensor[first][i][0] = tensor[last - offset][first][0];
              tensor[first][i][1] = tensor[last - offset][first][1];
              tensor[first][i][2] = tensor[last - offset][first][2];

              // bottom -> left
              tensor[last - offset][first][0] = tensor[last][last - offset][0];
              tensor[last - offset][first][1] = tensor[last][last - offset][1];
              tensor[last - offset][first][2] = tensor[last][last - offset][2];

              // right -> bottom
              tensor[last][last - offset][0] = tensor[i][last][0];
              tensor[last][last - offset][1] = tensor[i][last][1];
              tensor[last][last - offset][2] = tensor[i][last][2];

              // top -> right
              tensor[i][last][0] = top0;
              tensor[i][last][1] = top1;
              tensor[i][last][2] = top2;
            }
          }
          break;
          #endregion
        case 2: // rotate 90 degrees counter clockwise
          #region
          for (int layer = 0; layer < sideSize / 2; ++layer) {
            int first = layer;
            int last = sideSize - 1 - layer;
            for (int i = first; i < last; ++i) {
              int offset = i - first;
              //save top 
              byte top0 = tensor[first][i][0];
              byte top1 = tensor[first][i][1];
              byte top2 = tensor[first][i][2];

              // right -> top
              tensor[first][i][0] = tensor[i][last][0];
              tensor[first][i][1] = tensor[i][last][1];
              tensor[first][i][2] = tensor[i][last][2];

              // bottom -> right
              tensor[i][last][0] = tensor[last][last - offset][0];
              tensor[i][last][1] = tensor[last][last - offset][1];
              tensor[i][last][2] = tensor[last][last - offset][2];

              // left -> bottom
              tensor[last][last - offset][0] = tensor[last - offset][first][0];
              tensor[last][last - offset][1] = tensor[last - offset][first][1];
              tensor[last][last - offset][2] = tensor[last - offset][first][2];

              // top -> left
              tensor[last - offset][first][0] = top0;
              tensor[last - offset][first][1] = top1;
              tensor[last - offset][first][2] = top2;
            }
          }
          break;
          #endregion
        case 3: // Rotate 180 degrees
          #region
          for (int layer = 0; layer < sideSize / 2; ++layer) {
            int first = layer;
            int last = sideSize - 1 - layer;
            for (int i = first; i < last; ++i) {
              int offset = i - first;
              //save top 
              byte top0 = tensor[first][i][0];
              byte top1 = tensor[first][i][1];
              byte top2 = tensor[first][i][2];

              //save right
              byte right0 = tensor[i][last][0];
              byte right1 = tensor[i][last][1];
              byte right2 = tensor[i][last][2];

              // bottom -> top
              tensor[first][i][0] = tensor[last][last - offset][0];
              tensor[first][i][1] = tensor[last][last - offset][1];
              tensor[first][i][2] = tensor[last][last - offset][2];

              // top -> bottom
              tensor[last][last - offset][0] = top0;
              tensor[last][last - offset][1] = top1;
              tensor[last][last - offset][2] = top2;

              // left -> right
              tensor[i][last][0] = tensor[last - offset][first][0];
              tensor[i][last][1] = tensor[last - offset][first][1];
              tensor[i][last][2] = tensor[last - offset][first][2];

              // right -> left
              tensor[last - offset][first][0] = right0;
              tensor[last - offset][first][1] = right1;
              tensor[last - offset][first][2] = right2;
            }
          }
          break;
          #endregion
      }
    }


    /// <summary>
    /// Rotate a 1d array as it would be a 2d matrix in row major order
    /// determined by rotation type
    /// 1 -> 90 degrees clockwise
    /// 2 -> 90 degrees counterclockwise
    /// 3 -> 180 degrees
    /// </summary>
    /// <param name="array"></param>
    /// <param name="rotationType"></param>
    public static void RotateArray(byte[] array, byte rotationType)
    {
      int sideSize = (int)Math.Sqrt(array.Length);

      switch (rotationType) {
        case 1: // rotate 90 degrees clockwise
          #region
          for (int layer = 0; layer < sideSize / 2; ++layer) {
            int first = layer;
            int last = sideSize - 1 - layer;
            for (int i = first; i < last; ++i) {
              int offset = i - first;
              //save top 
              byte top0 = array[first * sideSize + i];

              // left -> top
              array[first * sideSize + i] = array[(last - offset) * sideSize + first];

              // bottom -> left
              array[(last - offset) * sideSize + first] = array[last * sideSize + (last - offset)];

              // right -> bottom
              array[last * sideSize + (last - offset)] = array[i * sideSize + last];

              // top -> right
              array[i * sideSize + last] = top0;
            }
          }
          break;
          #endregion
        case 2: // rotate 90 degrees counter clockwise
          #region
          for (int layer = 0; layer < sideSize / 2; ++layer) {
            int first = layer;
            int last = sideSize - 1 - layer;
            for (int i = first; i < last; ++i) {
              int offset = i - first;
              //save top 
              byte top0 = array[first * sideSize + i];

              // right -> top
              array[first * sideSize + i] = array[i * sideSize + last];

              // bottom -> right
              array[i * sideSize + last] = array[last * sideSize + (last - offset)];

              // left -> bottom
              array[last * sideSize + (last - offset)] = array[(last - offset) * sideSize + first];

              // top -> left
              array[(last - offset) * sideSize + first] = top0;
            }
          }
          break;
          #endregion
        case 3: // Rotate 180 degrees
          #region
          for (int layer = 0; layer < sideSize / 2; ++layer) {
            int first = layer;
            int last = sideSize - 1 - layer;
            for (int i = first; i < last; ++i) {
              int offset = i - first;
              //save top 
              byte top0 = array[first * sideSize + i];

              //save right
              byte right0 = array[i * sideSize + last];

              // bottom -> top
              array[first * sideSize + i] = array[last * sideSize + (last - offset)];

              // top -> bottom
              array[last * sideSize + (last - offset)] = top0;

              // left -> right
              array[i * sideSize + last] = array[(last - offset) * sideSize + first];

              // right -> left
              array[(last - offset) * sideSize + first] = right0;
            }
          }
          break;
          #endregion
      }
    }

  }
}
