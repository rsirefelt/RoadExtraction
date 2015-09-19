using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MiscLibrary
{
    public static class Initialize
    {
      /// <summary>
      /// Allocates a double array of length (dim1)
      /// </summary>
      /// <param name="dim1"></param>
      /// <returns></returns>
      public static T[] Allocate<T>(int dim1)
      {
        T[] array = new T[dim1];
        return array;
      }


      /// <summary>
      /// Allocates a double matrix of dimensions (dim1,dim2) 
      /// </summary>
      /// <param name="dim1"></param>
      /// <param name="dim2"></param>
      /// <returns></returns>
      public static T[][] Allocate<T>(int dim1, int dim2)
      {
        T[][] matrix = new T[dim1][];
        for (int i = 0; i < dim1; ++i) {
          matrix[i] = new T[dim2];
        }

        return matrix;
      }


      /// <summary>
      /// Allocates a double 3D tensor of dimensions (dim1,dim2,dim3)
      /// </summary>
      /// <param name="dim1"></param>
      /// <param name="dim2"></param>
      /// <param name="dim3"></param>
      /// <returns></returns>
      public static T[][][] Allocate<T>(int dim1, int dim2, int dim3)
      {
        T[][][] tensor3D = new T[dim1][][];
        for (int i = 0; i < dim1; ++i) {
          tensor3D[i] = new T[dim2][];
          for (int j = 0; j < dim2; ++j) {
            tensor3D[i][j] = new T[dim3];
          }
        }

        return tensor3D;
      }


      /// <summary>
      /// Allocates a double 4D tensor of dimensions (dim1,dim2,dim3,dim4)
      /// </summary>
      /// <param name="dim1"></param>
      /// <param name="dim2"></param>
      /// <param name="dim3"></param>
      /// <param name="dim4"></param>
      /// <returns></returns>
      public static T[][][][] Allocate<T>(int dim1, int dim2, int dim3, int dim4)
      {
        T[][][][] tensor4D = new T[dim1][][][];
        for (int i = 0; i < dim1; ++i) {
          tensor4D[i] = new T[dim2][][];
          for (int j = 0; j < dim2; ++j) {
            tensor4D[i][j] = new T[dim3][];
            for (int k = 0; k < dim3; ++k) {
              tensor4D[i][j][k] = new T[dim4];
            }
          }
        }

        return tensor4D;
      }


      /// <summary>
      /// Set all elements of the array to zero
      /// </summary>
      /// <param name="array"></param>
      public static void SetToZero(double[] array)
      {
        for (int i = 0; i < array.Length; ++i) {
          array[i] = 0;
        }
      }


      /// <summary>
      /// Set all elements of the 4D tensor to zero (note the tensor needs to be a 'square')
      /// </summary>
      /// <param name="tensor"></param>
      public static void SetToZero(double[][][][] tensor)
      {
        for (int i = 0; i < tensor.GetLength(0); ++i) {
          for (int j = 0; j < tensor[0].GetLength(0); ++j) {
            for (int k = 0; k < tensor[0][0].GetLength(0); ++k) {
              for (int l = 0; l < tensor[0][0][0].GetLength(0); ++l) {
                tensor[i][j][k][l] = 0;
              }
            }
          }
        }

      }

    }
}
