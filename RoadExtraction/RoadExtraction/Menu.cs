using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace RoadExtraction
{
  class Menu
  {
    /// <summary>
    /// Main menu
    /// </summary>
    /// <returns></returns>
    static public string MainMenu()
    {
      System.Console.WriteLine("Enter the digit for the wanted option");
      System.Console.WriteLine("  1: Train neural network");
      System.Console.WriteLine("  2: Run saved neural network");
      System.Console.WriteLine("  3: Run road extraction algorithm");
      System.Console.WriteLine("  4: Run neural network testing script");
      System.Console.WriteLine("  5: Data processing");
      System.Console.WriteLine("  6: Evaluate road map");
      System.Console.WriteLine("  7: Exit");
      System.Console.Write("> ");
      string menuSelection = System.Console.ReadLine();
      System.Console.WriteLine();

      return menuSelection;
    }


    /// <summary>
    /// Menu for selecting option 1 in main menu
    /// </summary>
    /// <returns></returns>
    static public string Sel1Menu()
    {
      System.Console.WriteLine("Enter the digit for the wanted option");
      System.Console.WriteLine("  1: Initialize random weights");
      System.Console.WriteLine("  2: Load saved weights");
      System.Console.Write("> ");
      string menuSelection = System.Console.ReadLine();
      System.Console.WriteLine();

      return menuSelection;
    }


    /// <summary>
    /// First menu for selecting option 2 in main menu
    /// </summary>
    /// <returns></returns>
    static public string Sel2Menu()
    {
      System.Console.WriteLine("Enter the digit for the wanted option");
      System.Console.WriteLine("  1: Get binary road map");
      System.Console.WriteLine("  2: Get raw road map");
      System.Console.Write("> ");
      string menuSelection = System.Console.ReadLine();
      System.Console.WriteLine();

      return menuSelection;
    }


    /// <summary>
    /// Second menu for selecting option 2 in main menu
    /// </summary>
    /// <returns></returns>
    static public string Sel2Menu2()
    {
      System.Console.WriteLine("Enter the digit for the wanted option");
      System.Console.WriteLine("  1: Run on single image");
      System.Console.WriteLine("  2: Run on a set of images");
      System.Console.Write("> ");
      string menuSelection = System.Console.ReadLine();
      System.Console.WriteLine();

      return menuSelection;
    }


    /// <summary>
    /// Menu for selecting option 5 in main menu
    /// </summary>
    /// <returns></returns>
    static public string Sel5Menu()
    {
      System.Console.WriteLine("Enter the digit for the wanted option");
      System.Console.WriteLine("  1: Preprocess Google map images");
      System.Console.WriteLine("  2: Crop images");
      System.Console.WriteLine("  3: Rename bitmaps");
      System.Console.Write("> ");
      string menuSelection = System.Console.ReadLine();
      System.Console.WriteLine();

      return menuSelection;
    }


    /// <summary>
    /// Second menu for selecting option 2 in main menu
    /// </summary>
    /// <returns></returns>
    static public string Sel5Menu2()
    {
      System.Console.WriteLine("Enter the digit for the wanted option");
      System.Console.WriteLine("  1: Crop a single image");
      System.Console.WriteLine("  2: Crop several images");
      System.Console.Write("> ");
      string menuSelection = System.Console.ReadLine();
      System.Console.WriteLine();

      return menuSelection;
    }

    static public string RunAnotherImage()
    {
      System.Console.WriteLine("Enter the digit for the wanted option");
      System.Console.WriteLine("  1: Select a new image");
      System.Console.WriteLine("  2: Go to main menu");
      System.Console.Write("> ");
      string menuSelection = System.Console.ReadLine();
      System.Console.WriteLine();

      return menuSelection;

    }

  }
}
