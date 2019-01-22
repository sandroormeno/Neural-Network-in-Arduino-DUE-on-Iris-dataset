
/******************************************************************
   ArduinoANN - An artificial neural network for the Arduino
   All basic settings can be controlled via the Network Configuration
   section.
   See robotics.hobbizine.com/arduinoann.html for details.
 ******************************************************************/

#include <math.h>


/******************************************************************
   Network Configuration - customized per network
 ******************************************************************/

const int PatternCount = 120;
const int InputNodes = 4;
const int HiddenNodes = 12;
const int OutputNodes = 3;
const float LearningRate = 0.03;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float Success = 2.0;

#include "iris.h"

/******************************************************************
   End Network Configuration
 ******************************************************************/


int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long  TrainingCycle;
float Rando;
float Error;
float Accum;

int acierto;


float Hidden[HiddenNodes];
float Output[OutputNodes];
float HiddenWeights[InputNodes + 1][HiddenNodes];
float OutputWeights[HiddenNodes + 1][OutputNodes];
float HiddenDelta[HiddenNodes];
float OutputDelta[OutputNodes];
float ChangeHiddenWeights[InputNodes + 1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes + 1][OutputNodes];

void setup() {
  Serial.begin(57600);
  randomSeed(analogRead(3));
  ReportEvery1000 = 1;
  for ( p = 0 ; p < PatternCount ; p++ ) {
    RandomizedIndex[p] = p ;
  }



}

void loop () {


  /******************************************************************
    Initialize OutputWeights and ChangeOutputWeights
  ******************************************************************/

  for ( i = 0 ; i < OutputNodes ; i ++ ) {
    for ( j = 0 ; j <= HiddenNodes ; j++ ) {
      ChangeOutputWeights[j][i] = 0.0 ;
      Rando = float(random(100)) / 100;
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
  Serial.println("Entrenamiento: ");
  //toTerminal();
  /******************************************************************
    Begin training
  ******************************************************************/

  for ( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++) {

    /******************************************************************
      Randomize order of training patterns
    ******************************************************************/

    for ( p = 0 ; p < PatternCount ; p++) {
      q = random(PatternCount);
      r = RandomizedIndex[p] ;
      RandomizedIndex[p] = RandomizedIndex[q] ;
      RandomizedIndex[q] = r ;
    }
    Error = 0.0 ;
    /******************************************************************
      Cycle through each training pattern in the randomized order
    ******************************************************************/
    for ( q = 0 ; q < PatternCount ; q++ ) {
      p = RandomizedIndex[q];

      /******************************************************************
        Compute hidden layer activations
      ******************************************************************/

      for ( i = 0 ; i < HiddenNodes ; i++ ) {
        Accum = HiddenWeights[InputNodes][i] ;
        for ( j = 0 ; j < InputNodes ; j++ ) {
          Accum += Input[p][j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;
      }

      /******************************************************************
        Compute output layer activations and calculate errors
      ******************************************************************/

      for ( i = 0 ; i < OutputNodes ; i++ ) {
        Accum = OutputWeights[HiddenNodes][i] ;
        for ( j = 0 ; j < HiddenNodes ; j++ ) {
          Accum += Hidden[j] * OutputWeights[j][i] ;
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum)) ;
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;
      }

      /******************************************************************
        Backpropagate errors to hidden layer
      ******************************************************************/

      for ( i = 0 ; i < HiddenNodes ; i++ ) {
        Accum = 0.0 ;
        for ( j = 0 ; j < OutputNodes ; j++ ) {
          Accum += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
      }


      /******************************************************************
        Update Inner-->Hidden Weights
      ******************************************************************/


      for ( i = 0 ; i < HiddenNodes ; i++ ) {
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for ( j = 0 ; j < InputNodes ; j++ ) {
          ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
      }

      /******************************************************************
        Update Hidden-->Output Weights
      ******************************************************************/

      for ( i = 0 ; i < OutputNodes ; i ++ ) {
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for ( j = 0 ; j < HiddenNodes ; j++ ) {
          ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
          OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
      }
    }

    /******************************************************************
      Every 1000 cycles send data to terminal for display
    ******************************************************************/
    ReportEvery1000 = ReportEvery1000 - 1;
    if (ReportEvery1000 == 0)
    {
      //Serial.println();
      //Serial.println();
      Serial.print ("Ciclo de entrenamiento: ");
      Serial.print (TrainingCycle);
      Serial.print ("  Error = ");
      Serial.print (Error, 5);





      toTerminal();

      if (TrainingCycle == 1)
      {
        ReportEvery1000 = 9;
      }
      else
      {
        ReportEvery1000 = 10;
      }
    }


    /******************************************************************
      If error rate is less than pre-determined threshold then end
    ******************************************************************/

    if ( Error < Success ) break ;
  }
  //Serial.println ();
  //Serial.println();
  Serial.print ("Ciclo de entrenamiento: ");
  Serial.print (TrainingCycle);
  Serial.print ("  Error = ");
  Serial.print (Error, 5);

  toTerminal();

  Serial.println ();
  Serial.println ();
  Serial.println ("Entrenamiento terminadao! ");
  Serial.println ("--------");
  Serial.println ();
  Serial.println ();


  InputToOutput(5.0, 3.3, 1.4, 0.2); // setosa
  InputToOutput(5.90, 3.0, 5.1, 1.8); // virginica
  InputToOutput(5.7, 2.8, 4.1, 1.3); // versicolor



  ReportEvery1000 = 1;


}

void toTerminal()
{
  acierto = 0;
  for ( p = 0 ; p < PatternCount ; p++ ) {

    /******************************************************************
      Compute hidden layer activations
    ******************************************************************/

    for ( i = 0 ; i < HiddenNodes ; i++ ) {
      Accum = HiddenWeights[InputNodes][i] ;
      for ( j = 0 ; j < InputNodes ; j++ ) {
        Accum += Input[p][j] * HiddenWeights[j][i] ;
      }
      Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;
    }

    /******************************************************************
      Compute output layer activations and calculate errors
    ******************************************************************/

    for ( i = 0 ; i < OutputNodes ; i++ ) {
      Accum = OutputWeights[HiddenNodes][i] ;
      for ( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i] ;
      }
      Output[i] = 1.0 / (1.0 + exp(-Accum)) ;
    }


    if ( Output[0] > Output[1] && Output[0] > Output[2] && Target[p][0] == 1 )
    {
      acierto++;
    }
    else if ( Output[1] > Output[0] && Output[1] > Output[2] && Target[p][1] == 1 )
    {
      acierto++;
    }
    else if ( Output[2] > Output[0] && Output[2] > Output[1] && Target[p][2] == 1 )
    {
      acierto++;
    }


  }
  Serial.print(" ");
  Serial.print ("  Acierto: ");
  float  porcentage = acierto * 100 / PatternCount ;
  Serial.print (porcentage, 1);
  Serial.println ("% ");



}

void InputToOutput(float In1, float In2, float In3, float In4)
{
  float TestInput[] = {0, 0, 0, 0};
  TestInput[0] = In1;
  TestInput[1] = In2;
  TestInput[2] = In3;
  TestInput[3] = In4;

  /******************************************************************
    Compute hidden layer activations
  ******************************************************************/

  for ( i = 0 ; i < HiddenNodes ; i++ ) {
    Accum = HiddenWeights[InputNodes][i] ;
    for ( j = 0 ; j < InputNodes ; j++ ) {
      Accum += TestInput[j] * HiddenWeights[j][i] ;
    }
    Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;
  }

  /******************************************************************
    Compute output layer activations and calculate errors
  ******************************************************************/

  for ( i = 0 ; i < OutputNodes ; i++ ) {
    Accum = OutputWeights[HiddenNodes][i] ;
    for ( j = 0 ; j < HiddenNodes ; j++ ) {
      Accum += Hidden[j] * OutputWeights[j][i] ;
    }
    Output[i] = 1.0 / (1.0 + exp(-Accum)) ;
  }

  Serial.print ("Predicción  para: ");
  for ( i = 0 ; i < InputNodes ; i++ ) {
    Serial.print (TestInput[i], 2);
    Serial.print (" ");
  }

  Serial.print ("  Solución: ");
  for ( i = 0 ; i < OutputNodes ; i++ ) {
    Serial.print (Output[i], 1);
    Serial.print (" ");
  }
  if ( Output[0] > Output[1] && Output[0] > Output[2])
  {
    Serial.print (" Iris setosa ");
  }
  else if ( Output[1] > Output[0] && Output[1] > Output[2])
  {
    Serial.print (" Iris versicolor ");
  }
  else if ( Output[2] > Output[0] && Output[2] > Output[1])
  {
    Serial.print (" Iris virginica ");
  }



  Serial.println ();
  Serial.println ();
  Serial.println ();
  Serial.println ("--------");
  Serial.println ();
  Serial.println ();
  delay(2000);


}



