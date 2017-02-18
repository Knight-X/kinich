#pragma once
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <string>
#include "nn_datatype.hpp"

using namespace std;
int nInputs = 0;
int nTargets = 0;

void p(string& line, nn::nn_vec_t* data, nn::nn_vec_t* target);

bool loadFile(const char* filename, int nI, int nT, nn::nn_vec_t* data, nn::nn_vec_t* target)
{
  nInputs = nI;
  nTargets = nT;

  fstream inputFile;
  inputFile.open(filename, ios::in);

  if (inputFile.is_open()) {
    string line = "";

    while (!inputFile.eof()) 
    {
      getline(inputFile, line);
      if (line.length() > 2) p(line, data, target);
    }


    //random_shuffle(data->begin(), data->end());

    cout << "Input file: " << filename << "\nRead Complete: " << data->size() << "Patterns Loaded" << endl;
    nn::nn_vec_t& g = *data;
    for (int i = 0; i < 16; i++) {
      cout << g[i] << " ";
    }
    cout << endl;

    inputFile.close();

    return true;
  } else {
      cout << "Error Opening File: " << filename << endl;
      return false;
  }
}

void p(string &line, nn::nn_vec_t* data, nn::nn_vec_t* targets)
{
  float_t pattern;
  float_t target;

  char* cstr = new char[line.size() + 1];
  char* t;
  strcpy(cstr, line.c_str());

  int i = 0;
  char* nextToken = NULL;
  t = strtok_r(cstr, ",", &nextToken);

  while (t != NULL && i < (nInputs + nTargets)) 
  {
    if (i < nInputs) {
      pattern = atof(t);
      data->push_back(pattern);
    }
    else  {
      target = atof(t);
      targets->push_back(target);
    }

    t = strtok_r(NULL, ",", &nextToken);
    i++;
  }
  return;
}
/*
int main()
{
  nn::nn_vec_t data;
  nn::nn_vec_t target;
  loadFile("letter-recognition-2.csv", 16, 3, &data, &target); 

  return 0;
}*/
