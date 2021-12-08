#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;

class GenerateFeatures {
public:
    bool ReadFile(string fileName) {
        ifstream fin(fileName);
        if (!fin.is_open()) {
            cout << "Error reading from file!" << endl;
            return false;
        }
        
        double num;
        vector<double> temp;
        while (fin >> num) {
            temp.push_back(num);
            if (fin.peek() == '\r') {
                data.push_back(temp);
                temp.clear();
            }
        }

        fin.close();
        return true;
    }

    void NormalizeData() {
        for (int i = 1; i < data[0].size(); i++) {
            double mean = 0, dev = 0;
            for (int j = 0; j < data.size(); j++) {
                mean += data[j][i];
            }
            mean /= data.size();

            for (int j = 0; j < data.size(); j++) {
                dev += pow((data[j][i] - mean), 2);
            }
            dev = sqrt(dev/(data.size() - 1));

            for (int j = 0; j < data.size(); j++) {
                data[j][i] = (data[j][i] - mean)/dev;
            }
        }
    }

    void UpdateBest(const vector<int> &features, float acc) {
        if (acc > accuracy) {
            accuracy = acc;
            bestFeatures = features;
        }
    }

    void ShiftColumns() {
        for (int i = 0; i < data.size(); i++) {
            swap(data[i][0], data[i][data[i].size()-1]);
        }
    }

    void OutputCurr(const vector<int> &currFeatures, float acc) {
        cout << "[" << currFeatures[0];
        for (int i = 1; i < currFeatures.size(); i++) {
            cout << " " << currFeatures[i];
        }
        cout << "] => " << acc << endl;
    }

    float LeaveOneOutCrossValidation(const vector<int> &currFeatures) {
        float pass = 0;
        vector<vector<double> > dataCopy;
        for (int i = 0; i < data.size(); i++) {
            dataCopy = data;
            dataCopy.erase(dataCopy.begin() + i);

            float nnDist = 999999;
            int nnLoc = data.size();
            for (int j = 0; j < dataCopy.size(); j++) {
                float dist = 0;
                for (int k = 0; k < currFeatures.size(); k++) {
                    dist += (pow(dataCopy[j][currFeatures[k]] - data[i][currFeatures[k]], 2));
                }
                dist = sqrt(dist);
                if (dist < nnDist) {
                    nnDist = dist;
                    nnLoc = j;
                }
            }
            int label = dataCopy[nnLoc][0];
            if (label == data[i][0]) {
                pass++;
            }
        }
        return pass / data.size();
    }

    void ForwardSelection() {
        vector<int> currFeatures, bestFeaturesI;
        float bestAcc, currAcc;
        // need to acct for empty set
        for (int i = 1; i < data[0].size(); i++) {
            bestAcc = 0;
            currFeatures.resize(i);
            for (int j = 1; j < data[0].size(); j++) {
                if (find(bestFeaturesI.begin(), bestFeaturesI.end(), j) == bestFeaturesI.end()) {
                    currFeatures[i-1] = j;
                    currAcc = LeaveOneOutCrossValidation(currFeatures);
                    OutputCurr(currFeatures, currAcc);
                    UpdateBest(currFeatures, currAcc);
                    if (currAcc > bestAcc) {
                        bestAcc = currAcc;
                        bestFeaturesI = currFeatures;
                    }
                }
            }
            currFeatures = bestFeaturesI;
        }

        cout << endl << "Best Features: " << endl;
        OutputCurr(bestFeatures, accuracy);
    }

    void BackwardElimination() {
        vector<int> currFeatures, bestFeaturesI, allFeatures;
        float currAcc, bestAcc;
        for (int i = 1; i < data[0].size(); i++) {
            allFeatures.push_back(i);
        }
        currFeatures = allFeatures;
        currFeatures.pop_back();

        while (!currFeatures.empty()) {
            int n = allFeatures.size();
            bestAcc = 0;
            currFeatures[0] = allFeatures[n-1];
            currAcc = LeaveOneOutCrossValidation(currFeatures);
            OutputCurr(currFeatures, currAcc);
            UpdateBest(currFeatures, currAcc);
            if (currAcc > bestAcc) {
                bestAcc = currAcc;
                bestFeaturesI = currFeatures;
            }
            for (int j = n - 1; j > 1; j--) {
                currFeatures[n-j] = allFeatures[n-j-1];
                currAcc = LeaveOneOutCrossValidation(currFeatures);
                OutputCurr(currFeatures, currAcc);
                UpdateBest(currFeatures, currAcc);
                if (currAcc > bestAcc) {
                    bestAcc = currAcc;
                    bestFeaturesI = currFeatures;
                }
            }
            currFeatures = bestFeaturesI;
            allFeatures = currFeatures;
            currFeatures.pop_back();
        }
        cout << endl << "Best Features: " << endl;
        OutputCurr(bestFeatures, accuracy);
    }

private:
    vector<vector<double> > data;
    vector<int> bestFeatures;
    float accuracy;
};

int main() {
    GenerateFeatures *g = new GenerateFeatures;
    string fileName = "large81.txt";
    cout << "Enter name of file: ";
    getline(cin, fileName);
    if (!g->ReadFile(fileName)) {
        return 0;
    }
    g->NormalizeData();
    while (1) {
        cout << "1) Foward Selection" << endl;
        cout << "2) Backward Elimination" << endl;
        cout << "3) Exit" << endl;
        int choice;
        cin >> choice;
        if (choice == 1) {
            g->ForwardSelection();
        } else if (choice == 2) {
            g->BackwardElimination();
        } else {
            break;
        }
    }

    return 0;
}