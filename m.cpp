#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <chrono>

using namespace std::chrono;
using namespace std;

/* GenerateFeatures
        Class containing our data and functions to manipulate our data.
*/ 
class GenerateFeatures {
public:
    GenerateFeatures() {
        accuracy = 0;
    }

/* ReadFile() 
        Reads the input data.
*/
    bool ReadFile(string fileName) {
        data.clear();
        ifstream fin(fileName);
        if (!fin.is_open()) {
            cout << "Error reading from file!" << endl;
            return false;
        }
        
        double num;
        vector<double> temp;
        char endFileChar = (fileName == "seeds_dataset.txt" ? '\n' : '\r');
        while (fin >> num) {
            temp.push_back(num);
            if (fin.peek() == endFileChar) {
                data.push_back(temp);
                temp.clear();
            }
        }
        cout << "Rows: " << data.size() << endl;

        fin.close();
        if (fileName == "seeds_dataset.txt") {
            ShiftColumns();
        }
        return true;
    }

/*  NormalizeData()
        Normalizes the columns of data by subtracting the mean and dividing by the 
        standard deviation.
*/
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

/* UpdateBest()
        Checks the current accuracy against our best saved accuracy
        and updates the corresponding set if needed.
*/
    void UpdateBest(const vector<int> &features, float acc) {
        if (acc > accuracy) {
            accuracy = acc;
            bestFeatures = features;
        }
    }

/* ShiftColumns()
        Switches columns 0 and 6 for the seeds dataset.
*/
    void ShiftColumns() {
        for (int i = 0; i < data.size(); i++) {
            swap(data[i][0], data[i][data[i].size()-1]);
        }
    }

/* OutputCurr()
        Outputs the current features being checked and corresponding accuracy.
*/
    void OutputCurr(const vector<int> &currFeatures, float acc) {
        if (!currFeatures.empty()) {
            cout << "[" << currFeatures[0];
            for (int i = 1; i < currFeatures.size(); i++) {
                cout << " " << currFeatures[i];
            }
            cout << "] => " << acc << endl;
        } else {
            cout << "Empty set: " << acc << endl;
        }
    }

/* LeaveOneOutCrossValidation()
        Calculates the accuracy of our current features being checked. The distance
        is calculated as sqrt((featureX1 - featureX2)^2), we check if the class we calculated
        is correct and if so we add it to the total correct class count and divide it by our
        total number of instances.
*/
    float LeaveOneOutCrossValidation(const vector<int> &currFeatures) {
        float pass = 0;
        vector<vector<double> > dataCopy;

        for (int i = 0; i < data.size(); i++) {
            dataCopy = data;
            dataCopy.erase(dataCopy.begin() + i);

            float nnDist = 2147483647;
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

/* ForwardSelection()
        Forward selection search: We start from the empty set and add features which
        give us the best accuracy to the current set. The accuracy is calculated from
        LeaveOneOutCrossValidation().
*/
    void ForwardSelection() {
        vector<int> currFeatures, bestFeaturesI;
        float bestAcc, currAcc;
        // Base Case - No features
        currAcc = LeaveOneOutCrossValidation(currFeatures);
        UpdateBest(currFeatures, currAcc);
        OutputCurr(currFeatures, currAcc);
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

/* BackwardElimination()
        We start with all features and for each set, we remove a feature and calculate the
        accuracy of that set. We move forward with the set which removes the feature that
        gives the highest accuracy and continue until we reach the empty set.
*/
    void BackwardElimination() {
        vector<int> currFeatures, bestFeaturesI, allFeatures;
        float currAcc, bestAcc;
        
        for (int i = 1; i < data[0].size(); i++) {
            allFeatures.push_back(i);
        }
        // Base case - all features
        currFeatures = allFeatures;
        bestAcc = LeaveOneOutCrossValidation(currFeatures);
        UpdateBest(currFeatures, bestAcc);
        OutputCurr(currFeatures, bestAcc);
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
        // No features
        currAcc = LeaveOneOutCrossValidation(currFeatures);
        UpdateBest(currFeatures, currAcc);
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
    string fileName;
    cout << "Enter name of file: ";
    getline(cin, fileName);
    if (!g->ReadFile(fileName)) {
        return 0;
    }
    g->NormalizeData();    
    
    while (1) {
        cout << "1) Foward Selection" << endl;
        cout << "2) Backward Elimination" << endl;
        cout << "Else) Exit" << endl;
        int choice;
        cin >> choice;
        auto start = high_resolution_clock::now();
        if (choice == 1) {
            g->ForwardSelection();
        } else if (choice == 2) {
            g->BackwardElimination();
        } else {
            break;
        }
        auto stop = high_resolution_clock::now();
        double duration = duration_cast<nanoseconds>(stop - start).count();
        duration *= 1e-9;
        cout << fixed << "Time taken: " << duration << setprecision(9) << " sec" << endl;
    }

    return 0;
}